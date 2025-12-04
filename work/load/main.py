# load_forecasting.py

from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from aix360.algorithms.tssaliency import TSSaliencyExplainer
from aix360.algorithms.tsutils.tsframe import tsFrame

import captum.attr as ca


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "load"
OUTPUT_DIR = PROJECT_ROOT / "work" / "load" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, seq_len, 1)
        # y: (N,)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(features: np.ndarray, target: np.ndarray, seq_len: int):
    """
    features: (T, 1)
    target:   (T,)
    Returns:
      X: (N, seq_len, 1)
      y: (N,)
    """
    X, y = [], []
    T = len(target)
    for i in range(T - seq_len):
        X.append(features[i : i + seq_len])
        y.append(target[i + seq_len])
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]  # (batch, hidden_size)
        y_hat = self.fc(last_hidden).squeeze(-1)  # (batch,)
        return y_hat


class SequenceModelWrapper(nn.Module):
    # Needed for Captum
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------
@dataclass
class TrainingConfig:
    seq_len: int = 7
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    batch_size: int = 64
    num_epochs: int = 30
    lr: float = 1e-3
    train_ratio: float = 0.8
    val_ratio: float = 0.1  # rest = test


# ---------------------------------------------------------------------
# Forecaster for one CSV (one series)
# ---------------------------------------------------------------------
class LoadForecaster:
    def __init__(
        self,
        csv_path: Path,
        config: TrainingConfig | None = None,
        device: str | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.config = config or TrainingConfig()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model: nn.Module | None = None
        self.scaler: StandardScaler | None = None

        self.X_train = self.y_train = None
        self.X_val = self.y_val = None
        self.X_test = self.y_test = None

        self.df: pd.DataFrame | None = None

    # -------------------------------------------------------------
    # Data
    # -------------------------------------------------------------
    def load_dataframe(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)

        df = pd.read_csv(self.csv_path)
        if "datetime" not in df.columns or "load" not in df.columns:
            raise ValueError("CSV must have columns 'datetime' and 'load'.")

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df = df.dropna(subset=["load"])

        self.df = df
        return df

    def preprocess(self):
        if self.df is None:
            raise RuntimeError("Call load_dataframe() first.")

        df = self.df

        load_vals = df["load"].values.astype(float).reshape(-1, 1)

        n = len(load_vals)
        if n <= self.config.seq_len + 2:
            raise ValueError(
                f"Not enough datapoints ({n}) for seq_len={self.config.seq_len}"
            )

        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        self.scaler = StandardScaler()
        self.scaler.fit(load_vals[:train_end])

        load_scaled = self.scaler.transform(load_vals).ravel()

        seq_len = self.config.seq_len

        # train
        X_train, y_train = create_sequences(
            load_scaled[:train_end].reshape(-1, 1),
            load_scaled[:train_end],
            seq_len,
        )

        # val (allow overlap)
        X_val, y_val = create_sequences(
            load_scaled[train_end - seq_len : val_end].reshape(-1, 1),
            load_scaled[train_end - seq_len : val_end],
            seq_len,
        )

        # test
        X_test, y_test = create_sequences(
            load_scaled[val_end - seq_len :].reshape(-1, 1),
            load_scaled[val_end - seq_len :],
            seq_len,
        )

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test

        self.model = LSTMRegressor(
            input_size=1,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

    # -------------------------------------------------------------
    # Train / evaluate
    # -------------------------------------------------------------
    def train(self):
        if self.model is None:
            raise RuntimeError("Call preprocess() first.")

        cfg = self.config

        train_ds = SeqDataset(self.X_train, self.y_train)
        val_ds = SeqDataset(self.X_val, self.y_val)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        for epoch in range(cfg.num_epochs):
            self.model.train()
            train_loss = 0.0
            for Xb, yb in train_loader:
                Xb = Xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                preds = self.model(Xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * Xb.size(0)

            train_loss /= len(train_loader.dataset)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb = Xb.to(self.device)
                    yb = yb.to(self.device)
                    preds = self.model(Xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item() * Xb.size(0)
            val_loss /= len(val_loader.dataset)

            print(
                f"[{self.csv_path.name}] Epoch {epoch+1}/{cfg.num_epochs} "
                f"| Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}"
            )

    def evaluate(self):
        if self.model is None:
            raise RuntimeError("Call preprocess() and train() first.")

        test_ds = SeqDataset(self.X_test, self.y_test)
        test_loader = DataLoader(
            test_ds, batch_size=self.config.batch_size, shuffle=False
        )

        self.model.eval()
        preds_all = []
        targets_all = []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb = Xb.to(self.device)
                yb = yb.to(self.device)
                preds = self.model(Xb)
                preds_all.append(preds.cpu().numpy())
                targets_all.append(yb.cpu().numpy())

        y_pred_scaled = np.concatenate(preds_all)
        y_true_scaled = np.concatenate(targets_all)

        # inverse transform to original units
        y_pred = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_true = self.scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).ravel()

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0

        print(f"[{self.csv_path.name}] Test MAE:  {mae:.4f}")
        print(f"[{self.csv_path.name}] Test RMSE: {rmse:.4f}")
        print(f"[{self.csv_path.name}] Test MAPE: {mape:.2f}%")

        metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
        return metrics, y_true, y_pred

    # -------------------------------------------------------------
    # Save artifacts
    # -------------------------------------------------------------
    def save_artifacts(self, artifacts_path: Path | None = None):
        if artifacts_path is None:
            stem = self.csv_path.stem
            artifacts_path = OUTPUT_DIR / f"{stem}_artifacts.joblib"

        artifacts = {
            "config": asdict(self.config),
            "csv_name": self.csv_path.name,
            "model": self.model,
            "scaler": self.scaler,
            "X_train": self.X_train,
            "y_train": self.y_train,
            "X_val": self.X_val,
            "y_val": self.y_val,
            "X_test": self.X_test,
            "y_test": self.y_test,
        }
        joblib.dump(artifacts, artifacts_path)
        print(f"[{self.csv_path.name}] Artifacts saved to {artifacts_path}")
        return artifacts_path


# ---------------------------------------------------------------------
# Captum helpers (saliency + "shapelet-like" extraction)
# ---------------------------------------------------------------------
def _load_artifacts(artifacts_path: Path | str):
    artifacts_path = Path(artifacts_path)
    if not artifacts_path.exists():
        raise FileNotFoundError(artifacts_path)
    return joblib.load(artifacts_path)


def _prepare_example_for_captum(
    artifacts, example_index: int = 0, device: str | None = None
):
    model: nn.Module = artifacts["model"]
    X_test: np.ndarray = artifacts["X_test"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model.to(device)
    model.eval()

    if example_index < 0 or example_index >= X_test.shape[0]:
        raise IndexError(
            f"example_index {example_index} out of range for X_test size {X_test.shape[0]}"
        )

    x = torch.from_numpy(X_test[example_index : example_index + 1]).float().to(device)
    x.requires_grad_(True)

    wrapper = SequenceModelWrapper(model).to(device)
    wrapper.eval()

    return wrapper, x, device


def plot_saliency_time(
    artifacts_path: Path | str,
    example_index: int = 0,
    method: str = "saliency",
    ig_steps: int = 64,
    top_k: int | None = None,
):
    """
    For a single prediction window:
    - show load(t) over the window (unscaled)
    - show saliency |d ŷ / d x_t| over the window
    - optionally highlight top_k most important time steps
    """
    artifacts = _load_artifacts(artifacts_path)
    wrapper, x, device = _prepare_example_for_captum(artifacts, example_index)
    scaler: StandardScaler = artifacts["scaler"]
    X_test: np.ndarray = artifacts["X_test"]

    if method == "saliency":
        explainer = ca.Saliency(wrapper)
        attr = explainer.attribute(x)
        title = "Saliency (gradients)"
    elif method == "ig":
        explainer = ca.IntegratedGradients(wrapper)
        baseline = torch.zeros_like(x)
        attr = explainer.attribute(x, baselines=baseline, n_steps=ig_steps)
        title = "Integrated Gradients"
    else:
        raise ValueError("method must be 'saliency' or 'ig'")

    attr = attr.detach().cpu().numpy()[0, :, 0]  # (T,)
    importance = np.abs(attr)

    x_scaled = X_test[example_index]  # (T, 1)
    x_unscaled = scaler.inverse_transform(x_scaled)[:, 0]
    T = len(x_unscaled)
    t_idx = np.arange(T)

    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(t_idx, x_unscaled, label="load", linewidth=2)
    ax1.set_xlabel("Time step in window")
    ax1.set_ylabel("Load")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(t_idx, importance, alpha=0.3, label="importance")
    ax2.set_ylabel("Attribution magnitude")

    fig.suptitle(f"{title} for one prediction window")

    if top_k is not None and top_k > 0:
        top_idx = np.argsort(-importance)[:top_k]
        for ti in top_idx:
            ax1.axvline(ti, color="red", linestyle="--", alpha=0.5)

    fig.tight_layout()
    plt.show()

    return {
        "time_indices": t_idx,
        "load_values": x_unscaled,
        "importance": importance,
    }


def extract_local_shapelet(
    artifacts_path: Path | str,
    example_index: int = 0,
    method: str = "saliency",
    ig_steps: int = 64,
    top_k: int = 5,
):
    """
    Returns the top_k time points (local shapelet) with highest attribution
    and their corresponding load values.
    """
    result = plot_saliency_time(
        artifacts_path,
        example_index=example_index,
        method=method,
        ig_steps=ig_steps,
        top_k=None,  # don't draw extra lines here
    )

    importance = result["importance"]
    load_vals = result["load_values"]
    t_idx = result["time_indices"]

    k = min(top_k, len(importance))
    top = np.argsort(-importance)[:k]
    top_sorted = np.sort(top)

    return {
        "time_indices": t_idx[top_sorted],
        "load_values": load_vals[top_sorted],
        "importance": importance[top_sorted],
    }


# ---------------------------------------------------------------------
# Training entrypoints
# ---------------------------------------------------------------------
def run_training_for_csv(csv_path: Path):
    forecaster = LoadForecaster(csv_path=csv_path)
    forecaster.load_dataframe()
    forecaster.preprocess()
    forecaster.train()
    metrics, y_true, y_pred = forecaster.evaluate()
    artifacts_path = forecaster.save_artifacts()
    return forecaster, metrics, y_true, y_pred, artifacts_path


def main():
    """
    Loop over all CSV files in DATA_DIR, train one model per file, save artifacts.
    """
    results = {}
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}")

    for csv_path in csv_files:
        print(f"=== Training model for {csv_path.name} ===")
        forecaster, metrics, y_true, y_pred, artifacts_path = run_training_for_csv(
            csv_path
        )
        results[csv_path.name] = {
            "metrics": metrics,
            "artifacts_path": artifacts_path,
        }

    joblib.dump(results, OUTPUT_DIR / "results.joblib")
    return results


if __name__ == "__main__":
    main()
