# lake_forecasting.py

from pathlib import Path
from dataclasses import dataclass, asdict
import os

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import captum.attr as ca


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "lake_oxygen"
OUTPUT_DIR = PROJECT_ROOT / "work" / "lake" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, seq_len, n_features)
        # y: (N,)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(features: np.ndarray, target: np.ndarray, seq_len: int):
    """
    features: (T, n_features)
    target:   (T,)
    Returns:
      X: (N, seq_len, n_features)
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
        input_size: int,
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
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]  # (batch, hidden_size)
        y_hat = self.fc(last_hidden).squeeze(-1)  # (batch,)
        return y_hat


# Wrapper for Captum
class SequenceModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out = self.model(x)  # (batch,)
        return out


# ---------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------
@dataclass
class TrainingConfig:
    seq_len: int = 48
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    batch_size: int = 64
    num_epochs: int = 30
    lr: float = 1e-3
    train_ratio: float = 0.8
    val_ratio: float = 0.1  # rest = test


# ---------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------
class LakeOxygenForecaster:
    def __init__(
        self,
        config: TrainingConfig | None = None,
        device: str | None = None,
        depth: float | None = None,
    ):
        self.config = config or TrainingConfig()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # depth == measuring station
        self.depth = depth

        self.model: nn.Module | None = None
        self.feature_scaler: StandardScaler | None = None
        self.target_scaler: StandardScaler | None = None
        self.feature_names: list[str] | None = None

        self.X_train = self.y_train = None
        self.X_val = self.y_val = None
        self.X_test = self.y_test = None

    # -----------------------------------------------------------------
    # Data loading / preprocessing
    # -----------------------------------------------------------------
    def load_dataframe(
        self, csv_name: str = "lake_oxygen.csv", depth: float | None = None
    ) -> pd.DataFrame:
        """
        Load CSV and optionally filter by a single Depth (measuring station).

        If depth is provided here, it overrides self.depth.
        """
        csv_path = DATA_DIR / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        # Expecting columns: Date.Time,Depth,Temp,DO,DOsat,pH,Cond
        if "Date.Time" not in df.columns:
            raise ValueError("Expected 'Date.Time' column in CSV.")
        required_cols = ["Depth", "Temp", "DO", "DOsat", "pH", "Cond"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df["Date.Time"] = pd.to_datetime(df["Date.Time"])
        df = df.sort_values("Date.Time").reset_index(drop=True)
        df = df.dropna(subset=required_cols)

        # decide which depth to use
        if depth is not None:
            self.depth = depth

        if self.depth is not None:
            # filter to a single measuring station
            df = df[df["Depth"] == self.depth].copy()
            if df.empty:
                raise ValueError(
                    f"No rows for Depth={self.depth}. "
                    f"Available depths: {df['Depth'].unique()}"
                )
        else:
            # sanity warning: mixing stations in one time series is usually wrong
            unique_depths = df["Depth"].unique()
            if len(unique_depths) > 1:
                print(
                    "Warning: multiple depths present and no depth filter set. "
                    "You are mixing measuring stations in one model. "
                    f"Depth values present: {unique_depths}"
                )

        # Optional simple time features
        df["day_of_year"] = df["Date.Time"].dt.dayofyear
        df["hour"] = df["Date.Time"].dt.hour + df["Date.Time"].dt.minute / 60.0

        df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
        df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)
        df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)

        self.df = df
        return df

    def preprocess(self):
        if not hasattr(self, "df"):
            raise RuntimeError("Call load_dataframe() first.")

        df = self.df

        # Input features include past DO as well; target is next-step DO
        feature_cols = [
            "Depth",
            "Temp",
            "DO",
            "DOsat",
            "pH",
            "Cond",
            "sin_doy",
            "cos_doy",
            "sin_hour",
            "cos_hour",
        ]
        self.feature_names = feature_cols

        features = df[feature_cols].values.astype(float)
        target = df["DO"].values.astype(float)

        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        # Scale using train portion only
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        features_train = features[:train_end]
        target_train = target[:train_end].reshape(-1, 1)

        self.feature_scaler.fit(features_train)
        self.target_scaler.fit(target_train)

        features_scaled = self.feature_scaler.transform(features)
        target_scaled = self.target_scaler.transform(target.reshape(-1, 1)).ravel()

        # Build sequences
        seq_len = self.config.seq_len

        # train
        X_train, y_train = create_sequences(
            features_scaled[:train_end], target_scaled[:train_end], seq_len
        )
        # val (allow overlap at boundary)
        X_val, y_val = create_sequences(
            features_scaled[train_end - seq_len : val_end],
            target_scaled[train_end - seq_len : val_end],
            seq_len,
        )
        # test
        X_test, y_test = create_sequences(
            features_scaled[val_end - seq_len :],
            target_scaled[val_end - seq_len :],
            seq_len,
        )

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test

        input_size = X_train.shape[-1]
        self.model = LSTMRegressor(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

    # -----------------------------------------------------------------
    # Train / evaluate
    # -----------------------------------------------------------------
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
                f"Epoch {epoch+1}/{cfg.num_epochs} | "
                f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}"
            )

    def evaluate(self):
        if self.model is None:
            raise RuntimeError("Call preprocess() and train() first.")

        test_ds = SeqDataset(self.X_test, self.y_test)
        test_loader = DataLoader(
            test_ds, batch_size=self.config.batch_size, shuffle=False
        )

        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb = Xb.to(self.device)
                yb = yb.to(self.device)
                preds = self.model(Xb)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(yb.cpu().numpy())

        y_pred_scaled = np.concatenate(all_preds)
        y_true_scaled = np.concatenate(all_targets)

        # Inverse-transform to original DO units
        y_pred = self.target_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).ravel()
        y_true = self.target_scaler.inverse_transform(
            y_true_scaled.reshape(-1, 1)
        ).ravel()

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0

        print(f"Test MAE:  {mae:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAPE: {mape:.2f}%")

        metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
        return metrics, y_true, y_pred

    # -----------------------------------------------------------------
    # Save artifacts (model, scalers, data)
    # -----------------------------------------------------------------
    def save_artifacts(self, artifacts_path: Path | None = None):
        if artifacts_path is None:
            artifacts_path = OUTPUT_DIR / "lake_oxygen_artifacts.joblib"

        artifacts = {
            "config": asdict(self.config),
            "model": self.model,
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "feature_names": self.feature_names,
            "X_train": self.X_train,
            "y_train": self.y_train,
            "X_val": self.X_val,
            "y_val": self.y_val,
            "X_test": self.X_test,
            "y_test": self.y_test,
        }
        joblib.dump(artifacts, artifacts_path)
        print(f"Artifacts saved to {artifacts_path}")


# ---------------------------------------------------------------------
# Captum explainability helpers
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
    feature_names: list[str] = artifacts["feature_names"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model.to(device)
    model.eval()

    if example_index < 0 or example_index >= X_test.shape[0]:
        raise IndexError(f"example_index out of range: {example_index}")

    x = torch.from_numpy(X_test[example_index : example_index + 1]).float().to(device)
    x.requires_grad_(True)

    wrapper = SequenceModelWrapper(model).to(device)
    wrapper.eval()

    return wrapper, x, feature_names, device


def _plot_attributions_heatmap(
    attributions: torch.Tensor, feature_names: list[str], title: str
):
    attr = attributions.detach().cpu().numpy()[0]  # (seq_len, n_features)
    seq_len, n_features = attr.shape

    plt.figure(figsize=(12, 4))
    im = plt.imshow(attr.T, aspect="auto", origin="lower")
    plt.yticks(np.arange(n_features), feature_names)
    plt.xlabel("Time step")
    plt.title(title)
    plt.colorbar(im, label="Attribution")
    plt.tight_layout()
    plt.show()


def _plot_feature_importance_bar(
    attributions: torch.Tensor, feature_names: list[str], title: str
):
    attr = attributions.detach().cpu().numpy()[0]  # (seq_len, n_features)
    # Aggregate over time using mean absolute attribution
    importance = np.mean(np.abs(attr), axis=0)

    plt.figure(figsize=(8, 4))
    idx = np.arange(len(feature_names))
    plt.bar(idx, importance)
    plt.xticks(idx, feature_names, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# --- Saliency ---
def explain_saliency(
    artifacts_path: Path | str, example_index: int = 0, device: str | None = None
):
    artifacts = _load_artifacts(artifacts_path)
    wrapper, x, feature_names, device = _prepare_example_for_captum(
        artifacts, example_index, device
    )

    saliency = ca.Saliency(wrapper)
    attributions = saliency.attribute(x)

    _plot_attributions_heatmap(attributions, feature_names, "Saliency (gradients)")
    _plot_feature_importance_bar(
        attributions, feature_names, "Saliency feature importance"
    )

    return attributions


# --- Integrated Gradients ---
def explain_integrated_gradients(
    artifacts_path: Path | str,
    example_index: int = 0,
    device: str | None = None,
    steps: int = 100,
):
    artifacts = _load_artifacts(artifacts_path)
    wrapper, x, feature_names, device = _prepare_example_for_captum(
        artifacts, example_index, device
    )

    baseline = torch.zeros_like(x)

    ig = ca.IntegratedGradients(wrapper)
    attributions = ig.attribute(x, baselines=baseline, n_steps=steps)

    _plot_attributions_heatmap(attributions, feature_names, "Integrated Gradients")
    _plot_feature_importance_bar(attributions, feature_names, "IG feature importance")

    return attributions


# --- Input X Gradient ---
def explain_input_x_gradient(
    artifacts_path: Path | str, example_index: int = 0, device: str | None = None
):
    artifacts = _load_artifacts(artifacts_path)
    wrapper, x, feature_names, device = _prepare_example_for_captum(
        artifacts, example_index, device
    )

    ixg = ca.InputXGradient(wrapper)
    attributions = ixg.attribute(x)

    _plot_attributions_heatmap(attributions, feature_names, "Input x Gradient")
    _plot_feature_importance_bar(attributions, feature_names, "IxG feature importance")

    return attributions


# --- DeepLift ---
def explain_deeplift(
    artifacts_path: Path | str, example_index: int = 0, device: str | None = None
):
    artifacts = _load_artifacts(artifacts_path)
    wrapper, x, feature_names, device = _prepare_example_for_captum(
        artifacts, example_index, device
    )

    baseline = torch.zeros_like(x)

    deeplift = ca.DeepLift(wrapper)
    attributions = deeplift.attribute(x, baselines=baseline)

    _plot_attributions_heatmap(attributions, feature_names, "DeepLift")
    _plot_feature_importance_bar(
        attributions, feature_names, "DeepLift feature importance"
    )

    return attributions


# --- Gradient SHAP ---
def explain_gradient_shap(
    artifacts_path: Path | str,
    example_index: int = 0,
    device: str | None = None,
    n_samples: int = 50,
):
    artifacts = _load_artifacts(artifacts_path)
    wrapper, x, feature_names, device = _prepare_example_for_captum(
        artifacts, example_index, device
    )

    # Baseline distribution: zeros and random noise around zero
    baseline_dist = torch.zeros_like(x).repeat(10, 1, 1)
    baseline_dist = baseline_dist + 0.01 * torch.randn_like(baseline_dist)

    grad_shap = ca.GradientShap(wrapper)
    attributions = grad_shap.attribute(
        x,
        baselines=baseline_dist,
        n_samples=n_samples,
        stdevs=0.01,
    )

    _plot_attributions_heatmap(attributions, feature_names, "GradientSHAP")
    _plot_feature_importance_bar(
        attributions, feature_names, "GradientSHAP feature importance"
    )

    return attributions


# --- Feature Ablation (perturbation-based) ---
def explain_feature_ablation(
    artifacts_path: Path | str, example_index: int = 0, device: str | None = None
):
    artifacts = _load_artifacts(artifacts_path)
    wrapper, x, feature_names, device = _prepare_example_for_captum(
        artifacts, example_index, device
    )

    fa = ca.FeatureAblation(wrapper)
    attributions = fa.attribute(x)

    _plot_attributions_heatmap(attributions, feature_names, "Feature Ablation")
    _plot_feature_importance_bar(
        attributions, feature_names, "Feature Ablation importance"
    )

    return attributions


# --- Occlusion (windowed masking over time) ---
def explain_occlusion(
    artifacts_path: Path | str,
    example_index: int = 0,
    device: str | None = None,
    sliding_window: int = 4,
):
    artifacts = _load_artifacts(artifacts_path)
    wrapper, x, feature_names, device = _prepare_example_for_captum(
        artifacts, example_index, device
    )

    # x: (1, seq_len, n_features)
    seq_len = x.shape[1]
    n_features = x.shape[2]

    sliding_window = max(1, min(sliding_window, seq_len))

    occ = ca.Occlusion(wrapper)

    # We occlude a block along the time axis, across ALL features at once.
    # For a single input tensor of shape (B, T, F):
    #   - sliding_window_shapes must be ((t_window, f_window),)
    #   - strides must be ((t_stride, f_stride),)
    # Here:
    #   - t_window = sliding_window
    #   - f_window = n_features  -> occlude all features together
    #   - t_stride = 1           -> slide one step in time
    #   - f_stride = n_features  -> only one position along feature axis
    attributions = occ.attribute(
        x,
        sliding_window_shapes=((sliding_window, n_features),),
        strides=((1, n_features),),
        baselines=0.0,
    )

    _plot_attributions_heatmap(attributions, feature_names, "Occlusion")
    _plot_feature_importance_bar(
        attributions, feature_names, "Occlusion feature importance"
    )

    return attributions


def explain_with_all_supported_methods(
    artifacts_path: Path | str, example_index: int = 0, device: str | None = None
):
    """
    Runs all supported Captum methods and plots their attributions.
    """
    explain_saliency(artifacts_path, example_index, device)
    explain_integrated_gradients(artifacts_path, example_index, device)
    explain_input_x_gradient(artifacts_path, example_index, device)
    explain_deeplift(artifacts_path, example_index, device)
    explain_gradient_shap(artifacts_path, example_index, device)
    explain_feature_ablation(artifacts_path, example_index, device)
    explain_occlusion(artifacts_path, example_index, device)


# ---------------------------------------------------------------------
# Convenience: training entrypoint
# ---------------------------------------------------------------------
def run_training(csv_name: str = "lake_oxygen.csv", depth: float | None = None):
    """
    Train a model for a single measuring station (Depth).

    depth: numeric depth value from the CSV (e.g. 0.1, 2.0, etc.)
    """
    forecaster = LakeOxygenForecaster(depth=depth)
    forecaster.load_dataframe(csv_name=csv_name, depth=depth)
    forecaster.preprocess()
    forecaster.train()
    metrics, y_true, y_pred = forecaster.evaluate()
    forecaster.save_artifacts()
    return forecaster, metrics, y_true, y_pred


def main():
    run_training(depth=2.5)
