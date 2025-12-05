import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import captum.attr as ca
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "load"
OUTPUT_DIR = PROJECT_ROOT / "work" / "load" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingConfig:
    seq_len: int = 14  # history length
    horizon: int = 1  # forecast horizon
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    batch_size: int = 64
    num_epochs: int = 50
    lr: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 1
    bidirectional: bool = False
    delta_mode: str = "absolute"  # "absolute" or "relative"
    thr_up: float = 0.5
    thr_down: float = 0.5


# -------------------------------------------------------------
# Delta model + utilities
# -------------------------------------------------------------
def delta_to_class(deltas: torch.Tensor, thr_up: float, thr_down: float):
    """
    0 = fall    (delta <= -thr_down)
    1 = same    (-thr_down < delta < thr_up)
    2 = rise    (delta >= thr_up)
    """
    classes = torch.ones(deltas.shape, dtype=torch.long, device=deltas.device)
    fall_mask = deltas <= -thr_down
    rise_mask = deltas >= thr_up
    classes[fall_mask] = 0
    classes[rise_mask] = 2
    return classes


def build_delta_targets(
    y_hist: torch.Tensor,
    y_future: torch.Tensor,
    mode: str = "absolute",
):
    """
    y_hist:   (B, T_hist)   past scalar values
    y_future: (B, H)        future scalar values
    mode: "absolute" or "relative"

    h=0: delta vs last history value
    h>0: delta vs previous future step
    """
    assert y_hist.ndim == 2 and y_future.ndim == 2
    B, H = y_future.shape

    last = y_hist[:, -1]  # (B,)
    delta_true = torch.empty_like(y_future)
    eps = 1e-8

    if mode == "absolute":
        delta_true[:, 0] = y_future[:, 0] - last
        if H > 1:
            delta_true[:, 1:] = y_future[:, 1:] - y_future[:, :-1]

    elif mode == "relative":
        prev0 = torch.clamp(last, min=eps)
        delta_true[:, 0] = (y_future[:, 0] - last) / prev0

        if H > 1:
            prev = torch.clamp(y_future[:, :-1], min=eps)
            delta_true[:, 1:] = (y_future[:, 1:] - y_future[:, :-1]) / prev
    else:
        raise ValueError(mode)

    return delta_true


class GRUDeltaModel(nn.Module):
    """
    x: (B, T_in, n_features)
    outputs:
      delta_pred: (B, H)
      class_pred: (B, H) with 0=fall,1=same,2=rise
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon: int,
        n_layers: int = 1,
        bidirectional: bool = False,
        delta_mode: str = "absolute",  # "absolute" or "relative"
        thr_up: float = 0.5,
        thr_down: float = 0.5,
    ):
        super().__init__()
        self.horizon = horizon
        self.delta_mode = delta_mode
        self.thr_up = thr_up
        self.thr_down = thr_down

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        enc_dim = hidden_dim * (2 if bidirectional else 1)

        self.fc1 = nn.Linear(enc_dim, enc_dim)
        self.fc2 = nn.Linear(enc_dim, horizon)

        self._reset_parameters()

    def _reset_parameters(self):
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T_in, n_features)
        """
        enc_out, _ = self.gru(x)  # (B, T_in, E)
        h_last = enc_out[:, -1, :]  # (B, E)

        z = F.relu(self.fc1(h_last))
        delta_pred = self.fc2(z)  # (B, H)

        class_pred = delta_to_class(delta_pred, self.thr_up, self.thr_down)  # (B, H)

        return delta_pred, class_pred


def reconstruct_from_deltas(
    last_value: torch.Tensor,  # (B,)
    delta_pred: torch.Tensor,  # (B, H)
    mode: str = "absolute",
):
    """
    Rebuild y_hat_future from predicted deltas.

    mode="absolute":
        y_hat[:,0] = last + delta[0]
        y_hat[:,h] = y_hat[:,h-1] + delta[h]

    mode="relative":
        y_hat[:,0] = last * (1 + delta[0])
        y_hat[:,h] = y_hat[:,h-1] * (1 + delta[h])
    """
    B, H = delta_pred.shape
    y_hat = torch.empty_like(delta_pred)

    if mode == "absolute":
        y_hat[:, 0] = last_value + delta_pred[:, 0]
        for h in range(1, H):
            y_hat[:, h] = y_hat[:, h - 1] + delta_pred[:, h]

    elif mode == "relative":
        y_hat[:, 0] = last_value * (1.0 + delta_pred[:, 0])
        for h in range(1, H):
            y_hat[:, h] = y_hat[:, h - 1] * (1.0 + delta_pred[:, h])
    else:
        raise ValueError(mode)

    return y_hat


def delta_loss(
    delta_pred: torch.Tensor,
    delta_true: torch.Tensor,
    reduction: str = "mean",
):
    return F.mse_loss(delta_pred, delta_true, reduction=reduction)


def train_step(
    model: GRUDeltaModel,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,  # (B, T_in, n_features)
    y_hist: torch.Tensor,  # (B, T_hist)
    y_future: torch.Tensor,  # (B, H)
):
    model.train()
    optimizer.zero_grad()

    delta_true = build_delta_targets(y_hist, y_future, mode=model.delta_mode)  # (B, H)

    delta_pred, _ = model(x)

    loss = delta_loss(delta_pred, delta_true)
    loss.backward()
    optimizer.step()

    return float(loss.item())


# -------------------------------------------------------------
# Captum forward wrappers and explainer
# -------------------------------------------------------------
def class_score_from_delta(delta_h: torch.Tensor, cls: int) -> torch.Tensor:
    """
    Scalar score for a given class and horizon.

    cls=0 (fall):    lower delta is better -> score = -delta
    cls=1 (same):    delta near 0 is better -> score = -|delta|
    cls=2 (rise):    higher delta is better -> score = +delta
    """
    if cls == 0:
        return -delta_h
    elif cls == 1:
        return -delta_h.abs()
    elif cls == 2:
        return delta_h
    else:
        raise ValueError(cls)


def forward_delta(model, x):
    """
    Forward wrapper for delta explanation.

    x: (B, T_in, F)
    returns: (B, H)  -- deltas per horizon
    """
    delta_pred, _ = model(x)
    return delta_pred


def forward_class_scores(model, x):
    """
    Forward wrapper for class explanation.

    x: (B, T_in, F)
    returns: (B, 3, H)  -- smooth scores for (fall, same, rise) per horizon
    """
    delta_pred, _ = model(x)  # (B, H)

    fall = -delta_pred  # (B, H)
    same = -delta_pred.abs()  # (B, H)
    rise = delta_pred  # (B, H)

    scores = torch.stack([fall, same, rise], dim=1)  # (B, 3, H)
    return scores


def explain_instance_captum(
    model: GRUDeltaModel,
    x_single: torch.Tensor,  # (T_in, n_features)
    h_index: int,
    method: str = "saliency",  # "saliency", "integrated_gradients", "occlusion"
    target_type: str = "class",  # "class" or "delta"
    cls: int | None = None,  # 0/1/2 if target_type=="class"; None → use predicted
    baseline: torch.Tensor | float | None = None,
    n_steps_ig: int = 50,
    top_k: int = 3,
    aggregate: str = "abs",  # "abs" or "signed"
) -> dict[str, Any]:
    """
    Captum-based explanation for a single instance and horizon.

    aggregate:
        "abs"    -> sum(|attr|) over features
        "signed" -> sum(attr) over features
    """
    device = next(model.parameters()).device

    x_single = x_single.to(device)
    x0 = x_single.unsqueeze(0)  # (1, T_in, F)

    # base prediction (no grad)
    with torch.no_grad():
        delta_pred, class_pred = model(x0)
        delta_h = delta_pred[0, h_index]
        pred_cls_h = int(class_pred[0, h_index].item())

    # forward func + target
    if target_type == "delta":
        forward_func = lambda x: forward_delta(model, x)  # (B, H)
        target = h_index
        target_cls = None
    elif target_type == "class":
        forward_func = lambda x: forward_class_scores(model, x)  # (B, 3, H)
        if cls is None:
            cls = pred_cls_h
        target = (cls, h_index)  # (class_idx, horizon_idx)
        target_cls = int(cls)
    else:
        raise ValueError(f"Unknown target_type {target_type}")

    method = method.lower()
    T_in, F_in = x_single.shape

    # cuDNN RNN backward requires training mode
    was_training = model.training
    model.train()

    try:
        if method == "saliency":
            explainer = ca.Saliency(forward_func)
            x_attr = x0.clone().detach().requires_grad_(True)
            attr = explainer.attribute(x_attr, target=target)  # (1, T_in, F)

        elif method == "integrated_gradients":
            explainer = ca.IntegratedGradients(forward_func)
            if baseline is None:
                baseline_t = torch.zeros_like(x0)
            else:
                if not torch.is_tensor(baseline):
                    baseline_t = torch.full_like(x0, float(baseline))
                else:
                    baseline_t = baseline.to(device).unsqueeze(0)
            attr = explainer.attribute(
                x0,
                baselines=baseline_t,
                target=target,
                n_steps=n_steps_ig,
            )  # (1, T_in, F)

        elif method == "occlusion":
            explainer = ca.Occlusion(forward_func)
            if baseline is None:
                baseline_val = 0.0
            else:
                baseline_val = float(baseline)

            sliding_window_shapes = (1, 1, F_in)
            strides = (1, 1, F_in)

            attr = explainer.attribute(
                x0,
                strides=strides,
                sliding_window_shapes=sliding_window_shapes,
                baselines=baseline_val,
                target=target,
            )  # (1, T_in, F)

        else:
            raise ValueError(f"Unknown method {method}")
    finally:
        model.train(was_training)

    if aggregate == "abs":
        importance = attr.abs().sum(dim=-1).squeeze(0)  # (T_in,)
    elif aggregate == "signed":
        importance = attr.sum(dim=-1).squeeze(0)  # (T_in,)
    else:
        raise ValueError(f"Unknown aggregate {aggregate}")

    vals, idx = torch.topk(importance.abs(), k=min(top_k, T_in))
    top_idx = idx.cpu().tolist()
    top_vals = vals.cpu().tolist()

    return {
        "importance": importance.detach().cpu(),  # (T_in,)
        "top_indices": top_idx,
        "top_scores": top_vals,
        "pred_delta_h": float(delta_h.item()),
        "pred_class_h": pred_cls_h,
        "target_class": target_cls,
        "method": method,
        "target_type": target_type,
    }


# -------------------------------------------------------------
# Sequence creation (multi-step)
# -------------------------------------------------------------
def create_multi_step_sequences(
    series: np.ndarray,  # shape (N,)
    seq_len: int,
    horizon: int,
):
    """
    Build (X, y_hist, y_future) for multi-step forecasting.

    series: 1D array (scaled load)
    Returns:
        X:        (num_samples, seq_len, 1)
        y_hist:   (num_samples, seq_len)
        y_future: (num_samples, horizon)
    """
    X_list = []
    y_hist_list = []
    y_future_list = []

    for i in range(seq_len, len(series) - horizon + 1):
        hist = series[i - seq_len : i]  # (seq_len,)
        future = series[i : i + horizon]  # (horizon,)

        X_list.append(hist.reshape(seq_len, 1))
        y_hist_list.append(hist)
        y_future_list.append(future)

    X = np.array(X_list, dtype=np.float32)
    y_hist = np.array(y_hist_list, dtype=np.float32)
    y_future = np.array(y_future_list, dtype=np.float32)
    return X, y_hist, y_future


# -------------------------------------------------------------
# LoadForecaster
# -------------------------------------------------------------
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

        self.X_train = self.y_hist_train = self.y_future_train = None
        self.X_val = self.y_hist_val = self.y_future_val = None
        self.X_test = self.y_hist_test = self.y_future_test = None

        self.df: pd.DataFrame | None = None

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
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
        if n <= self.config.seq_len + self.config.horizon:
            raise ValueError(
                f"Not enough datapoints ({n}) for seq_len={self.config.seq_len}, "
                f"horizon={self.config.horizon}"
            )

        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        self.scaler = StandardScaler()
        self.scaler.fit(load_vals[:train_end])

        load_scaled = self.scaler.transform(load_vals).ravel()  # 1D

        seq_len = self.config.seq_len
        horizon = self.config.horizon

        # train
        train_series = load_scaled[:train_end]
        X_train, y_hist_train, y_future_train = create_multi_step_sequences(
            train_series, seq_len, horizon
        )

        # val (allow overlap from train_end - seq_len)
        val_series = load_scaled[train_end - seq_len : val_end]
        X_val, y_hist_val, y_future_val = create_multi_step_sequences(
            val_series, seq_len, horizon
        )

        # test
        test_series = load_scaled[val_end - seq_len :]
        X_test, y_hist_test, y_future_test = create_multi_step_sequences(
            test_series, seq_len, horizon
        )

        self.X_train, self.y_hist_train, self.y_future_train = (
            X_train,
            y_hist_train,
            y_future_train,
        )
        self.X_val, self.y_hist_val, self.y_future_val = (
            X_val,
            y_hist_val,
            y_future_val,
        )
        self.X_test, self.y_hist_test, self.y_future_test = (
            X_test,
            y_hist_test,
            y_future_test,
        )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    def train(self):
        if self.X_train is None:
            raise RuntimeError("Call preprocess() first.")

        input_dim = self.X_train.shape[-1]
        cfg = self.config

        self.model = GRUDeltaModel(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_size,
            horizon=cfg.horizon,
            n_layers=cfg.num_layers,
            bidirectional=cfg.bidirectional,
            delta_mode=cfg.delta_mode,
            thr_up=cfg.thr_up,
            thr_down=cfg.thr_down,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        # build dataloaders
        def make_loader(X, y_hist, y_future, shuffle: bool):
            X_t = torch.from_numpy(X).float()
            y_hist_t = torch.from_numpy(y_hist).float()
            y_future_t = torch.from_numpy(y_future).float()
            ds = torch.utils.data.TensorDataset(X_t, y_hist_t, y_future_t)
            return torch.utils.data.DataLoader(
                ds, batch_size=cfg.batch_size, shuffle=shuffle
            )

        train_loader = make_loader(
            self.X_train, self.y_hist_train, self.y_future_train, shuffle=True
        )
        val_loader = make_loader(
            self.X_val, self.y_hist_val, self.y_future_val, shuffle=False
        )

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(cfg.num_epochs):
            # train
            train_losses = []
            for X_batch, y_hist_batch, y_future_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_hist_batch = y_hist_batch.to(self.device)
                y_future_batch = y_future_batch.to(self.device)

                loss = train_step(
                    self.model, optimizer, X_batch, y_hist_batch, y_future_batch
                )
                train_losses.append(loss)

            # validation
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_hist_batch, y_future_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_hist_batch = y_hist_batch.to(self.device)
                    y_future_batch = y_future_batch.to(self.device)

                    delta_true = build_delta_targets(
                        y_hist_batch, y_future_batch, mode=self.model.delta_mode
                    )
                    delta_pred, _ = self.model(X_batch)
                    loss_val = delta_loss(delta_pred, delta_true).item()
                    val_losses.append(loss_val)

            mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
            mean_val = float(np.mean(val_losses)) if val_losses else float("nan")

            print(
                f"Epoch {epoch+1}/{cfg.num_epochs} "
                f"- train_loss={mean_train:.4f} val_loss={mean_val:.4f}"
            )

            if mean_val < best_val_loss:
                best_val_loss = mean_val
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to(self.device)

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    def evaluate(self):
        if self.model is None:
            raise RuntimeError("Call train() first.")

        cfg = self.config
        self.model.eval()

        X_t = torch.from_numpy(self.X_test).float().to(self.device)
        y_hist_t = torch.from_numpy(self.y_hist_test).float().to(self.device)
        y_future_t = torch.from_numpy(self.y_future_test).float().to(self.device)

        with torch.no_grad():
            delta_pred, class_pred = self.model(X_t)
            last_vals_scaled = y_hist_t[:, -1]  # (B,)
            y_pred_scaled = reconstruct_from_deltas(
                last_vals_scaled, delta_pred, mode=self.model.delta_mode
            )  # (B, H)

        # true scaled
        y_true_scaled = y_future_t  # (B, H)

        # inverse scale
        assert self.scaler is not None
        y_pred = self.scaler.inverse_transform(
            y_pred_scaled.cpu().numpy().reshape(-1, 1)
        ).reshape(y_pred_scaled.shape)
        y_true = self.scaler.inverse_transform(
            y_true_scaled.cpu().numpy().reshape(-1, 1)
        ).reshape(y_true_scaled.shape)

        # metrics across all horizons flattened
        diff = y_pred - y_true
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
        mape = float(np.mean(np.abs(diff) / np.maximum(np.abs(y_true), 1e-8)) * 100.0)

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
        }

        return metrics, y_true, y_pred

    # ---------------------------------------------------------
    # Artifacts
    # ---------------------------------------------------------
    def save_artifacts(self) -> Path:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Nothing to save, call train() and preprocess() first.")

        out_dir = OUTPUT_DIR / self.csv_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        artifacts = {
            "model": self.model,
            "scaler": self.scaler,
            "config": self.config,
        }

        joblib.dump(artifacts, out_dir / "artifacts.joblib")
        return out_dir


# -------------------------------------------------------------
# Top-level training loop
# -------------------------------------------------------------
def run_training_for_csv(csv_path: Path):
    forecaster = LoadForecaster(csv_path=csv_path)
    forecaster.load_dataframe()
    forecaster.preprocess()
    forecaster.train()
    metrics, y_true, y_pred = forecaster.evaluate()
    artifacts_path = forecaster.save_artifacts()
    return forecaster, metrics, y_true, y_pred, artifacts_path


def load_artifacts(csv_stem: str, device: str | None = None):
    out_dir = OUTPUT_DIR / csv_stem
    artifacts = joblib.load(out_dir / "artifacts.joblib")

    cfg: TrainingConfig = artifacts["config"]
    scaler: StandardScaler = artifacts["scaler"]
    model: GRUDeltaModel = artifacts["model"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = model.to(device)
    model.eval()

    return model, scaler, cfg


def main():
    """
    Loop over all CSV files in DATA_DIR, train one model per file, save artifacts.
    """
    results: dict[str, Any] = {}
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
            "artifacts_path": str(artifacts_path),
        }

    joblib.dump(results, OUTPUT_DIR / "results_custom.joblib")
    return results


def load_forecaster_from_disk(
    csv_stem: str, device: str | None = None
) -> LoadForecaster:
    """
    Recreate a forecaster for an already-trained CSV:
      - loads model, scaler, config from artifacts.joblib
      - reloads CSV and rebuilds sequences
      - attaches model and scaler
    """
    model, scaler, cfg = load_artifacts(csv_stem, device=device)

    csv_path = DATA_DIR / f"{csv_stem}.csv"
    forecaster = LoadForecaster(csv_path=csv_path, config=cfg, device=device)
    forecaster.load_dataframe()
    forecaster.preprocess()

    forecaster.model = model
    forecaster.scaler = scaler
    return forecaster


def plot_forecast_with_explanation(
    forecaster: LoadForecaster,
    sample_idx: int = 0,
    h_index: int = 0,  # which horizon step to explain
    method: str = "integrated_gradients",
    n_steps_ig: int = 50,
):
    """
    Single plot with:
      - history
      - true future
      - predicted future
      - attribution over lookback window for the chosen horizon h_index

    The attribution explains: "why did the model predict y_hat at horizon h_index?"
    """

    if forecaster.model is None or forecaster.scaler is None:
        raise RuntimeError("Train/load first.")

    model = forecaster.model
    scaler = forecaster.scaler
    cfg = forecaster.config
    device = forecaster.device

    seq_len = cfg.seq_len
    horizon = cfg.horizon

    # -----------------------------
    # 1) Get data (scaled) for this sample
    # -----------------------------
    hist_scaled = forecaster.y_hist_test[sample_idx]  # (seq_len,)
    fut_true_scaled = forecaster.y_future_test[sample_idx]  # (horizon,)

    X_sample_np = forecaster.X_test[sample_idx : sample_idx + 1]  # (1, T, 1)
    X_sample = torch.from_numpy(X_sample_np).float().to(device)  # (1, T, 1)

    # -----------------------------
    # 2) Forward pass -> predicted future (scaled)
    # -----------------------------
    with torch.no_grad():
        delta_pred, _ = model(X_sample)  # (1, H)
        last_val_scaled = torch.tensor(
            [hist_scaled[-1]], dtype=torch.float32, device=device
        )  # (1,)
        y_hat_scaled = reconstruct_from_deltas(
            last_val_scaled, delta_pred, mode=model.delta_mode
        )  # (1, H)
    y_hat_scaled = y_hat_scaled[0].cpu().numpy()  # (H,)
    y_true_scaled = fut_true_scaled  # (H,)

    # inverse scale everything
    hist_real = scaler.inverse_transform(hist_scaled.reshape(-1, 1)).ravel()
    y_true_real = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).ravel()
    y_hat_real = scaler.inverse_transform(y_hat_scaled.reshape(-1, 1)).ravel()

    # -----------------------------
    # 3) Captum: explain y_hat at horizon h_index wrt lookback window
    # -----------------------------
    def forward_y(x, last_vals):
        # x: (B, T, F), last_vals: (B,)
        d_pred, _ = model(x)
        y_pred = reconstruct_from_deltas(
            last_vals, d_pred, mode=model.delta_mode
        )  # (B, H)
        return y_pred

    x0 = (
        torch.from_numpy(forecaster.X_test[sample_idx : sample_idx + 1])
        .float()
        .to(device)
    )
    last_vals_t = torch.tensor(
        [hist_scaled[-1]], dtype=torch.float32, device=device
    )  # (1,)

    # cuDNN RNN backward quirk
    was_training = model.training
    model.train()

    method = method.lower()
    if method == "saliency":
        explainer = ca.Saliency(forward_y)
        x_attr = x0.clone().detach().requires_grad_(True)
        attr = explainer.attribute(
            x_attr,
            target=h_index,
            additional_forward_args=(last_vals_t,),
        )  # (1, T, F)
    elif method == "integrated_gradients":
        explainer = ca.IntegratedGradients(forward_y)
        baseline = torch.zeros_like(x0)
        attr = explainer.attribute(
            x0,
            baselines=baseline,
            target=h_index,
            additional_forward_args=(last_vals_t,),
            n_steps=n_steps_ig,
        )  # (1, T, F)
    else:
        raise ValueError(f"Unsupported method: {method}")

    model.train(was_training)

    # aggregate over features → importance per timestep
    attr = attr[0]  # (T, F)
    imp = attr.abs().sum(dim=-1).cpu().numpy()  # (T,)
    # normalize for plotting
    max_abs = np.max(imp) + 1e-8
    imp_norm = imp / max_abs

    # strongest contributor index in lookback window
    t_star = int(np.argmax(imp_norm))

    # -----------------------------
    # 4) Build unified time axis and plot
    # -----------------------------
    t_hist = np.arange(seq_len)  # 0..seq_len-1
    t_future = seq_len + np.arange(horizon)  # seq_len..seq_len+H-1

    fig, ax1 = plt.subplots(figsize=(12, 4))

    # history + futures
    ax1.plot(t_hist, hist_real, marker="o", label="History")
    ax1.plot(t_future, y_true_real, marker="o", label="True future")
    ax1.plot(t_future, y_hat_real, marker="o", linestyle="--", label="Pred future")

    ax1.axvline(seq_len - 0.5, linestyle="--", linewidth=1)  # split past/future
    ax1.set_xlabel("Time step (window + horizon)")
    ax1.set_ylabel("Load")

    # attribution on secondary axis, over history only
    ax2 = ax1.twinx()
    ax2.bar(
        t_hist,
        imp_norm,
        alpha=0.3,
        label=f"{method} importance (normalized)",
    )
    ax2.set_ylabel("Attribution weight (0–1)")

    # highlight single biggest contributor
    ax1.axvline(t_star, color="red", linestyle=":", linewidth=1)
    ax1.text(
        t_star,
        np.max(hist_real),
        f" max contrib t={t_star}",
        color="red",
        ha="center",
        va="bottom",
    )

    fig.suptitle(
        f"Sample {sample_idx} – explain horizon {h_index}\n"
        f"y_true={y_true_real[h_index]:.2f}, y_hat={y_hat_real[h_index]:.2f}"
    )

    # merged legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()
    plt.show()


def extract_and_plot_shapelets_for_class(
    forecaster: LoadForecaster,
    h_index: int,
    target_class: int,
    L: int = 24,
    method: str = "integrated_gradients",
    max_samples: int | None = None,
):
    """
    For a fixed horizon h_index and class (0=fall,1=same,2=rise):
      - find all test samples where model predicts that class at horizon h_index
      - for each such sample:
          * run IG/saliency on that horizon & class
          * take argmax over time of attribution on the history
          * cut out a length-L window around that time index
      - inverse-scale these subsequences and plot them overlaid

    Returns:
        subseqs_real: np.ndarray with shape (N_c, L_eff)
                      (N_c = number of collected subsequences,
                       L_eff = min(L, seq_len))
    """
    if forecaster.model is None or forecaster.scaler is None:
        raise RuntimeError("Forecaster must have model and scaler loaded.")

    model = forecaster.model
    model.eval()
    device = forecaster.device

    cfg = forecaster.config
    seq_len = cfg.seq_len
    L_eff = min(L, seq_len)  # effective window length

    # --- 1) get predicted classes on the whole test set (no gradients) ---
    X_test = torch.from_numpy(forecaster.X_test).float().to(device)
    with torch.no_grad():
        _, class_pred = model(X_test)  # (B, H)
    class_pred_np = class_pred.cpu().numpy().astype(int)

    B = class_pred_np.shape[0]
    indices = [i for i in range(B) if class_pred_np[i, h_index] == target_class]
    if max_samples is not None:
        indices = indices[:max_samples]

    if not indices:
        print(f"No test samples with predicted class={target_class} at h={h_index}.")
        return np.zeros((0, L_eff), dtype=float)

    subseqs_real: list[np.ndarray] = []

    for i in indices:
        # history (scaled) for this sample
        hist_scaled = forecaster.y_hist_test[i]  # (seq_len,)
        x_np = forecaster.X_test[i]  # (seq_len, 1)
        x_single = torch.from_numpy(x_np).float()

        # --- 2) explanation for this sample & horizon & class ---
        res = explain_instance_captum(
            model,
            x_single=x_single,
            h_index=h_index,
            method=method,
            target_type="class",
            cls=target_class,
            aggregate="abs",
        )
        importance = res["importance"].numpy()  # (seq_len,)

        # --- 3) pick max-attribution timestep and cut local window ---
        center = int(np.argmax(importance))
        half = L_eff // 2
        start = max(0, center - half)
        end = start + L_eff
        if end > seq_len:
            end = seq_len
            start = end - L_eff
        # safety
        start = max(0, start)
        end = min(seq_len, end)

        subseq_scaled = hist_scaled[start:end]  # (L_eff,)

        # --- 4) inverse scale to real load ---
        scaler = forecaster.scaler
        subseq_real = scaler.inverse_transform(
            subseq_scaled.reshape(-1, 1)
        ).ravel()  # (L_eff,)

        subseqs_real.append(subseq_real)

    subseqs_real = np.stack(subseqs_real, axis=0)  # (N_c, L_eff)

    # --- 5) plot all shapelet candidates overlaid + mean pattern ---
    t = np.arange(L_eff)

    plt.figure(figsize=(10, 4))
    for s in subseqs_real:
        plt.plot(t, s, alpha=0.2)
    mean_shapelet = subseqs_real.mean(axis=0)
    plt.plot(t, mean_shapelet, linewidth=2.5, label="Mean shapelet", zorder=10)

    plt.xlabel("Relative position inside shapelet window")
    plt.ylabel("Load")
    plt.title(
        f"Shapelet candidates for class={target_class}, "
        f"horizon={h_index}, N={subseqs_real.shape[0]}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    return subseqs_real
