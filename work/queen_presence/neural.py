# queen_shapelets_end2end.py

from pathlib import Path
import joblib

import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "bee_sounds"
OUTPUT_DIR = PROJECT_ROOT / "work" / "queen_presence" / "output"

CLASS_DIRS = {
    "QueenBee Absent": 0,
    "QueenBee Present": 1,
}


# ---------------------------------------------------------
# Audio → RMS time series (same idea as earlier)
# ---------------------------------------------------------


def load_bee_audio_as_rms(
    data_dir: Path = DATA_DIR,
    frame_length: int = 2048,
    hop_length: int = 512,
    target_sr: int = 22050,
    min_duration_sec: float = 7.0,  # keep only files >= this duration
):
    """
    Load WAVs from:
        data_dir / "QueenBee Absent"
        data_dir / "QueenBee Present"

    Convert to RMS time series.

    Only keep files with duration >= min_duration_sec.
    From the remaining ones, truncate all to the shortest of those.
    """

    X_list = []
    y_list = []
    file_paths = []

    for class_name, label in CLASS_DIRS.items():
        class_path = data_dir / class_name
        wav_files = sorted(class_path.glob("*.wav"))

        for f in wav_files:
            y_audio, sr = librosa.load(f, sr=target_sr, mono=True)
            rms = librosa.feature.rms(
                y=y_audio,
                frame_length=frame_length,
                hop_length=hop_length,
                center=True,
            )[0]

            X_list.append(rms.astype(np.float32))
            y_list.append(label)
            file_paths.append(str(f))

    # minimum number of RMS frames required
    min_frames = int(np.floor(min_duration_sec * target_sr / hop_length))

    # filter out short series
    X_long = []
    y_long = []
    paths_long = []
    for x, y_lab, p in zip(X_list, y_list, file_paths):
        if len(x) >= min_frames:
            X_long.append(x)
            y_long.append(y_lab)
            paths_long.append(p)

    if len(X_long) == 0:
        raise RuntimeError(
            f"No audio longer than {min_duration_sec} s "
            f"(min_frames={min_frames}) was found."
        )

    # now compute min_len only over the kept (long) series
    lengths = [len(x) for x in X_long]
    min_len = min(lengths)

    X_trunc = [x[:min_len] for x in X_long]

    X = np.stack(X_trunc, axis=0)  # (N, T)
    y = np.array(y_long, dtype=np.int64)

    return X, y, sr, hop_length, paths_long


def z_norm_np(x: np.ndarray) -> np.ndarray:
    mu = x.mean()
    sigma = x.std()
    return (x - mu) / (sigma + 1e-8)


# ---------------------------------------------------------
# Dataset for full time series
# ---------------------------------------------------------


class BeeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, T)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------
# Shapelet layer: learned shapelets as parameters
# ---------------------------------------------------------


class ShapeletLayer(nn.Module):
    """
    Learnable shapelet layer.

    Input:
        x: (B, T) time series

    Internal:
        shapelets: (K, L) parameters

    Output:
        features: (B, K) where each feature is the min
        z-normalized Euclidean distance between one shapelet
        and any sliding window of the series.
    """

    def __init__(self, n_shapelets: int, shapelet_length: int):
        super().__init__()
        self.n_shapelets = n_shapelets
        self.shapelet_length = shapelet_length
        # initialize with small random patterns
        self.shapelets = nn.Parameter(0.01 * torch.randn(n_shapelets, shapelet_length))

    @staticmethod
    def _z_norm_torch(x, dim=-1, eps=1e-8):
        mu = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True)
        return (x - mu) / (std + eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T)
        returns: (B, K)
        """
        B, T = x.shape
        L = self.shapelet_length
        K = self.n_shapelets

        if T < L:
            raise ValueError(f"Time series length {T} < shapelet_length {L}")

        # reshape to 4D so we can use F.unfold as 1D sliding windows
        # treat time as H dimension
        x_4d = x.view(B, 1, T, 1)  # (B, C=1, H=T, W=1)

        # extract sliding windows of length L with stride 1 over time
        # kernel_size = (L, 1), stride = (1, 1)
        patches = F.unfold(
            x_4d,
            kernel_size=(L, 1),
            stride=(1, 1),
        )  # (B, L*1, T-L+1)

        # (B, T-L+1, L)
        patches = patches.transpose(1, 2)

        # z-normalize windows and shapelets
        patches_z = self._z_norm_torch(patches, dim=-1)  # (B, W, L)
        shapelets_z = self._z_norm_torch(self.shapelets, dim=-1)  # (K, L)

        # compute distances: (B, W, K)
        # patches_z: (B, W, 1, L)
        # shapelets_z: (1, 1, K, L)
        patches_exp = patches_z.unsqueeze(2)  # (B, W, 1, L)
        shapelets_exp = shapelets_z.view(1, 1, K, L)  # (1, 1, K, L)

        diff = patches_exp - shapelets_exp  # (B, W, K, L)
        dist = torch.sqrt((diff**2).mean(dim=-1) + 1e-8)  # (B, W, K)

        # min distance over windows W = T-L+1
        min_dist, _ = dist.min(dim=1)  # (B, K)

        return min_dist


# ---------------------------------------------------------
# Full model: shapelet layer + classifier head
# ---------------------------------------------------------


class ShapeletClassifier(nn.Module):
    def __init__(self, n_shapelets: int, shapelet_length: int, hidden_dim: int = 32):
        super().__init__()
        self.shapelet_layer = ShapeletLayer(n_shapelets, shapelet_length)
        self.classifier = nn.Sequential(
            nn.Linear(n_shapelets, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        feats = self.shapelet_layer(x)  # (B, K)
        logits = self.classifier(feats)  # (B, 1)
        return logits.squeeze(-1), feats  # (B,), (B, K)


# ---------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------


def main(
    test_size: float = 0.2,
    random_state: int = 0,
    n_shapelets: int = 5,
    shapelet_length: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    epochs: int = 30,
    device: str | None = None,
):
    # Load audio → RMS
    X_raw, y, sr, hop_length, file_paths = load_bee_audio_as_rms()

    # Split first (no normalization yet)
    X_train_raw, X_test_raw, y_train, y_test, paths_train, paths_test = (
        train_test_split(
            X_raw,
            y,
            file_paths,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    )

    # Per-series z-normalization, done separately for train and test
    X_train = np.stack([z_norm_np(x) for x in X_train_raw], axis=0)
    X_test = np.stack([z_norm_np(x) for x in X_test_raw], axis=0)

    train_ds = BeeSeriesDataset(X_train, y_train)
    test_ds = BeeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ShapeletClassifier(
        n_shapelets=n_shapelets,
        shapelet_length=shapelet_length,
        hidden_dim=32,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Training
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)  # (B, T)
            yb = yb.to(device)  # (B,)

            optimizer.zero_grad()
            logits, feats = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch:03d} | train loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_logits = []
    all_y = []
    all_feats = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits, feats = model(xb)
            all_logits.append(logits.cpu().numpy())
            all_y.append(yb.numpy())
            all_feats.append(feats.cpu().numpy())

    logits_all = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    feats_all = np.concatenate(all_feats, axis=0)  # (N_test, K)

    probs = 1.0 / (1.0 + np.exp(-logits_all))
    y_pred = (probs >= 0.5).astype(int)

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=3))
    try:
        auc = roc_auc_score(y_true, probs)
        print(f"ROC AUC: {auc:.3f}")
    except ValueError:
        print("ROC AUC undefined (single-class test set).")

    run_info = {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_true,
        "sr": sr,
        "hop_length": hop_length,
        "paths_train": paths_train,
        "paths_test": paths_test,
        "feats_test": feats_all,
    }
    joblib.dump(run_info, OUTPUT_DIR / "nn_shapelets.joblib")


# ---------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------


def plot_learned_shapelets(
    model: ShapeletClassifier, sr: int, hop_length: int, max_cols: int = 4
):
    shapelets = model.shapelet_layer.shapelets.detach().cpu().numpy()  # (K, L)
    K, L = shapelets.shape
    n_cols = min(max_cols, K)
    n_rows = int(np.ceil(K / n_cols))

    time_axis = np.arange(L) * hop_length / sr

    plt.figure(figsize=(4 * n_cols, 2.5 * n_rows))
    for i in range(K):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        ax.plot(time_axis, shapelets[i])
        ax.set_title(f"Shapelet {i}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("value")
    plt.tight_layout()


def plot_shapelet_importance(model: ShapeletClassifier):
    # use first Linear layer in classifier as importance proxy
    first_linear = None
    for m in model.classifier.modules():
        if isinstance(m, nn.Linear):
            first_linear = m
            break
    if first_linear is None:
        raise RuntimeError("No Linear layer found in classifier.")

    W = first_linear.weight.detach().cpu().numpy()  # (hidden_dim, K)
    importance = np.mean(np.abs(W), axis=0)  # (K,)

    idx = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(importance)), importance[idx])
    plt.xticks(np.arange(len(importance)), idx, rotation=90)
    plt.xlabel("Shapelet index")
    plt.ylabel("Importance (|w| avg)")
    plt.title("Shapelet importance")
    plt.tight_layout()


def plot_series_with_shapelet_match(
    model: ShapeletClassifier,
    X_series: np.ndarray,
    shapelet_idx: int,
    sr: int,
    hop_length: int,
):
    """
    For a fixed learned shapelet, find its best-matching window on one series
    and plot them overlaid.
    """
    series = torch.from_numpy(z_norm_np(X_series.astype(np.float32))).unsqueeze(
        0
    )  # (1, T)
    shapelets = model.shapelet_layer.shapelets.detach().cpu().numpy()  # (K, L)
    s = shapelets[shapelet_idx]
    L = s.shape[0]

    # recompute distances exactly like in ShapeletLayer, but keep argmin
    model.eval()
    with torch.no_grad():
        B, T = series.shape
        x_4d = series.view(B, 1, T, 1)
        patches = F.unfold(x_4d, kernel_size=(L, 1), stride=(1, 1))
        patches = patches.transpose(1, 2)  # (1, W, L)
        patches_z = ShapeletLayer._z_norm_torch(patches, dim=-1)
        s_t = torch.from_numpy(s).view(1, 1, L)  # (1,1,L)
        s_t = ShapeletLayer._z_norm_torch(s_t, dim=-1)
        diff = patches_z - s_t  # broadcast (1, W, L)
        dist = torch.sqrt((diff**2).mean(dim=-1) + 1e-8)  # (1, W)
        best_pos = int(torch.argmin(dist, dim=1).item())

    T = X_series.shape[0]
    time_series = np.arange(T) * hop_length / sr
    time_shapelet = (np.arange(L) + best_pos) * hop_length / sr

    plt.figure(figsize=(10, 4))
    plt.plot(time_series, z_norm_np(X_series), label="RMS z-norm")
    plt.plot(time_shapelet, s, label=f"Shapelet {shapelet_idx}", linewidth=2)
    plt.axvspan(time_shapelet[0], time_shapelet[-1], alpha=0.2, label="Matched region")
    plt.xlabel("Time [s]")
    plt.ylabel("z-RMS / shapelet value")
    plt.title(f"Series vs learned shapelet {shapelet_idx}")
    plt.legend()
    plt.tight_layout()


def plot_shapelet_explanation(
    model: ShapeletClassifier,
    X_series: np.ndarray,
    y_true: int,
    sr: int,
    hop_length: int,
    top_k_pos: int = 2,
    top_k_neg: int = 2,
    device: str | None = None,
):
    """
    Explain one sample with shapelets.

    - shows true vs predicted class
    - finds shapelets with largest positive grad (push -> class 1)
      and largest negative grad (push -> class 0)
    - for each, shows series + best-matching window + shapelet curve
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    # normalize same as training
    x_np = z_norm_np(X_series.astype(np.float32))
    xb = torch.from_numpy(x_np).unsqueeze(0).to(device)  # (1, T)
    xb.requires_grad_(True)

    # forward and keep grad on shapelet features
    logits, feats = model(xb)  # (1,), (1, K)
    feats.retain_grad()
    logit = logits[0]
    prob = torch.sigmoid(logit).item()
    pred_class = int(prob >= 0.5)

    model.zero_grad(set_to_none=True)
    logit.backward()
    grads = feats.grad.detach().cpu().numpy()[0]  # (K,)
    importance = np.abs(grads)

    shapelets = model.shapelet_layer.shapelets.detach().cpu().numpy()  # (K, L)
    K, L = shapelets.shape

    # indices pushing toward class 1 (positive grad) and class 0 (negative grad)
    pos_idx = np.where(grads > 0)[0]
    neg_idx = np.where(grads < 0)[0]

    pos_idx = pos_idx[np.argsort(importance[pos_idx])[::-1]]
    neg_idx = neg_idx[np.argsort(importance[neg_idx])[::-1]]

    pos_idx = pos_idx[: min(top_k_pos, len(pos_idx))]
    neg_idx = neg_idx[: min(top_k_neg, len(neg_idx))]

    # precompute z-normalized windows for this series once
    with torch.no_grad():
        series_t = torch.from_numpy(x_np).unsqueeze(0).to(device)  # (1, T)
        B, T = series_t.shape
        x_4d = series_t.view(B, 1, T, 1)
        patches = F.unfold(x_4d, kernel_size=(L, 1), stride=(1, 1))  # (1, L, W)
        patches = patches.transpose(1, 2)  # (1, W, L)
        patches_z = ShapeletLayer._z_norm_torch(patches, dim=-1)  # (1, W, L)

    time_series = np.arange(T) * hop_length / sr

    n_rows = (len(pos_idx) + len(neg_idx)) or 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    all_idx = list(pos_idx) + list(neg_idx)
    directions = ["→ class 1"] * len(pos_idx) + ["→ class 0"] * len(neg_idx)

    for ax, j, direction in zip(axes, all_idx, directions):
        s = shapelets[j]

        # best match position for this shapelet
        with torch.no_grad():
            s_t = torch.from_numpy(s).to(device).view(1, 1, L)  # (1,1,L)
            s_t = ShapeletLayer._z_norm_torch(s_t, dim=-1)
            diff = patches_z - s_t  # (1, W, L)
            dist = torch.sqrt((diff**2).mean(dim=-1) + 1e-8)  # (1, W)
            best_pos = int(torch.argmin(dist, dim=1).item())

        time_shapelet = (np.arange(L) + best_pos) * hop_length / sr

        ax.plot(time_series, x_np, label="RMS z-norm")
        ax.plot(time_shapelet, s, label=f"Shapelet {j}", linewidth=2)
        ax.axvspan(
            time_shapelet[0], time_shapelet[-1], alpha=0.2, label="Matched window"
        )
        ax.set_title(
            f"Shapelet {j} {direction} | grad={grads[j]:.3f} | |grad|={importance[j]:.3f}"
        )
        ax.set_ylabel("value")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(
        f"True class = {y_true}, Predicted = {pred_class}, prob(class 1) = {prob:.3f}",
        y=1.02,
    )
    plt.tight_layout()


if __name__ == "__main__":
    main()
