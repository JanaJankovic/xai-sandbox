#!/usr/bin/env python
"""
Band-wise spectrogram shapelets + Logistic Regression with majority vote.

Directory structure expected:

ROOT_DIR/
    QueenBee Absent/
        *.wav
    QueenBee Present/
        *.wav

Run:
    python bee_shapelet_spectro_bands.py
"""

import os
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib


# ---------------- PATHS ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "bee_sounds"
OUTPUT_DIR = PROJECT_ROOT / "work" / "queen_presence" / "output"


# ============================================================
# 1) AUDIO -> MEL SPECTROGRAM (BANDS x TIME)
# ============================================================


def extract_mel_spectrogram(
    path,
    sr=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=32,
):
    """
    Load audio, resample to sr, mono, and compute log-mel spectrogram.

    Returns:
        S_db: (n_mels, n_frames) float32
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        return np.zeros((n_mels, 1), dtype=np.float32)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )  # (n_mels, n_frames)
    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    return S_db


def load_bee_dataset_spectro(
    root_dir,
    class_dirs,
    max_len=None,  # optional cap on time length (frames)
    sr=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=32,
):
    """
    Read all WAVs, compute mel spectrograms, truncate in time (no padding),
    and stack into X (N, n_mels, T), y (N,).

    Returns:
        X: (N, n_mels, target_len) float32
        y: (N,) int64
        paths: list of file paths
        target_len: int (time steps per band)
    """
    specs = []
    y_list = []
    paths = []

    for label, sub_name in class_dirs.items():
        folder = os.path.join(root_dir, sub_name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")

        for fname in os.listdir(folder):
            if not fname.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                continue
            path = os.path.join(folder, fname)
            S = extract_mel_spectrogram(
                path,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
            # S: (n_mels, T)
            if S.shape[1] == 0:
                continue
            specs.append(S)
            y_list.append(label)
            paths.append(path)

    if not specs:
        raise RuntimeError("No audio files found under given directories.")

    lengths = np.array([S.shape[1] for S in specs], dtype=np.int64)
    min_len = int(lengths.min())

    if max_len is not None:
        target_len = min(min_len, int(max_len))
    else:
        target_len = min_len

    print(
        f"[Data] n_files={len(specs)}, "
        f"n_mels={n_mels}, min_len={min_len}, target_len={target_len}"
    )

    X_list = []
    y_clean = []
    paths_clean = []

    for S, label, path in zip(specs, y_list, paths):
        if S.shape[1] < target_len:
            continue
        X_list.append(S[:, :target_len].astype(np.float32))  # (n_mels, target_len)
        y_clean.append(label)
        paths_clean.append(path)

    X = np.stack(X_list, axis=0)  # (N, n_mels, target_len)
    y = np.array(y_clean, dtype=np.int64)

    print(f"[Data] Loaded {X.shape[0]} samples after truncation.")
    print(f"[Data] X shape: {X.shape} (N, bands, T)")
    return X, y, paths_clean, target_len


# ============================================================
# 2) SHAPELET UTILS (1D, USED PER BAND)
# ============================================================


def sample_random_shapelet_segments(X, n_candidates=200, shapelet_length=50, rng=None):
    """
    Randomly sample candidate shapelets from training series.

    X: (N, T)
    Returns: (n_candidates, shapelet_length)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N, T = X.shape
    if shapelet_length > T:
        raise ValueError("shapelet_length > series length")

    shapelets = []
    for _ in range(n_candidates):
        i = rng.integers(0, N)
        start = rng.integers(0, T - shapelet_length + 1)
        seg = X[i, start : start + shapelet_length]
        shapelets.append(seg.astype(np.float32))

    return np.stack(shapelets, axis=0)


def compute_windows(X, L):
    """
    Precompute sliding windows of length L for all series.

    X: (N, T)
    Returns: windows: (N, num_windows, L)
    """
    N, T = X.shape
    if L > T:
        raise ValueError("window length > series length")
    windows = sliding_window_view(X, window_shape=L, axis=1)
    return windows  # (N, num_windows, L)


def min_distances_from_windows(windows, shapelet):
    """
    Compute min distance from each series to shapelet, using precomputed windows.

    windows: (N, num_windows, L)
    shapelet: (L,)
    Returns: (N,) distances
    """
    diff = windows - shapelet.reshape(1, 1, -1)
    dists = np.sqrt(np.sum(diff * diff, axis=-1))  # (N, num_windows)
    return np.min(dists, axis=1)


def score_shapelet_for_class(distances, y, cls):
    """
    distances: (N,) distances to one candidate shapelet
    y: (N,) labels
    cls: class label we want this shapelet to represent

    Returns positive effect if class `cls` tends to have *smaller* distances
    (i.e., shapelet is closer / present in that class).
    """
    pos = distances[y == cls]
    neg = distances[y != cls]
    if len(pos) < 2 or len(neg) < 2:
        return 0.0

    mu_pos = pos.mean()
    mu_neg = neg.mean()
    var_pos = pos.var(ddof=1)
    var_neg = neg.var(ddof=1)
    n_pos = len(pos)
    n_neg = len(neg)

    pooled = ((n_pos - 1) * var_pos + (n_neg - 1) * var_neg) / (n_pos + n_neg - 2)
    pooled = np.sqrt(pooled + 1e-8)

    effect = (mu_neg - mu_pos) / pooled  # >0 if class cls closer
    return float(effect)


def learn_shapelets_per_class(
    X_train,
    y_train,
    n_shapelets_per_class=3,
    shapelet_length=40,
    candidate_factor=5,
    random_state=0,
):
    """
    Learn a *balanced* set of shapelets from 1D series X_train:

      - for each class c:
          * sample candidates only from that class
          * score them as "how much closer in class c vs others"
          * keep top `n_shapelets_per_class` for that class

    Total shapelets = n_classes * n_shapelets_per_class
    """
    rng = np.random.default_rng(random_state)
    classes = np.unique(y_train)
    all_shapelets = []

    print("[Shapelets] Precomputing windows for scoring ...")
    windows = compute_windows(X_train, shapelet_length)  # (N, num_windows, L)

    for cls in classes:
        X_c = X_train[y_train == cls]
        n_candidates = n_shapelets_per_class * candidate_factor

        print(f"[Shapelets] Class {cls}: sampling {n_candidates} candidates ...")
        candidates = sample_random_shapelet_segments(
            X_c,
            n_candidates=n_candidates,
            shapelet_length=shapelet_length,
            rng=rng,
        )

        scores = []
        print(f"[Shapelets] Class {cls}: scoring candidates ...")
        for i in range(n_candidates):
            d = min_distances_from_windows(windows, candidates[i])
            s = score_shapelet_for_class(d, y_train, cls)
            scores.append(s)
            if (i + 1) % 20 == 0 or (i + 1) == n_candidates:
                print(f"  scored {i+1}/{n_candidates}", end="\r")

        scores = np.array(scores)
        idx_sorted = np.argsort(-scores)  # biggest effect first
        top_idx = idx_sorted[:n_shapelets_per_class]
        class_shapelets = candidates[top_idx]

        print(
            f"\n[Shapelets] Class {cls}: best={scores[top_idx[0]]:.3f}, "
            f"worst_selected={scores[top_idx[-1]]:.3f}"
        )
        all_shapelets.append(class_shapelets)

    shapelets = np.vstack(all_shapelets)
    print(f"[Shapelets] Total shapelets learned: {shapelets.shape[0]}")
    return shapelets


def shapelet_transform(X, shapelets):
    """
    Compute min-distance features to all shapelets in one shot.

    X: (N, T)
    shapelets: (K, L)
    Returns: features: (N, K)
    """
    N, T = X.shape
    K, L = shapelets.shape
    if L > T:
        raise ValueError("shapelet_length > series length")

    windows = compute_windows(X, L)  # (N, num_windows, L)

    W = windows[:, np.newaxis, :, :]  # (N, 1, num_windows, L)
    S = shapelets[np.newaxis, :, np.newaxis, :]  # (1, K, 1, L)

    diff = W - S
    dists = np.sqrt(np.sum(diff * diff, axis=-1))  # (N, K, num_windows)

    features = np.min(dists, axis=2).astype(np.float32)  # (N, K)
    return features


# ============================================================
# 3) MAIN PIPELINE (BAND-WISE, MAJORITY VOTE)
# ============================================================


def majority_vote(votes):
    """
    votes: (B, N) array of integer class labels
    Returns: (N,) majority label per sample
    """
    B, N = votes.shape
    y_pred = np.empty(N, dtype=np.int64)
    for i in range(N):
        vals, counts = np.unique(votes[:, i], return_counts=True)
        y_pred[i] = vals[np.argmax(counts)]
    return y_pred


def main():
    # ------------ CONFIG ------------
    CLASS_DIRS = {
        0: "QueenBee Absent",
        1: "QueenBee Present",
    }

    model_out = OUTPUT_DIR / "shapelet_spectro_bands_model.joblib"
    test_size = 0.2
    random_state = 0

    # audio -> spectrogram
    sr = 32000
    n_fft = 512
    hop_length = 256
    n_mels = 16  # number of bands (rows) in spectrogram

    # shapelets per band
    n_shapelets_per_class = 3
    shapelet_length = 40
    candidate_factor = 5

    # ------------ LOAD DATA (SPECTROGRAMS) ------------
    X, y, paths, target_len = load_bee_dataset_spectro(
        DATA_DIR,
        CLASS_DIRS,
        max_len=None,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    # X: (N, n_mels, T)
    N, B, T = X.shape

    if shapelet_length > target_len:
        raise ValueError(
            f"shapelet_length ({shapelet_length}) > target_len ({target_len}). "
            "Reduce shapelet_length or allow longer series."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"[Data] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    band_models = []
    band_votes_test = []  # hard predictions per band
    band_proba_test = []  # probabilities per band
    band_acc = []  # band-wise accuracies

    for b in range(B):
        print(f"\n========== Band {b} / {B-1} ==========")
        Xb_train = X_train[:, b, :]  # (N_train, T)
        Xb_test = X_test[:, b, :]  # (N_test, T)

        # learn class-specific shapelets for this band
        shapelets_b = learn_shapelets_per_class(
            Xb_train,
            y_train,
            n_shapelets_per_class=n_shapelets_per_class,
            shapelet_length=shapelet_length,
            candidate_factor=candidate_factor,
            random_state=random_state + b,
        )

        # transform
        print("[Transform] band", b, "train features ...")
        Xb_train_feat = shapelet_transform(Xb_train, shapelets_b)
        print("[Transform] band", b, "test features ...")
        Xb_test_feat = shapelet_transform(Xb_test, shapelets_b)

        # scale + LR
        scaler_b = StandardScaler()
        Xb_train_scaled = scaler_b.fit_transform(Xb_train_feat)
        Xb_test_scaled = scaler_b.transform(Xb_test_feat)

        clf_b = LogisticRegression(
            max_iter=1000,
            multi_class="auto",
            solver="lbfgs",
            n_jobs=-1,
        )

        print("[Train] Fitting classifier for band", b, "...")
        clf_b.fit(Xb_train_scaled, y_train)

        # hard preds and probabilities
        yb_pred_test = clf_b.predict(Xb_test_scaled)
        yb_proba_test = clf_b.predict_proba(Xb_test_scaled)

        band_votes_test.append(yb_pred_test)
        band_proba_test.append(yb_proba_test)

        acc_b = (yb_pred_test == y_test).mean()
        band_acc.append(acc_b)

        print("[Eval] Band", b, "test performance (acc={:.3f}):".format(acc_b))
        print(classification_report(y_test, yb_pred_test))

        band_models.append(
            {
                "band_index": b,
                "shapelets": shapelets_b,
                "scaler": scaler_b,
                "classifier": clf_b,
                "shapelet_length": shapelet_length,
                "accuracy": acc_b,
            }
        )

    # ------------ ENSEMBLE: WEIGHTED SOFT VOTE OVER BANDS ------------
    band_votes_test = np.stack(band_votes_test, axis=0)  # (B, N_test)
    band_proba_test = np.stack(band_proba_test, axis=0)  # (B, N_test, n_classes)
    band_acc = np.array(band_acc)  # (B,)

    print("\n[Ensemble] Band accuracies:", band_acc)

    # normalize accuracies to sum to 1 -> weights
    # (this downweights trash bands automatically)
    weights = band_acc / (band_acc.sum() + 1e-8)  # (B,)

    # weighted average of probabilities over bands
    # result: (N_test, n_classes)
    proba_weighted = np.tensordot(weights, band_proba_test, axes=(0, 0))

    y_pred_ensemble = proba_weighted.argmax(axis=1)

    print("\n[Eval] Weighted soft-vote ensemble performance (across bands):")
    print(classification_report(y_test, y_pred_ensemble))

    # ------------ SAVE MODEL ------------
    obj = {
        "bands": band_models,
        "n_mels": n_mels,
        "target_len": target_len,
        "sr": sr,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "class_dirs": CLASS_DIRS,
        "report_ensemble": classification_report(y_test, y_pred_ensemble),
    }
    joblib.dump(obj, model_out)
    print(f"[Save] Band-wise spectro shapelet model saved to {model_out}")


if __name__ == "__main__":
    main()
