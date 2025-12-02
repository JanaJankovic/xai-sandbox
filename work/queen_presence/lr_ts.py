#!/usr/bin/env python
"""
Shapelet transform + Logistic Regression for bee audio (time-series only, no spectrogram).

Directory structure expected:

ROOT_DIR/
    QueenBee Absent/
        *.wav
    QueenBee Present/
        *.wav

Run:
    python bee_shapelet_timeseries.py
"""

import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "bee_sounds"
OUTPUT_DIR = PROJECT_ROOT / "work" / "queen_presence" / "output"
# ============================================================
# 1) AUDIO -> 1D TIME SERIES (FRAME-WISE ENERGY)
# ============================================================


def extract_energy_series(path, sr=16000, frame_length=1024, hop_length=512):
    """
    Load audio and convert to 1D time series:
      - mono waveform
      - frame-wise RMS energy -> energy(t)
    Output: 1D np.array of length n_frames
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        return np.zeros(1, dtype=np.float32)

    # rms: shape (1, n_frames) -> take [0]
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]

    return rms.astype(np.float32)


def load_bee_dataset(
    root_dir,
    class_dirs,
    max_len=None,  # optional cap on time length (frames)
    sr=16000,
    frame_length=1024,
    hop_length=512,
):
    """
    Read all WAVs, compute 1D energy series, truncate (no padding),
    and stack into X (N, T), y (N,).

    Returns:
        X: (N, target_len) float32
        y: (N,) int64
        paths: list of file paths
        target_len: int (time steps per series)
    """
    X_raw = []
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
            ts = extract_energy_series(
                path,
                sr=sr,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            if ts.shape[0] == 0:
                continue
            X_raw.append(ts)
            y_list.append(label)
            paths.append(path)

    if not X_raw:
        raise RuntimeError("No audio files found under given directories.")

    lengths = np.array([x.shape[0] for x in X_raw], dtype=np.int64)
    min_len = int(lengths.min())

    if max_len is not None:
        target_len = min(min_len, int(max_len))
    else:
        target_len = min_len

    print(f"[Data] min_len={min_len}, target_len={target_len}")

    X_list = []
    y_clean = []
    paths_clean = []

    for ts, label, path in zip(X_raw, y_list, paths):
        if ts.shape[0] < target_len:
            # Should not happen because target_len <= min_len, but safety check
            continue
        X_list.append(ts[:target_len].astype(np.float32))
        y_clean.append(label)
        paths_clean.append(path)

    X = np.stack(X_list, axis=0)  # (N, target_len)
    y = np.array(y_clean, dtype=np.int64)

    print(f"[Data] Loaded {X.shape[0]} samples after truncation.")
    print(f"[Data] Series length T = {X.shape[1]}")
    return X, y, paths_clean, target_len


# ============================================================
# 2) SHAPELET UTILS (MORE OPTIMAL, WINDOWS REUSED)
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
    # diff: (N, num_windows, L)
    diff = windows - shapelet.reshape(1, 1, -1)
    dists = np.sqrt(np.sum(diff * diff, axis=-1))  # (N, num_windows)
    return np.min(dists, axis=1)  # (N,)


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

    # positive if class `cls` has smaller distances
    effect = (mu_neg - mu_pos) / pooled
    return float(effect)


def learn_shapelets_per_class(
    X_train,
    y_train,
    n_shapelets_per_class=5,
    shapelet_length=40,
    candidate_factor=5,
    random_state=0,
):
    """
    Learn a *balanced* set of shapelets:
      - for each class c:
          * sample candidates only from that class
          * score them as "how much closer in class c vs others"
          * keep top `n_shapelets_per_class` for that class
      - stack all per-class shapelets together

    Total shapelets = n_classes * n_shapelets_per_class
    """
    rng = np.random.default_rng(random_state)
    classes = np.unique(y_train)
    all_shapelets = []

    # precompute windows over *all* X_train for scoring
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

    # windows: (N, 1, num_windows, L)
    W = windows[:, np.newaxis, :, :]
    # shapelets: (1, K, 1, L)
    S = shapelets[np.newaxis, :, np.newaxis, :]

    # squared distances: (N, K, num_windows)
    diff = W - S
    dists = np.sqrt(np.sum(diff * diff, axis=-1))

    # min over positions -> (N, K)
    features = np.min(dists, axis=2).astype(np.float32)
    return features


# ============================================================
# 3) MAIN PIPELINE
# ============================================================


def main():
    # ------------ CONFIG ------------
    CLASS_DIRS = {
        0: "QueenBee Absent",
        1: "QueenBee Present",
    }

    model_out = OUTPUT_DIR / "shapelet_model.joblib"
    test_size = 0.2
    random_state = 0

    # audio -> 1D TS
    max_len = None  # optional upper cap on frames; None = use true min length
    sr = 16000
    frame_length = 1024
    hop_length = 512

    # shapelets
    n_shapelets_per_class = 5  # total number of shapelets
    shapelet_length = 40  # in time steps (frames)
    candidate_factor = 5  # candidates = n_shapelets * candidate_factor

    # ------------ LOAD DATA ------------
    X, y, paths, target_len = load_bee_dataset(
        DATA_DIR,
        CLASS_DIRS,
        max_len=max_len,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    print(f"[Data] X shape: {X.shape}, y shape: {y.shape}")

    if shapelet_length > target_len:
        raise ValueError(
            f"shapelet_length ({shapelet_length}) > target_len ({target_len}). "
            "Reduce shapelet_length or allow longer series."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Data] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # ------------ LEARN SHAPELETS ------------
    shapelets = learn_shapelets_per_class(
        X_train,
        y_train,
        n_shapelets_per_class=n_shapelets_per_class,
        shapelet_length=shapelet_length,
        candidate_factor=candidate_factor,
        random_state=random_state,
    )
    print(f"[Shapelets] Learned shapelets shape: {shapelets.shape}")

    # ------------ TRANSFORM DATA ------------
    print("[Transform] Computing shapelet features for train ...")
    X_train_feat = shapelet_transform(X_train, shapelets)
    print("[Transform] Computing shapelet features for test ...")
    X_test_feat = shapelet_transform(X_test, shapelets)

    # ------------ SCALE + CLASSIFIER ------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    clf = LogisticRegression(
        max_iter=1000,
        multi_class="auto",
        solver="lbfgs",
        n_jobs=-1,
    )

    print("[Train] Fitting classifier ...")
    clf.fit(X_train_scaled, y_train)

    print("[Eval] Test performance:")
    y_pred = clf.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

    # ------------ SAVE MODEL ------------
    obj = {
        "shapelets": shapelets,  # (K, L)
        "scaler": scaler,
        "classifier": clf,
        "shapelet_length": shapelet_length,
        "target_len": target_len,
        "sr": sr,
        "frame_length": frame_length,
        "hop_length": hop_length,
        "class_dirs": CLASS_DIRS,
        "report": classification_report(y_test, y_pred),
    }
    joblib.dump(obj, model_out)
    print(f"[Save] Model saved to {model_out}")


if __name__ == "__main__":
    main()
