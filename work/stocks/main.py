import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from shapeletpy import ShapeletClassifier


# -----------------------------------------------------------
# 1. Data loading: read CSVs, extract aligned Close series
# -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "stocks" / "etfs"
OUTPUT_DIR = PROJECT_ROOT / "work" / "stocks" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_etf_close_series(
    data_dir: str,
    use_column: str = "Close",
    fallback_column: str = "Adj Close",
    min_length: int = 200,  # skip very short series
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Load all *.csv in data_dir.
    Returns:
        tickers: list of ETF tickers (filenames without extension)
        prices:  array (n_series, T) of aligned Close prices (last T points)
        returns: array (n_series, T-1) of log-returns
    """
    csv_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    series_list = []
    tickers = []

    for path in csv_paths:
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date")

        col = use_column if use_column in df.columns else fallback_column
        if col not in df.columns:
            continue

        s = df[col].astype(float).copy()
        s = s.replace([np.inf, -np.inf], np.nan)
        s = s.ffill().bfill().dropna()
        if len(s) < min_length:
            continue

        series_list.append(s.values)
        ticker = os.path.splitext(os.path.basename(path))[0]
        tickers.append(ticker)

    if len(series_list) == 0:
        raise RuntimeError("No usable CSV files found.")

    # Align all series to the same length: use last L points
    lengths = [len(s) for s in series_list]
    L = min(lengths)

    price_mat = np.stack([s[-L:] for s in series_list], axis=0)  # (n_series, L)

    # Log returns (one step)
    eps = 1e-8
    log_prices = np.log(price_mat + eps)
    returns = np.diff(log_prices, axis=1)  # (n_series, L-1)

    return tickers, price_mat, returns


# -----------------------------------------------------------
# 2. Preprocessing and normalization for shapelets
# -----------------------------------------------------------


def standardize_per_series(x: np.ndarray) -> np.ndarray:
    """
    x: (n_series, T) → zero-mean, unit-std per series.
    """
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mean) / std


# -----------------------------------------------------------
# 3. Random shapelet sampling (Ultra-Fast style)
# -----------------------------------------------------------


def sample_random_shapelets(
    X: Tensor,
    n_shapelets: int = 50,
    min_len: int = 10,
    max_len: int = 60,
    seed: int = 0,
) -> List[Dict]:
    """
    X: (n_series, T) standardized time series (torch.FloatTensor).
    Returns list of shapelet dicts:
        {
          "tensor": Tensor of shape (L,),
          "length": L,
          "series_idx": i,
          "start_pos": t
        }
    """
    torch.manual_seed(seed)
    n_series, T = X.shape
    max_len = min(max_len, T)

    shapelets: List[Dict] = []
    for _ in range(n_shapelets):
        Ls = int(torch.randint(low=min_len, high=max_len + 1, size=(1,)).item())
        i = int(torch.randint(low=0, high=n_series, size=(1,)).item())
        start_max = T - Ls
        if start_max <= 0:
            start = 0
        else:
            start = int(torch.randint(low=0, high=start_max + 1, size=(1,)).item())

        s = X[i, start : start + Ls].clone()
        s = (s - s.mean()) / (s.std() + 1e-8)

        shapelets.append(
            {
                "tensor": s,
                "length": Ls,
                "series_idx": i,
                "start_pos": start,
            }
        )

    return shapelets


# -----------------------------------------------------------
# 4. Shapelet transform: distance features
# -----------------------------------------------------------


def compute_shapelet_features(
    X: Tensor,
    shapelets: List[Dict],
) -> np.ndarray:
    """
    X: (n_series, T) standardized time series (torch.FloatTensor)
    shapelets: list of dicts as returned by sample_random_shapelets
    Returns:
        features: (n_series, n_shapelets) matrix of min Euclidean distance
                  to each shapelet (per series).
    """
    n_series, T = X.shape
    n_shapelets = len(shapelets)
    features = torch.empty((n_series, n_shapelets), dtype=torch.float32)

    for i in range(n_series):
        x = X[i]  # (T,)
        for j, sh in enumerate(shapelets):
            s: Tensor = sh["tensor"]
            Ls = sh["length"]

            if Ls > T:
                # Should not happen, but guard anyway
                diff = x - torch.nn.functional.pad(s, (0, T - Ls))
                features[i, j] = torch.norm(diff)
                continue

            # All sliding windows of length Ls
            windows = x.unfold(dimension=0, size=Ls, step=1)  # (T-Ls+1, Ls)
            # Normalize each window to match shapelet normalization
            w_mean = windows.mean(dim=1, keepdim=True)
            w_std = windows.std(dim=1, keepdim=True) + 1e-8
            windows_norm = (windows - w_mean) / w_std
            # Broadcast s to windows shape
            diff = windows_norm - s.view(1, -1)
            dists = torch.norm(diff, dim=1)  # (T-Ls+1,)
            features[i, j] = dists.min()

    return features.numpy()


# -----------------------------------------------------------
# 5. Clustering ETFs in shapelet space
# -----------------------------------------------------------


def cluster_etfs(
    features: np.ndarray,
    n_clusters: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """
    features: (n_series, n_shapelets)
    Returns:
        labels: cluster index per ETF (n_series,)
    """
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    labels = km.fit_predict(features)
    return labels


# -----------------------------------------------------------
# 6. ETF-level metrics: volatility and trend
# -----------------------------------------------------------


def compute_etf_metrics(
    prices: np.ndarray,
    returns: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    prices:  (n_series, T)
    returns: (n_series, T-1), log-returns
    Returns:
        {
          "volatility": per-ETF std of returns,
          "trend_slope": per-ETF OLS slope over time on log-price
        }
    """
    n_series, T = prices.shape
    vol = returns.std(axis=1)

    eps = 1e-8
    log_p = np.log(prices + eps)
    t = np.arange(T)
    # Fit slope for each series
    slopes = np.empty(n_series)
    for i in range(n_series):
        # slope from np.polyfit(t, log_price, deg=1)
        coef = np.polyfit(t, log_p[i], deg=1)
        slopes[i] = coef[0]

    return {
        "volatility": vol,
        "trend_slope": slopes,
    }


def classify_dynamic_stable(volatility: np.ndarray) -> List[str]:
    """
    Simple categorization:
        bottom 25% vol  -> 'stable'
        middle 50%      -> 'moderate'
        top 25%         -> 'dynamic'
    """
    q25 = np.quantile(volatility, 0.25)
    q75 = np.quantile(volatility, 0.75)
    labels = []
    for v in volatility:
        if v <= q25:
            labels.append("stable")
        elif v >= q75:
            labels.append("dynamic")
        else:
            labels.append("moderate")
    return labels


def classify_trend(trend_slope: np.ndarray, tol: float = 1e-4) -> List[str]:
    """
    Classify long-term direction from log-price slope.
    """
    labels = []
    for s in trend_slope:
        if s > tol:
            labels.append("growing")
        elif s < -tol:
            labels.append("falling")
        else:
            labels.append("flat")
    return labels


# -----------------------------------------------------------
# 7. Visualization utilities
# -----------------------------------------------------------


def plot_tsne_clusters(
    features: np.ndarray,
    cluster_labels: np.ndarray,
    tickers: List[str],  # kept for API compatibility, not used
    perplexity: float = 20.0,
    seed: int = 0,
):
    """
    2D t-SNE of shapelet features, colored by cluster.

    Labels only cluster names (C0, C1, ...) at the cluster centroids.
    """
    n_series = features.shape[0]
    perplexity = min(perplexity, max(5.0, n_series - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="random",
        learning_rate="auto",
    )
    emb = tsne.fit_transform(features)  # (n_series, 2)

    plt.figure(figsize=(8, 6))
    plt.scatter(emb[:, 0], emb[:, 1], c=cluster_labels, s=10, alpha=0.8)

    unique_clusters = np.unique(cluster_labels)
    for c in unique_clusters:
        idx_c = np.where(cluster_labels == c)[0]
        if len(idx_c) == 0:
            continue
        centroid = emb[idx_c].mean(axis=0)
        plt.text(
            centroid[0],
            centroid[1],
            f"C{c}",
            fontsize=14,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.7),
        )

    plt.title("ETF clusters in shapelet feature space (t-SNE)")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    plt.show()


def plot_cluster_price_profiles(
    prices: np.ndarray,
    cluster_labels: np.ndarray,
    tickers: List[str],
    max_series_per_cluster: int = 10,
):
    """
    For each cluster, plot normalized price curves from that cluster.
    """
    n_series, T = prices.shape
    t = np.arange(T)

    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 3 * n_clusters), sharex=True)
    if n_clusters == 1:
        axes = [axes]

    for ax, c in zip(axes, unique_clusters):
        idx = np.where(cluster_labels == c)[0]
        ax.set_title(f"Cluster {c} (n={len(idx)})")
        # subset for plotting
        idx = idx[:max_series_per_cluster]
        for i in idx:
            p = prices[i]
            p_norm = (p - p.mean()) / (p.std() + 1e-8)
            ax.plot(t, p_norm, alpha=0.4)
        ax.set_ylabel("norm. price")
    axes[-1].set_xlabel("time index")
    plt.tight_layout()
    plt.show()


def show_cluster_summary(
    tickers: List[str],
    cluster_labels: np.ndarray,
    volatility: np.ndarray,
    trend_slope: np.ndarray,
):
    dyn_label = classify_dynamic_stable(volatility)
    trend_label = classify_trend(trend_slope)

    n_clusters = len(np.unique(cluster_labels))
    for c in range(n_clusters):
        idx = np.where(cluster_labels == c)[0]
        print(f"\n=== Cluster {c} ===")
        print(f"Members ({len(idx)}): {', '.join(tickers[i] for i in idx)}")

        v_mean = volatility[idx].mean()
        s_mean = trend_slope[idx].mean()
        print(f"  avg volatility: {v_mean:.4f}")
        print(f"  avg trend slope (log-price): {s_mean:.6f}")

        for i in idx:
            print(
                f"    {tickers[i]}: "
                f"{dyn_label[i]} / {trend_label[i]} "
                f"(vol={volatility[i]:.4f}, slope={trend_slope[i]:.6f})"
            )


def assign_shapelets_to_clusters(
    features: np.ndarray,
    cluster_labels: np.ndarray,
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infer a cluster assignment for each shapelet from ETF clusters.

    features:       (n_series, n_shapelets) min distances
    cluster_labels: (n_series,) ETF cluster indices
    top_k:          use this many nearest ETFs (smallest distance)
                    to vote for each shapelet's cluster

    Returns:
        sh_cluster:  (n_shapelets,) inferred cluster index per shapelet
        sh_score:    (n_shapelets,) representativeness score
                     (1 / mean distance over top_k neighbors)
    """
    n_series, n_shapelets = features.shape
    sh_cluster = np.full(n_shapelets, -1, dtype=int)
    sh_score = np.zeros(n_shapelets, dtype=float)

    for j in range(n_shapelets):
        d = features[:, j]  # distances ETF→shapelet j
        idx = np.argsort(d)[: min(top_k, n_series)]
        cl = cluster_labels[idx]

        vals, counts = np.unique(cl, return_counts=True)
        best = np.argmax(counts)
        sh_cluster[j] = vals[best]

        sh_score[j] = 1.0 / (d[idx].mean() + 1e-8)

    return sh_cluster, sh_score


def plot_shapelets_by_cluster(
    shapelets: List[Dict],
    sh_cluster: np.ndarray,
    sh_score: np.ndarray,
    max_shapelets_per_cluster: int = 10,
):
    """
    Plot shapelets grouped by inferred cluster.
    Shows up to max_shapelets_per_cluster most representative
    shapelets (highest sh_score) per cluster, normalized.
    """
    unique_clusters = np.unique(sh_cluster)
    n_clusters = len(unique_clusters)

    fig, axes = plt.subplots(n_clusters, 1, figsize=(8, 2.5 * n_clusters), sharex=False)
    if n_clusters == 1:
        axes = [axes]

    for ax, c in zip(axes, unique_clusters):
        idx = np.where(sh_cluster == c)[0]
        if len(idx) == 0:
            continue

        # sort by representativeness, take top N
        order = idx[np.argsort(sh_score[idx])[::-1]]
        order = order[:max_shapelets_per_cluster]

        for j in order:
            s = shapelets[j]["tensor"].detach().cpu().numpy()
            s = (s - s.mean()) / (s.std() + 1e-8)
            t = np.arange(len(s))
            ax.plot(t, s, alpha=0.5)

        ax.set_title(f"Shapelets associated with cluster {c} (n={len(idx)})")
        ax.set_ylabel("norm. value")

    axes[-1].set_xlabel("shapelet time index")
    plt.tight_layout()
    plt.show()


def plot_representative_price_profiles(
    prices: np.ndarray,
    features: np.ndarray,
    cluster_labels: np.ndarray,
    tickers: List[str],
    n_per_cluster: int = 3,
):
    """
    For each cluster, plot `n_per_cluster` most representative ETFs
    (closest to cluster centroid in *feature* space), with labels.

    prices:         (n_series, T)   raw (or aligned) price matrix
    features:       (n_series, F)   shapelet feature matrix
    cluster_labels: (n_series,)     cluster index per ETF
    tickers:        list of ETF names, len = n_series
    """
    n_series, T = prices.shape
    t = np.arange(T)

    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 3 * n_clusters), sharex=True)
    if n_clusters == 1:
        axes = [axes]

    for ax, c in zip(axes, unique_clusters):
        idx_c = np.where(cluster_labels == c)[0]
        if len(idx_c) == 0:
            ax.set_visible(False)
            continue

        # centroid in feature space
        centroid = features[idx_c].mean(axis=0)
        feats_c = features[idx_c]

        # distances to centroid → most "common" / representative
        dists = np.linalg.norm(feats_c - centroid, axis=1)
        order = idx_c[np.argsort(dists)]
        top_idx = order[: min(n_per_cluster, len(order))]

        for j in top_idx:
            p = prices[j]
            p_norm = (p - p.mean()) / (p.std() + 1e-8)
            ax.plot(t, p_norm, label=tickers[j])

        ax.set_title(f"Cluster {c} – {len(top_idx)} most representative ETFs")
        ax.set_ylabel("norm. price")
        ax.legend(fontsize=8, loc="best")

    axes[-1].set_xlabel("time index")
    plt.tight_layout()
    plt.show()


def plot_representative_by_vol_trend(
    prices: np.ndarray,
    features: np.ndarray,
    cluster_labels: np.ndarray,
    tickers: List[str],
    volatility: np.ndarray,
    trend_slope: np.ndarray,
    n_per_combo: int = 3,
):
    """
    For each (volatility_class, trend_class) combination, plot up to
    `n_per_combo` most representative ETFs (closest to feature centroid
    of that combination). Each subplot = one combination.

    Line label: "<TICKER> (C<cluster_id>)".
    """
    # 1) Classify by volatility + trend
    dyn_label = np.array(
        classify_dynamic_stable(volatility)
    )  # 'stable'/'moderate'/'dynamic'
    trend_label = np.array(classify_trend(trend_slope))  # 'growing'/'flat'/'falling'

    n_series, T = prices.shape
    t = np.arange(T)

    # 2) Determine which (dyn, trend) combos actually exist
    combos = []
    for d in ["stable", "moderate", "dynamic"]:
        for tr in ["growing", "flat", "falling"]:
            mask = (dyn_label == d) & (trend_label == tr)
            if mask.any():
                combos.append((d, tr))

    if not combos:
        print("No volatility/trend combinations found.")
        return

    n_combos = len(combos)

    # 3) Prepare subplots
    fig, axes = plt.subplots(n_combos, 1, figsize=(10, 3 * n_combos), sharex=True)
    if n_combos == 1:
        axes = [axes]

    # 4) For each combo, pick n_per_combo most representative in feature space
    for ax, (d, tr) in zip(axes, combos):
        mask = (dyn_label == d) & (trend_label == tr)
        idx_combo = np.where(mask)[0]
        if len(idx_combo) == 0:
            ax.set_visible(False)
            continue

        feats_c = features[idx_combo]
        centroid = feats_c.mean(axis=0)
        dists = np.linalg.norm(feats_c - centroid, axis=1)
        order = idx_combo[np.argsort(dists)]
        top_idx = order[: min(n_per_combo, len(order))]

        for j in top_idx:
            p = prices[j]
            p_norm = (p - p.mean()) / (p.std() + 1e-8)
            ax.plot(t, p_norm, label=f"{tickers[j]} (C{cluster_labels[j]})")

        ax.set_title(
            f"{d.capitalize()} & {tr.capitalize()} – "
            f"{len(top_idx)} most representative ETFs (total n={len(idx_combo)})"
        )
        ax.set_ylabel("norm. price")
        ax.legend(fontsize=8, loc="best")

    axes[-1].set_xlabel("time index")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# 8. Main orchestration
# -----------------------------------------------------------


def main():
    n_shapelets = 100
    min_len = 40
    max_len = 80
    n_clusters = 8
    seed = 42

    # 1) Load ETF prices
    tickers, price_mat, returns = load_etf_close_series(DATA_DIR)

    # 2) Prepare series for shapelets (standardized log-returns)
    X_np = standardize_per_series(returns)
    X = torch.from_numpy(X_np).float()

    # 3) Sample random shapelets
    shapelets = sample_random_shapelets(
        X,
        n_shapelets=n_shapelets,
        min_len=min_len,
        max_len=max_len,
        seed=seed,
    )

    # 4) Shapelet transform
    features = compute_shapelet_features(X, shapelets)

    # 5) Cluster ETFs
    cluster_labels = cluster_etfs(features, n_clusters=n_clusters, seed=seed)

    # 6) ETF metrics (dynamic / stable, growing / falling)
    metrics = compute_etf_metrics(price_mat, returns)
    volatility = metrics["volatility"]
    trend_slope = metrics["trend_slope"]

    # 7) Persist everything needed for visualization/analysis
    results = {
        "tickers": tickers,
        "price_mat": price_mat,
        "returns": returns,
        "X_np": X_np,
        "shapelets": shapelets,
        "features": features,
        "cluster_labels": cluster_labels,
        "volatility": volatility,
        "trend_slope": trend_slope,
        "n_shapelets": n_shapelets,
        "min_len": min_len,
        "max_len": max_len,
        "n_clusters": n_clusters,
        "seed": seed,
    }

    joblib.dump(results, OUTPUT_DIR / "etf_shapelet_results.joblib")
    print(f"Saved results to {OUTPUT_DIR / "etf_shapelet_results.joblib"}")


if __name__ == "__main__":
    main()
