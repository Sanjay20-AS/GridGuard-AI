from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

try:
    from scipy.stats import kurtosis, skew
except ImportError:  # pragma: no cover - fallback for minimal environments
    def skew(data: np.ndarray, axis: int = 1, bias: bool = False) -> np.ndarray:
        values = np.asarray(data, dtype=float)
        mean = np.mean(values, axis=axis, keepdims=True)
        centered = values - mean
        std = np.std(values, axis=axis, keepdims=True)
        standardized = np.divide(centered, std, out=np.zeros_like(centered), where=std != 0)
        return np.mean(standardized ** 3, axis=axis)

    def kurtosis(
        data: np.ndarray,
        axis: int = 1,
        bias: bool = False,
        fisher: bool = True,
    ) -> np.ndarray:
        values = np.asarray(data, dtype=float)
        mean = np.mean(values, axis=axis, keepdims=True)
        centered = values - mean
        std = np.std(values, axis=axis, keepdims=True)
        standardized = np.divide(centered, std, out=np.zeros_like(centered), where=std != 0)
        excess = np.mean(standardized ** 4, axis=axis) - 3.0
        return excess if fisher else excess + 3.0

try:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression
except ImportError:  # pragma: no cover - fallback for minimal environments
    class LinearRegression:
        def __init__(self) -> None:
            self.coef_: np.ndarray | None = None
            self.intercept_: float | None = None

        def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearRegression":
            x = np.asarray(x, dtype=float).reshape(-1, 1)
            y = np.asarray(y, dtype=float)
            x_mean = float(np.mean(x))
            y_mean = float(np.mean(y))
            numerator = float(np.sum((x[:, 0] - x_mean) * (y - y_mean)))
            denominator = float(np.sum((x[:, 0] - x_mean) ** 2))
            slope = 0.0 if denominator == 0.0 else numerator / denominator
            self.coef_ = np.array([slope], dtype=float)
            self.intercept_ = y_mean - slope * x_mean
            return self

    class KMeans:
        def __init__(
            self,
            n_clusters: int = 8,
            random_state: int | None = None,
            n_init: int = 10,
            max_iter: int = 100,
        ) -> None:
            self.n_clusters = n_clusters
            self.random_state = random_state
            self.n_init = n_init
            self.max_iter = max_iter
            self.cluster_centers_: np.ndarray | None = None
            self.labels_: np.ndarray | None = None

        def fit(self, x: np.ndarray) -> "KMeans":
            values = np.asarray(x, dtype=float)
            if values.ndim != 2:
                raise ValueError("KMeans fallback expects a 2D array.")

            n_samples = values.shape[0]
            n_clusters = min(self.n_clusters, n_samples)
            rng = np.random.default_rng(self.random_state)
            initial_indices = rng.choice(n_samples, size=n_clusters, replace=False)
            centers = values[initial_indices].copy()

            for _ in range(self.max_iter):
                distances = _pairwise_squared_distances(values, centers)
                labels = np.argmin(distances, axis=1)
                new_centers = centers.copy()
                for cluster_index in range(n_clusters):
                    members = values[labels == cluster_index]
                    if len(members) > 0:
                        new_centers[cluster_index] = np.mean(members, axis=0)
                if np.allclose(new_centers, centers):
                    centers = new_centers
                    break
                centers = new_centers

            self.cluster_centers_ = centers
            self.labels_ = np.argmin(_pairwise_squared_distances(values, centers), axis=1)
            return self

        def fit_predict(self, x: np.ndarray) -> np.ndarray:
            return self.fit(x).labels_

RANDOM_SEED = 42
N_NEIGHBORHOOD_GROUPS = 500
DAY_COUNT = 450
QUARTER_LENGTH = 90
FEATURE_DIR = Path("data") / "processed"

FEATURE_NAMES = [
    "mean_consumption",
    "median_consumption",
    "std_consumption",
    "min_consumption",
    "max_consumption",
    "p25_consumption",
    "p75_consumption",
    "skewness_consumption",
    "kurtosis_consumption",
    "peak_to_mean_ratio",
    "coefficient_of_variation",
    "quarter_1_mean",
    "quarter_2_mean",
    "quarter_3_mean",
    "quarter_4_mean",
    "trend_slope",
    "near_zero_days",
    "longest_near_zero_streak",
    "sudden_drop_count",
    "neighborhood_mean_diff",
    "neighborhood_std_diff",
    "neighborhood_mean_anomaly_score",
]


def _pairwise_squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Compute squared Euclidean distances without materializing the full 3D tensor."""
    x_sq = np.sum(x**2, axis=1, keepdims=True)
    c_sq = np.sum(centers**2, axis=1)
    return x_sq + c_sq - 2.0 * (x @ centers.T)


def load_processed_arrays(processed_dir: Path = FEATURE_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """Load the normalized train and test consumption sequences."""
    x_train = np.load(processed_dir / "X_train.npy")
    x_test = np.load(processed_dir / "X_test.npy")
    return x_train, x_test


def compute_quarter_means(sequence: np.ndarray) -> np.ndarray:
    """Compute four quarter-style averages over 90-day windows."""
    if sequence.shape[0] >= QUARTER_LENGTH * 4:
        return np.array(
            [
                float(np.mean(sequence[i * QUARTER_LENGTH : (i + 1) * QUARTER_LENGTH]))
                for i in range(4)
            ],
            dtype=float,
        )

    chunks = np.array_split(sequence, 4)
    return np.array([float(np.mean(chunk)) if len(chunk) else 0.0 for chunk in chunks], dtype=float)


def longest_near_zero_streak(sequence: np.ndarray, threshold: float = 0.05) -> int:
    """Measure the longest consecutive run of near-zero days."""
    mask = sequence < threshold
    longest = 0
    current = 0
    for is_near_zero in mask:
        if is_near_zero:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def sudden_drop_count(sequence: np.ndarray, drop_threshold: float = 0.40) -> int:
    """Count day-over-day drops greater than the specified percentage."""
    previous = sequence[:-1]
    current = sequence[1:]
    valid = previous > 0
    drop_ratio = np.zeros_like(previous, dtype=float)
    drop_ratio[valid] = (previous[valid] - current[valid]) / previous[valid]
    return int(np.sum(drop_ratio > drop_threshold))


def compute_statistical_features(sequences: np.ndarray) -> np.ndarray:
    """Extract the statistical summaries requested for each customer."""
    means = np.mean(sequences, axis=1)
    medians = np.median(sequences, axis=1)
    stds = np.std(sequences, axis=1)
    minimums = np.min(sequences, axis=1)
    maximums = np.max(sequences, axis=1)
    p25 = np.percentile(sequences, 25, axis=1)
    p75 = np.percentile(sequences, 75, axis=1)
    skewness = skew(sequences, axis=1, bias=False)
    kurt = kurtosis(sequences, axis=1, bias=False, fisher=True)

    peak_to_mean = np.divide(maximums, means, out=np.zeros_like(maximums), where=means != 0)
    coefficient_of_variation = np.divide(stds, means, out=np.zeros_like(stds), where=means != 0)

    return np.column_stack(
        [
            means,
            medians,
            stds,
            minimums,
            maximums,
            p25,
            p75,
            np.nan_to_num(skewness, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(kurt, nan=0.0, posinf=0.0, neginf=0.0),
            peak_to_mean,
            coefficient_of_variation,
        ]
    )


def compute_temporal_features(sequences: np.ndarray) -> np.ndarray:
    """Extract temporal patterns and trend-related features."""
    quarter_features = np.vstack([compute_quarter_means(sequence) for sequence in sequences])

    day_index = np.arange(sequences.shape[1], dtype=float).reshape(-1, 1)
    slopes = np.empty(sequences.shape[0], dtype=float)
    regression = LinearRegression()
    for row_index, sequence in enumerate(sequences):
        regression.fit(day_index, sequence)
        slopes[row_index] = float(regression.coef_[0])

    near_zero_days = np.sum(sequences < 0.05, axis=1)
    streaks = np.array([longest_near_zero_streak(sequence) for sequence in sequences], dtype=float)
    sudden_drops = np.array([sudden_drop_count(sequence) for sequence in sequences], dtype=float)

    return np.column_stack(
        [
            quarter_features,
            slopes,
            near_zero_days,
            streaks,
            sudden_drops,
        ]
    )


def compute_neighborhood_features(sequences: np.ndarray, random_state: int = RANDOM_SEED) -> np.ndarray:
    """Simulate neighborhood behavior via KMeans on mean and standard deviation."""
    means = np.mean(sequences, axis=1)
    stds = np.std(sequences, axis=1)
    cluster_inputs = np.column_stack([means, stds])

    n_clusters = min(N_NEIGHBORHOOD_GROUPS, len(sequences))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(cluster_inputs)

    neighborhood_mean_diff = np.empty(len(sequences), dtype=float)
    neighborhood_std_diff = np.empty(len(sequences), dtype=float)
    neighborhood_anomaly = np.empty(len(sequences), dtype=float)

    eps = 1e-8
    for cluster_index in range(n_clusters):
        members = np.where(cluster_labels == cluster_index)[0]
        if len(members) == 0:
            continue

        cluster_means = means[members]
        cluster_stds = stds[members]
        cluster_mean_center = float(np.mean(cluster_means))
        cluster_std_center = float(np.mean(cluster_stds))
        cluster_mean_spread = float(np.std(cluster_means))

        neighborhood_mean_diff[members] = cluster_means - cluster_mean_center
        neighborhood_std_diff[members] = cluster_stds - cluster_std_center
        neighborhood_anomaly[members] = np.abs(cluster_means - cluster_mean_center) / (cluster_mean_spread + eps)

    return np.column_stack(
        [
            neighborhood_mean_diff,
            neighborhood_std_diff,
            neighborhood_anomaly,
        ]
    )


def extract_feature_matrix(sequences: np.ndarray) -> np.ndarray:
    """Combine statistical, temporal, and neighborhood features into one matrix."""
    statistical = compute_statistical_features(sequences)
    temporal = compute_temporal_features(sequences)
    neighborhood = compute_neighborhood_features(sequences, random_state=RANDOM_SEED)
    return np.hstack([statistical, temporal, neighborhood])


def save_feature_outputs(feature_matrix: np.ndarray, output_prefix: str) -> None:
    """Save feature arrays and human-readable CSVs with headers."""
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    np.save(FEATURE_DIR / f"{output_prefix}.npy", feature_matrix)

    csv_path = FEATURE_DIR / f"{output_prefix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(FEATURE_NAMES)
        writer.writerows(feature_matrix.tolist())


def run_feature_pipeline() -> Tuple[np.ndarray, np.ndarray]:
    """Load processed data, extract features, and persist the outputs."""
    x_train, x_test = load_processed_arrays()
    train_features = extract_feature_matrix(x_train)
    test_features = extract_feature_matrix(x_test)

    save_feature_outputs(train_features, "feature_train")
    save_feature_outputs(test_features, "feature_test")

    return train_features, test_features


def main() -> None:
    """Run the full feature extraction pipeline."""
    train_features, test_features = run_feature_pipeline()
    print(f"Feature train shape: {train_features.shape}")
    print(f"Feature test shape: {test_features.shape}")
    print(f"First 3 feature names: {FEATURE_NAMES[:3]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract time-series features from processed SGCC data.")
    _ = parser.parse_args()
    main()
