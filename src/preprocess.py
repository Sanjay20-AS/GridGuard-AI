from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:  # pragma: no cover - fallback for minimal environments
    class MinMaxScaler:
        """Lightweight MinMaxScaler fallback compatible with sklearn's interface."""

        def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
            self.feature_range = feature_range
            self.data_min_: np.ndarray | None = None
            self.data_max_: np.ndarray | None = None
            self.data_range_: np.ndarray | None = None

        def fit(self, data: pd.DataFrame | np.ndarray) -> "MinMaxScaler":
            array = np.asarray(data, dtype=float)
            self.data_min_ = np.nanmin(array, axis=0)
            self.data_max_ = np.nanmax(array, axis=0)
            self.data_range_ = self.data_max_ - self.data_min_
            return self

        def transform(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
            if self.data_min_ is None or self.data_max_ is None or self.data_range_ is None:
                raise ValueError("This MinMaxScaler instance is not fitted yet.")

            array = np.asarray(data, dtype=float)
            lower, upper = self.feature_range
            scale = np.where(self.data_range_ == 0, 0.0, (upper - lower) / self.data_range_)
            transformed = (array - self.data_min_) * scale + lower
            transformed = np.where(self.data_range_ == 0, lower, transformed)
            return np.clip(transformed, lower, upper)

        def fit_transform(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
            return self.fit(data).transform(data)

try:
    import joblib
except ImportError:  # pragma: no cover - fallback for minimal environments
    joblib = None

RANDOM_SEED = 42
MISSING_THRESHOLD = 0.30
ATTACK_TYPES = {
    1: "scale_down",
    2: "flatline",
    3: "average",
    4: "random_noise",
}
IDENTIFIER_COLUMN_HINTS = {"cons_no", "customer_id", "customerid", "customer_no", "meter_id", "id"}


def set_random_seed(seed: int = RANDOM_SEED) -> np.random.Generator:
    """Create a deterministic random generator for reproducible preprocessing."""
    return np.random.default_rng(seed)


def resolve_input_path(input_filename: str) -> Path:
    """Resolve the SGCC input file from data/raw or an explicit path."""
    candidate = Path(input_filename)
    if candidate.exists():
        return candidate

    raw_dir = Path("data") / "raw"
    if candidate.name != input_filename:
        candidate = raw_dir / candidate.name
    else:
        candidate = raw_dir / input_filename

    if candidate.exists():
        return candidate

    if candidate.suffix.lower() != ".csv":
        csv_candidate = candidate.with_suffix(".csv")
        if csv_candidate.exists():
            return csv_candidate

    csv_files = sorted(raw_dir.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    preferred_files = [
        csv_file
        for csv_file in csv_files
        if "sgcc" in csv_file.name.lower() or "scgg" in csv_file.name.lower()
    ]
    if preferred_files:
        return max(preferred_files, key=lambda path: path.stat().st_size)

    if csv_files:
        return max(csv_files, key=lambda path: path.stat().st_size)

    raise FileNotFoundError(
        f"Could not find input file '{input_filename}'. Place the SGCC CSV in 'data/raw/' or pass an explicit path."
    )


def detect_label_column(df: pd.DataFrame) -> str:
    """Detect the label column from common SGCC naming patterns."""
    preferred_names = ["label", "Label", "target", "Target", "y", "Y", "class", "Class"]
    for column_name in preferred_names:
        if column_name in df.columns:
            return column_name

    if df.shape[1] < 2:
        raise ValueError("Input CSV must contain at least one feature column and one label column.")

    last_column = df.columns[-1]
    unique_values = set(pd.Series(df[last_column]).dropna().unique().tolist())
    if unique_values.issubset({0, 1, 0.0, 1.0, "0", "1"}):
        return last_column

    raise ValueError(
        "Could not detect the label column. Add a column named 'label' or place the binary label in the final column."
    )


def select_feature_columns(df: pd.DataFrame, label_column: str) -> list[str]:
    """Keep the daily consumption columns and drop identifier-like metadata columns."""
    feature_columns = []
    for column_name in df.columns:
        if column_name == label_column:
            continue

        lowered = column_name.strip().lower()
        if lowered in IDENTIFIER_COLUMN_HINTS or lowered.endswith("_id"):
            continue

        numeric_values = pd.to_numeric(df[column_name], errors="coerce")
        numeric_ratio = float(numeric_values.notna().mean())
        if numeric_ratio >= 0.70:
            feature_columns.append(column_name)

    if not feature_columns:
        raise ValueError("No numeric consumption columns were detected in the input file.")

    return feature_columns


def clean_missing_values(features: pd.DataFrame) -> pd.DataFrame:
    """Drop customers with too many missing values, then fill remaining gaps row-wise."""
    missing_ratio = features.isna().mean(axis=1)
    kept_features = features.loc[missing_ratio <= MISSING_THRESHOLD].copy()

    # Fill along the customer time series so each row is completed from neighboring days.
    kept_features = kept_features.ffill(axis=1).bfill(axis=1)
    kept_features = kept_features.ffill(axis=1).bfill(axis=1)
    return kept_features


def normalize_train_test(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Fit a MinMaxScaler on the training normals and apply it to both splits."""
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    return train_scaled, test_scaled, scaler


def _choose_window_length(rng: np.random.Generator, min_days: int, max_days: int) -> int:
    return int(rng.integers(min_days, max_days + 1))


def augment_theft_patterns(
    theft_readings: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic modern theft variants from existing theft samples.

    Returns
    -------
    augmented_readings : np.ndarray
        Synthetic theft samples with the same shape as the input samples.
    attack_types : np.ndarray
        Integer attack labels where 1-4 map to the four augmentation strategies.
    """
    theft_readings = np.asarray(theft_readings, dtype=float)
    augmented_samples = []
    attack_labels = []

    for sample in theft_readings:
        sample = sample.copy()
        sample_length = sample.shape[0]

        # Type 1: Scale Down Attack
        scale_factor = float(rng.uniform(0.1, 0.4))
        scale_down = np.clip(sample * scale_factor, 0.0, None)
        augmented_samples.append(scale_down)
        attack_labels.append(1)

        # Type 2: Flatline Attack
        flatline = sample.copy()
        window_length = min(_choose_window_length(rng, 30, 60), sample_length)
        start_idx = 0 if sample_length == window_length else int(rng.integers(0, sample_length - window_length + 1))
        end_idx = start_idx + window_length
        flatline[start_idx:end_idx] = rng.uniform(0.0, 0.03, size=window_length)
        augmented_samples.append(np.clip(flatline, 0.0, None))
        attack_labels.append(2)

        # Type 3: Average Attack
        average_attack = np.full(sample_length, float(np.nanmean(sample)))
        augmented_samples.append(np.clip(average_attack, 0.0, None))
        attack_labels.append(3)

        # Type 4: Random Noise Attack
        noise_reduction = rng.uniform(0.2, 0.5, size=sample_length)
        random_noise_attack = np.clip(sample * (1.0 - noise_reduction), 0.0, None)
        augmented_samples.append(random_noise_attack)
        attack_labels.append(4)

    return np.asarray(augmented_samples, dtype=float), np.asarray(attack_labels, dtype=int)


def build_processed_dataset(
    input_filename: str = "sgcc.csv",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Load, clean, augment, normalize, and split the SGCC dataset."""
    rng = set_random_seed()
    input_path = resolve_input_path(input_filename)

    raw_df = pd.read_csv(input_path)
    label_column = detect_label_column(raw_df)

    feature_columns = select_feature_columns(raw_df, label_column)
    feature_frame = raw_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    labels = pd.to_numeric(raw_df[label_column], errors="coerce")

    if labels.isna().any():
        raise ValueError("Label column contains missing or non-numeric values.")

    labels = labels.astype(int)

    cleaned_features = clean_missing_values(feature_frame)
    labels = labels.loc[cleaned_features.index].reset_index(drop=True)
    cleaned_features = cleaned_features.reset_index(drop=True)

    if cleaned_features.isna().any().any():
        raise ValueError("Missing values remain after forward fill and backward fill. Check the raw CSV structure.")

    normal_mask = labels == 0
    theft_mask = labels == 1

    normal_features = cleaned_features.loc[normal_mask].reset_index(drop=True)
    theft_features = cleaned_features.loc[theft_mask].reset_index(drop=True)

    if normal_features.empty:
        raise ValueError("No normal customers were found in the input file.")
    if theft_features.empty:
        raise ValueError("No theft-labeled customers were found in the input file.")

    normal_indices = np.arange(len(normal_features))
    rng.shuffle(normal_indices)
    split_index = int(len(normal_indices) * 0.8)
    train_indices = normal_indices[:split_index]
    test_normal_indices = normal_indices[split_index:]

    train_features = normal_features.iloc[train_indices].reset_index(drop=True)
    test_normal_features = normal_features.iloc[test_normal_indices].reset_index(drop=True)
    theft_original_features = theft_features.reset_index(drop=True)

    augmented_features, augmented_attack_types = augment_theft_patterns(
        theft_original_features.to_numpy(dtype=float),
        rng,
    )

    test_features_combined = pd.concat(
        [
            test_normal_features,
            theft_original_features,
            pd.DataFrame(augmented_features, columns=feature_columns),
        ],
        ignore_index=True,
    )

    train_scaled, test_scaled, scaler = normalize_train_test(
        train_features,
        test_features_combined,
    )

    train_labels = np.zeros(len(train_scaled), dtype=int)
    train_attack_types = np.zeros(len(train_scaled), dtype=int)

    test_labels = np.concatenate(
        [
            np.zeros(len(test_normal_features), dtype=int),
            np.ones(len(theft_original_features), dtype=int),
            np.ones(len(augmented_features), dtype=int),
        ]
    )
    test_attack_types = np.concatenate(
        [
            np.zeros(len(test_normal_features), dtype=int),
            np.zeros(len(theft_original_features), dtype=int),
            augmented_attack_types,
        ]
    )

    test_features_scaled = test_scaled

    if len(test_features_scaled) != len(test_labels):
        raise ValueError("Test feature and label counts do not match after preprocessing.")

    return (
        train_scaled,
        train_labels,
        train_attack_types,
        test_features_scaled,
        test_labels,
        test_attack_types,
        scaler,
    )


def save_outputs(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_attack_types: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    test_attack_types: np.ndarray,
    scaler: MinMaxScaler,
) -> None:
    """Persist processed data arrays and the fitted scaler."""
    processed_dir = Path("data") / "processed"
    models_dir = Path("models")
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    np.save(processed_dir / "X_train.npy", train_features)
    np.save(processed_dir / "y_train.npy", train_labels)
    np.save(processed_dir / "attack_types_train.npy", train_attack_types)
    np.save(processed_dir / "X_test.npy", test_features)
    np.save(processed_dir / "y_test.npy", test_labels)
    np.save(processed_dir / "attack_types_test.npy", test_attack_types)
    np.save(processed_dir / "attack_types.npy", test_attack_types)

    scaler_path = models_dir / "minmax_scaler.joblib"
    if joblib is not None:
        joblib.dump(scaler, scaler_path)
    else:
        with scaler_path.open("wb") as scaler_file:
            pickle.dump(scaler, scaler_file)


def print_split_summary(
    train_features: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    test_attack_types: np.ndarray,
) -> None:
    """Print the final split sizes and the test-set class distribution."""
    print(f"Train set shape: {train_features.shape}")
    print(f"Test set shape: {test_features.shape}")

    normal_count = int(np.sum(test_labels == 0))
    theft_count = int(np.sum(test_labels == 1))
    print("Test set class distribution:")
    print(f"  normal: {normal_count}")
    print(f"  theft: {theft_count}")

    for attack_type, attack_name in ATTACK_TYPES.items():
        attack_count = int(np.sum(test_attack_types == attack_type))
        print(f"  attack_type_{attack_type} ({attack_name}): {attack_count}")


def main(input_filename: str = "sgcc.csv") -> None:
    """Run the full preprocessing pipeline end to end."""
    (
        train_features,
        train_labels,
        train_attack_types,
        test_features,
        test_labels,
        test_attack_types,
        scaler,
    ) = build_processed_dataset(input_filename=input_filename)

    save_outputs(
        train_features=train_features,
        train_labels=train_labels,
        train_attack_types=train_attack_types,
        test_features=test_features,
        test_labels=test_labels,
        test_attack_types=test_attack_types,
        scaler=scaler,
    )

    print_split_summary(
        train_features=train_features,
        test_features=test_features,
        test_labels=test_labels,
        test_attack_types=test_attack_types,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the SGCC energy theft dataset.")
    parser.add_argument(
        "--input-filename",
        default="sgcc.csv",
        help="SGCC CSV filename or path. Defaults to data/raw/sgcc.csv.",
    )
    args = parser.parse_args()
    main(input_filename=args.input_filename)
