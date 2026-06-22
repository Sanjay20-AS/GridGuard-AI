from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


def _resolve_base_dir() -> Path:
    """Resolve the repository root in local runs and notebooks."""
    if "__file__" in globals():
        return Path(__file__).resolve().parents[1]
    return Path.cwd()


BASE_DIR = _resolve_base_dir()

FEATURE_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
DASHBOARD_DIR = BASE_DIR / "dashboard"
SURROGATE_MODEL_PATH = MODEL_DIR / "surrogate_clf.pkl"
SHAP_VALUES_PATH = FEATURE_DIR / "shap_values.npy"
SHAP_SUMMARY_PATH = DASHBOARD_DIR / "shap_summary.png"
SHAP_BAR_PATH = DASHBOARD_DIR / "shap_bar.png"
FEATURE_NAMES_PATH = FEATURE_DIR / "feature_test.csv"
RECONSTRUCTION_ERRORS_PATH = FEATURE_DIR / "test_errors.npy"
ATTACK_TYPES_PATH = FEATURE_DIR / "attack_types.npy"

FEATURE_CACHE: np.ndarray | None = None
LABEL_CACHE: np.ndarray | None = None
SHAP_CACHE: np.ndarray | None = None
MODEL_CACHE: RandomForestClassifier | None = None
FEATURE_NAMES_CACHE: list[str] | None = None

ATTACK_TYPE_NAMES = {
    1: "scale_down",
    2: "flatline",
    3: "average",
    4: "random_noise",
}


def _balanced_attack_sample_indices(
    attack_types: np.ndarray,
    sample_per_attack: int = 100,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Select a balanced sample of attack records for faster SHAP computation."""
    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []

    for attack_type in sorted(ATTACK_TYPE_NAMES):
        attack_indices = np.where(attack_types == attack_type)[0]
        if len(attack_indices) == 0:
            continue

        sample_size = min(sample_per_attack, len(attack_indices))
        chosen = rng.choice(attack_indices, size=sample_size, replace=False)
        selected_indices.extend(chosen.tolist())

    if not selected_indices:
        raise ValueError("No attack samples were found for explainability analysis.")

    return np.asarray(selected_indices)


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducible surrogate training and SHAP output."""
    random.seed(seed)
    np.random.seed(seed)


def load_feature_matrix() -> np.ndarray:
    """Load the engineered feature matrix used for surrogate training."""
    return np.load(FEATURE_DIR / "feature_test.npy")


def load_labels() -> np.ndarray:
    """Load the corresponding normal/theft labels."""
    return np.load(FEATURE_DIR / "y_test.npy")


def load_attack_types() -> np.ndarray:
    """Load attack-type labels for the full test set."""
    return np.load(ATTACK_TYPES_PATH)


def load_aligned_test_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the test feature matrix, labels, and attack types with strict alignment checks."""
    features = load_feature_matrix()
    labels = load_labels()
    attack_types = load_attack_types()

    if len(features) != len(labels) or len(features) != len(attack_types):
        raise ValueError(
            "feature_test.npy, y_test.npy, and attack_types.npy must have the same length for explainability analysis."
        )

    return features, labels, attack_types


def load_feature_names(expected_count: int | None = None) -> list[str]:
    """Load human-readable feature names from the saved CSV header."""
    global FEATURE_NAMES_CACHE

    if FEATURE_NAMES_CACHE is None:
        feature_frame = pd.read_csv(FEATURE_NAMES_PATH, nrows=0)
        FEATURE_NAMES_CACHE = list(feature_frame.columns)

    if expected_count is not None and len(FEATURE_NAMES_CACHE) != expected_count:
        raise ValueError(
            f"Feature name count ({len(FEATURE_NAMES_CACHE)}) does not match feature matrix width ({expected_count})."
        )

    return FEATURE_NAMES_CACHE


def load_reconstruction_errors() -> np.ndarray:
    """Load the reconstruction errors produced by the LSTM autoencoder."""
    return np.load(RECONSTRUCTION_ERRORS_PATH)


def build_surrogate_classifier(random_state: int = RANDOM_SEED) -> RandomForestClassifier:
    """Create a tree-based surrogate model compatible with SHAP TreeExplainer."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def train_surrogate_classifier(
    features: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> Tuple[RandomForestClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train the surrogate classifier and report performance on a held-out split."""
    global FEATURE_CACHE, LABEL_CACHE, MODEL_CACHE

    set_random_seed(RANDOM_SEED)
    features = load_feature_matrix() if features is None else features
    labels = load_labels() if labels is None else labels

    FEATURE_CACHE = features
    LABEL_CACHE = labels

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    model = build_surrogate_classifier()
    model.fit(x_train, y_train)
    predictions = model.predict(x_holdout)

    print("Surrogate classifier classification report (held-out split):")
    print(classification_report(y_holdout, predictions, digits=4, zero_division=0))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SURROGATE_MODEL_PATH)

    MODEL_CACHE = model
    return model, x_train, x_holdout, y_train, y_holdout


def load_surrogate_classifier() -> RandomForestClassifier:
    """Load the saved surrogate classifier, training it first if needed."""
    global MODEL_CACHE

    if MODEL_CACHE is not None:
        return MODEL_CACHE

    if SURROGATE_MODEL_PATH.exists():
        MODEL_CACHE = joblib.load(SURROGATE_MODEL_PATH)
        return MODEL_CACHE

    model, *_ = train_surrogate_classifier()
    return model


def _extract_positive_class_shap_values(explainer: shap.TreeExplainer, x_data: np.ndarray) -> np.ndarray:
    """Return SHAP values for the positive class in binary classification."""
    raw_values = explainer.shap_values(x_data)
    if isinstance(raw_values, list):
        if len(raw_values) == 1:
            return np.asarray(raw_values[0])
        return np.asarray(raw_values[1])

    raw_array = np.asarray(raw_values)
    if raw_array.ndim == 3:
        return raw_array[:, :, 1]
    return raw_array


def compute_tree_shap_values(
    model: RandomForestClassifier | None = None,
    features: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """Compute TreeExplainer SHAP values for the surrogate classifier."""
    global SHAP_CACHE

    model = load_surrogate_classifier() if model is None else model
    features = load_feature_matrix() if features is None else features
    feature_names = load_feature_names(features.shape[1]) if feature_names is None else feature_names

    explainer = shap.TreeExplainer(model)
    shap_values = _extract_positive_class_shap_values(explainer, features)

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SHAP_VALUES_PATH, shap_values)
    SHAP_CACHE = shap_values
    return shap_values


def _load_or_compute_aligned_shap_values() -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return aligned test features, attack types, SHAP values, and feature names."""
    features, _, attack_types = load_aligned_test_data()
    feature_names = load_feature_names(features.shape[1])
    shap_values = load_shap_values()

    if shap_values.shape[0] != features.shape[0]:
        model = load_surrogate_classifier()
        shap_values = compute_tree_shap_values(model, features, feature_names)
        np.save(SHAP_VALUES_PATH, shap_values)

    if shap_values.shape[0] != features.shape[0]:
        raise ValueError(
            "SHAP values must align with feature_test.npy; recompute shap_values.npy from the full test set."
        )

    return features, attack_types, shap_values, feature_names


def save_shap_plots(
    shap_values: np.ndarray | None = None,
    features: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> None:
    import os
    import matplotlib

    matplotlib.use("Agg")

    dashboard_path = DASHBOARD_DIR
    os.makedirs(dashboard_path, exist_ok=True)

    shap_values = load_shap_values() if shap_values is None else shap_values
    features = load_feature_matrix() if features is None else features
    feature_names = load_feature_names(features.shape[1]) if feature_names is None else feature_names

    # Handle RF binary classification output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    features_plot = features[: len(shap_values)]

    # Summary plot
    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        features_plot,
        feature_names=feature_names,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        SHAP_SUMMARY_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # Bar plot
    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        features_plot,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        SHAP_BAR_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print("Saved:", SHAP_SUMMARY_PATH)
    print("Saved:", SHAP_BAR_PATH)


def load_shap_values() -> np.ndarray:
    """Load cached SHAP values or compute them if necessary."""
    global SHAP_CACHE

    if SHAP_CACHE is not None:
        return SHAP_CACHE

    if SHAP_VALUES_PATH.exists():
        SHAP_CACHE = np.load(SHAP_VALUES_PATH)
        return SHAP_CACHE

    return compute_tree_shap_values()


def print_attack_type_analysis(
    shap_values: np.ndarray | None = None,
    attack_types: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> None:
    """Print the top three SHAP features for each synthetic theft attack type."""
    if shap_values is None or attack_types is None or feature_names is None:
        features, attack_types, shap_values, feature_names = _load_or_compute_aligned_shap_values()

    if len(shap_values) != len(attack_types):
        raise ValueError("Attack-type analysis inputs must have the same number of rows.")

    for attack_type, attack_name in ATTACK_TYPE_NAMES.items():
        mask = attack_types == attack_type
        if not np.any(mask):
            print(f"Attack type {attack_type} ({attack_name}): no samples found")
            continue

        group_mean = np.mean(np.abs(shap_values[mask]), axis=0)
        top_indices = np.argsort(group_mean)[::-1][:3]
        top_features = [feature_names[index] for index in top_indices]
        print(f"Attack type {attack_type} ({attack_name}) top features: {', '.join(top_features)}")


def explain_customer(customer_index: int, model: RandomForestClassifier | None = None) -> Dict[str, object]:
    """Explain a single test customer using feature values, reconstruction error, and SHAP values."""
    features, labels, attack_types = load_aligned_test_data()
    shap_values = load_shap_values()
    reconstruction_errors = load_reconstruction_errors()
    feature_names = load_feature_names(features.shape[1])

    if shap_values.shape[0] != features.shape[0]:
        model = load_surrogate_classifier() if model is None else model
        shap_values = compute_tree_shap_values(model, features, feature_names)

    if customer_index < 0 or customer_index >= len(features):
        raise IndexError(f"Customer index {customer_index} is out of range for the test set of size {len(features)}.")

    model = load_surrogate_classifier() if model is None else model
    customer_features = features[customer_index]
    customer_shap = shap_values[customer_index]
    customer_error = float(reconstruction_errors[customer_index])
    prediction = int(model.predict(customer_features.reshape(1, -1))[0])
    prediction_label = "theft" if prediction == 1 else "normal"

    top_indices = np.argsort(np.abs(customer_shap))[::-1][:5]
    top_shap_features = [
        {
            "feature": feature_names[index],
            "shap_value": float(customer_shap[index]),
            "feature_value": float(customer_features[index]),
        }
        for index in top_indices
    ]

    print(f"Customer {customer_index} predicted as: {prediction_label}")
    print(f"Reconstruction error: {customer_error:.6f}")
    print("Top 5 SHAP features:")
    for item in top_shap_features:
        print(f"  {item['feature']}: shap={item['shap_value']:.6f}, value={item['feature_value']:.6f}")

    return {
        "customer_index": customer_index,
        "features": {feature_names[index]: float(customer_features[index]) for index in range(len(feature_names))},
        "reconstruction_error": customer_error,
        "prediction": prediction_label,
        "top_shap_features": top_shap_features,
        "actual_label": int(labels[customer_index]),
        "attack_type": int(attack_types[customer_index]),
    }


def run_explainability_pipeline(sample_per_attack: int | None = 100) -> None:
    """Train the surrogate, compute SHAP values, generate plots, and print attack summaries."""
    model, *_ = train_surrogate_classifier()
    features = load_feature_matrix()
    attack_types = load_attack_types()
    feature_names = load_feature_names(features.shape[1])

    if sample_per_attack is None:
        sample_indices = np.arange(len(features))
    else:
        sample_indices = _balanced_attack_sample_indices(attack_types, sample_per_attack)

    sample_features = features[sample_indices]
    sample_attack_types = attack_types[sample_indices]

    print("Sample shape:", sample_features.shape)
    print("Attack distribution:")
    print(np.unique(sample_attack_types, return_counts=True))

    shap_values = compute_tree_shap_values(model, sample_features, feature_names)
    save_shap_plots(shap_values, sample_features, feature_names)
    print_attack_type_analysis(shap_values, sample_attack_types, feature_names)


def main() -> None:
    """Execute the full explainability workflow."""
    parser = argparse.ArgumentParser(description="Train a surrogate classifier and generate SHAP explanations.")
    parser.add_argument(
        "--sample-per-attack",
        type=int,
        default=100,
        help="Number of rows to sample per attack type for the main SHAP pipeline.",
    )
    parser.add_argument(
        "--full-shap",
        action="store_true",
        help="Compute SHAP values for the full test set instead of a balanced sample.",
    )
    args, _ = parser.parse_known_args()

    sample_per_attack = None if args.full_shap else args.sample_per_attack
    run_explainability_pipeline(sample_per_attack=sample_per_attack)


if __name__ == "__main__":
    main()
