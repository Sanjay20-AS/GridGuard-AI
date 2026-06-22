from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.9
THRESHOLD_PERCENTILE = 95
PROCESSED_DIR = Path("data") / "processed"
MODELS_DIR = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "autoencoder.pt"
LOSS_HISTORY_PATH = MODELS_DIR / "loss_history.npy"
TRAIN_ERRORS_PATH = PROCESSED_DIR / "train_errors.npy"
TEST_ERRORS_PATH = PROCESSED_DIR / "test_errors.npy"


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set all random seeds for reproducible training and scoring."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use CUDA when available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMAutoencoder(nn.Module):
    """LSTM autoencoder for consumption sequence reconstruction."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder_lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
        )
        self.encoder_lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.decoder_lstm1 = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.decoder_lstm2 = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
        )
        self.output_layer = nn.Linear(64, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input sequence and return the final hidden representation."""
        encoded_sequence, _ = self.encoder_lstm1(x)
        _, (hidden_state, _) = self.encoder_lstm2(encoded_sequence)
        return hidden_state[-1]

    def decode(self, bottleneck: torch.Tensor, sequence_length: int) -> torch.Tensor:
        """Repeat the bottleneck across time and reconstruct the original sequence."""
        repeated = bottleneck.unsqueeze(1).repeat(1, sequence_length, 1)
        decoded_sequence, _ = self.decoder_lstm1(repeated)
        decoded_sequence, _ = self.decoder_lstm2(decoded_sequence)
        return self.output_layer(decoded_sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.encode(x)
        return self.decode(bottleneck, x.size(1))


def load_training_sequences(processed_dir: Path = PROCESSED_DIR) -> np.ndarray:
    """Load the normal customer sequences used for autoencoder training."""
    return np.load(processed_dir / "X_train.npy")


def prepare_tensor_datasets(x_train: np.ndarray, train_split: float = TRAIN_SPLIT) -> Tuple[TensorDataset, TensorDataset]:
    """Create train and validation datasets from the normal-only training sequences."""
    train_sequences, validation_sequences = train_test_split(
        x_train,
        train_size=train_split,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    train_tensor = torch.tensor(train_sequences, dtype=torch.float32).unsqueeze(-1)
    validation_tensor = torch.tensor(validation_sequences, dtype=torch.float32).unsqueeze(-1)

    train_dataset = TensorDataset(train_tensor, train_tensor)
    validation_dataset = TensorDataset(validation_tensor, validation_tensor)
    return train_dataset, validation_dataset


def create_dataloaders(train_dataset: TensorDataset, validation_dataset: TensorDataset) -> Tuple[DataLoader, DataLoader]:
    """Create mini-batch loaders for training and validation."""
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    return train_loader, validation_loader


def train_autoencoder(
    model: LSTMAutoencoder,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> list[tuple[float, float]]:
    """Train the autoencoder and save the best checkpoint by validation loss."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_validation_loss = float("inf")
    history: list[tuple[float, float]] = []

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_sample_count = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            train_loss_total += float(loss.item()) * batch_size
            train_sample_count += batch_size

        train_loss = train_loss_total / max(train_sample_count, 1)
        validation_loss = evaluate_loss(model, validation_loader, criterion, device)
        history.append((train_loss, validation_loss))

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "sequence_length": int(train_loader.dataset.tensors[0].shape[1]),
                },
                BEST_MODEL_PATH,
            )

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs} - train_loss: {train_loss:.6f} - val_loss: {validation_loss:.6f}"
            )

    np.save(LOSS_HISTORY_PATH, np.asarray(history, dtype=float))
    return history


@torch.no_grad()
def evaluate_loss(
    model: LSTMAutoencoder,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute mean reconstruction loss over a dataset."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        batch_size = inputs.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def reconstruct_sequences(
    model: LSTMAutoencoder,
    sequences: np.ndarray,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Run the model in inference mode and return reconstructed sequences."""
    tensor = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    reconstructed_batches: list[np.ndarray] = []

    for (inputs,) in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        reconstructed_batches.append(outputs.squeeze(-1).cpu().numpy())

    return np.vstack(reconstructed_batches)


def _safe_group_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Return a group mean when available, otherwise NaN without warnings."""
    if np.any(mask):
        return float(np.mean(values[mask]))
    return float("nan")


def compute_reconstruction_errors(device: torch.device | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load the best checkpoint and compute per-customer reconstruction errors."""
    device = device or get_device()

    x_train = np.load(PROCESSED_DIR / "X_train.npy")
    x_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model = LSTMAutoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_reconstructed = reconstruct_sequences(model, x_train, device)
    test_reconstructed = reconstruct_sequences(model, x_test, device)

    train_errors = np.mean((x_train - train_reconstructed) ** 2, axis=1)
    test_errors = np.mean((x_test - test_reconstructed) ** 2, axis=1)

    np.save(TRAIN_ERRORS_PATH, train_errors)
    np.save(TEST_ERRORS_PATH, test_errors)

    normal_mask = y_test == 0
    theft_mask = y_test == 1
    print(f"Mean train reconstruction error: {float(np.mean(train_errors)):.6f}")
    normal_mean = _safe_group_mean(test_errors, normal_mask)
    theft_mean = _safe_group_mean(test_errors, theft_mask)
    print(f"Mean test reconstruction error - normal: {normal_mean:.6f}" if np.isfinite(normal_mean) else "Mean test reconstruction error - normal: n/a")
    print(f"Mean test reconstruction error - theft: {theft_mean:.6f}" if np.isfinite(theft_mean) else "Mean test reconstruction error - theft: n/a")

    return train_errors, test_errors


def select_threshold(train_errors: np.ndarray, test_errors: np.ndarray, y_test: np.ndarray) -> float:
    """Set a percentile threshold from training errors and evaluate on the test set."""
    threshold = float(np.percentile(train_errors, THRESHOLD_PERCENTILE))
    predictions = (test_errors > threshold).astype(int)

    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print(f"Threshold ({THRESHOLD_PERCENTILE}th percentile): {threshold:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1 score: {f1:.6f}")

    return threshold


def run_training_pipeline() -> LSTMAutoencoder:
    """Train the LSTM autoencoder from the processed training sequences."""
    set_random_seed(RANDOM_SEED)
    device = get_device()
    print(f"Using device: {device}")

    x_train = load_training_sequences()
    train_dataset, validation_dataset = prepare_tensor_datasets(x_train)
    train_loader, validation_loader = create_dataloaders(train_dataset, validation_dataset)

    model = LSTMAutoencoder().to(device)
    train_autoencoder(model, train_loader, validation_loader, device)
    return model


def run_scoring_pipeline() -> None:
    """Load the best model, compute reconstruction errors, and report threshold metrics."""
    device = get_device()
    train_errors, test_errors = compute_reconstruction_errors(device=device)
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    select_threshold(train_errors, test_errors, y_test)


def main() -> None:
    """Train the model, then score train/test reconstruction errors."""
    run_training_pipeline()
    run_scoring_pipeline()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate an LSTM autoencoder for energy theft detection.")
    _ = parser.parse_args()
    main()
