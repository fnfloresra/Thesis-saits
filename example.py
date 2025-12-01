#!/usr/bin/env python
"""Example script demonstrating SAITS for time series imputation.

This script shows how to:
1. Generate synthetic time series data with missing values
2. Train a SAITS model for imputation
3. Evaluate imputation quality
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from saits import SAITS, TimeSeriesDataset, SAITSTrainer, evaluate_imputation
from saits.data import normalize_data, train_test_split, create_missing_mask


def generate_synthetic_data(
    n_samples: int = 200,
    seq_len: int = 100,
    n_features: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic time series data with missing values.

    Args:
        n_samples: Number of time series samples.
        seq_len: Length of each time series.
        n_features: Number of features.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (complete_data, missing_mask).
    """
    np.random.seed(seed)

    # Generate base signals with different patterns
    t = np.linspace(0, 4 * np.pi, seq_len)
    data = np.zeros((n_samples, seq_len, n_features))

    for i in range(n_samples):
        for j in range(n_features):
            # Combine multiple patterns
            freq = 1 + j * 0.5 + np.random.uniform(-0.1, 0.1)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = 1 + np.random.uniform(-0.2, 0.2)

            # Sinusoidal + trend + noise
            signal = amplitude * np.sin(freq * t + phase)
            signal += 0.1 * t  # Linear trend
            signal += np.random.normal(0, 0.1, seq_len)  # Noise

            data[i, :, j] = signal

    # Create missing mask (20% missing)
    mask = create_missing_mask((n_samples, seq_len, n_features), missing_rate=0.2, seed=seed)

    return data, mask.numpy()


def main():
    """Run the SAITS imputation example."""
    # Configuration
    n_features = 10
    seq_len = 100
    batch_size = 32
    epochs = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Generating synthetic data...")

    # Generate data
    complete_data, original_mask = generate_synthetic_data(
        n_samples=500, seq_len=seq_len, n_features=n_features
    )

    # Normalize data
    normalized_data, norm_stats = normalize_data(complete_data, method="standard")

    # Split data
    train_data, val_data, test_data = train_test_split(
        normalized_data, test_ratio=0.2, val_ratio=0.1
    )
    _, val_mask, test_mask = train_test_split(
        original_mask, test_ratio=0.2, val_ratio=0.1
    )

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, mode="train", artificial_mask_rate=0.2)
    val_dataset = TimeSeriesDataset(val_data, masks=val_mask, mode="eval")
    test_dataset = TimeSeriesDataset(test_data, masks=test_mask, mode="eval")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = SAITS(
        n_features=n_features,
        n_layers=2,
        d_model=128,
        n_heads=4,
        d_ff=256,
        dropout=0.1,
    )

    print(f"\nModel architecture:")
    print(f"  - Encoder layers: 2")
    print(f"  - Model dimension: 128")
    print(f"  - Attention heads: 4")
    print(f"  - Feed-forward dimension: 256")

    # Create trainer
    trainer = SAITSTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        weight_decay=0.0001,
    )

    # Train model
    print(f"\nTraining for {epochs} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=15,
        checkpoint_path="checkpoints/saits_best.pt",
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    all_predictions = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for x, mask, original in test_loader:
            x = x.to(device)
            mask = mask.to(device)
            imputed = model.impute(x, mask)
            all_predictions.append(imputed.cpu().numpy())
            all_targets.append(original.numpy())
            all_masks.append((1 - mask).cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    # Compute metrics
    metrics = evaluate_imputation(targets, predictions, masks)

    print("\nTest Set Results:")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MRE:  {metrics['mre']:.4f}")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
