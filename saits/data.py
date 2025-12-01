"""Data preprocessing utilities for SAITS.

This module provides utilities for preparing time series data for imputation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series imputation.

    Args:
        data: Time series data of shape (n_samples, seq_len, n_features).
        masks: Missing value masks of shape (n_samples, seq_len, n_features).
               1 indicates observed, 0 indicates missing.
        artificial_mask_rate: Rate of artificial masking for training (default: 0.2).
        mode: Either 'train' or 'eval'. Training mode applies artificial masking.

    Example:
        >>> data = np.random.randn(100, 50, 10)  # 100 samples, 50 timesteps, 10 features
        >>> dataset = TimeSeriesDataset(data, mode='train', artificial_mask_rate=0.2)
        >>> x, mask, original = dataset[0]
    """

    def __init__(
        self,
        data: np.ndarray,
        masks: np.ndarray = None,
        artificial_mask_rate: float = 0.2,
        mode: str = "train",
    ):
        self.data = torch.tensor(data, dtype=torch.float32)
        
        if masks is not None:
            self.original_masks = torch.tensor(masks, dtype=torch.float32)
        else:
            self.original_masks = torch.ones_like(self.data)
        
        self.artificial_mask_rate = artificial_mask_rate
        self.mode = mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Tuple containing:
                - x: Input data with missing values set to 0
                - mask: Combined mask (original + artificial for training)
                - original: Original complete data for loss computation
        """
        original = self.data[idx].clone()
        original_mask = self.original_masks[idx].clone()

        if self.mode == "train":
            # Apply artificial masking for training
            artificial_mask = create_missing_mask(
                original.shape, self.artificial_mask_rate, seed=None
            )
            # Only apply artificial mask to observed values
            combined_mask = original_mask * artificial_mask
        else:
            combined_mask = original_mask

        # Set missing values to 0
        x = original * combined_mask

        return x, combined_mask, original


def create_missing_mask(
    shape: tuple, missing_rate: float, seed: int = None
) -> torch.Tensor:
    """Create a random missing value mask.

    Args:
        shape: Shape of the mask (seq_len, n_features) or (batch, seq_len, n_features).
        missing_rate: Proportion of values to mark as missing (0 to 1).
        seed: Random seed for reproducibility.

    Returns:
        Binary mask tensor where 1 = observed, 0 = missing.

    Example:
        >>> mask = create_missing_mask((100, 10), missing_rate=0.2)
        >>> mask.shape
        torch.Size([100, 10])
    """
    if seed is not None:
        np.random.seed(seed)

    mask = np.random.random(shape) > missing_rate
    return torch.tensor(mask, dtype=torch.float32)


def normalize_data(
    data: np.ndarray, method: str = "standard"
) -> tuple[np.ndarray, dict]:
    """Normalize time series data.

    Args:
        data: Time series data of shape (n_samples, seq_len, n_features).
        method: Normalization method, either 'standard' or 'minmax'.

    Returns:
        Tuple containing:
            - normalized_data: Normalized data array
            - stats: Dictionary with normalization statistics for inverse transform

    Example:
        >>> data = np.random.randn(100, 50, 10) * 10 + 5
        >>> normalized, stats = normalize_data(data, method='standard')
        >>> normalized.mean(), normalized.std()
        (approximately 0.0, approximately 1.0)
    """
    if method == "standard":
        # Compute mean and std across samples and timesteps (per feature)
        mean = np.nanmean(data, axis=(0, 1), keepdims=True)
        std = np.nanstd(data, axis=(0, 1), keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        normalized_data = (data - mean) / std
        stats = {"method": "standard", "mean": mean.squeeze(), "std": std.squeeze()}

    elif method == "minmax":
        min_val = np.nanmin(data, axis=(0, 1), keepdims=True)
        max_val = np.nanmax(data, axis=(0, 1), keepdims=True)
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        normalized_data = (data - min_val) / range_val
        stats = {"method": "minmax", "min": min_val.squeeze(), "max": max_val.squeeze()}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized_data, stats


def denormalize_data(data: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse transform normalized data.

    Args:
        data: Normalized data array.
        stats: Dictionary with normalization statistics from normalize_data().

    Returns:
        Denormalized data array.
    """
    method = stats["method"]

    if method == "standard":
        return data * stats["std"] + stats["mean"]
    elif method == "minmax":
        range_val = stats["max"] - stats["min"]
        return data * range_val + stats["min"]
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def sliding_window(
    data: np.ndarray, window_size: int, stride: int = 1
) -> np.ndarray:
    """Create sliding windows from time series data.

    Args:
        data: Time series data of shape (total_timesteps, n_features).
        window_size: Size of each window.
        stride: Step size between windows (default: 1).

    Returns:
        Array of shape (n_windows, window_size, n_features).

    Example:
        >>> data = np.random.randn(1000, 10)  # 1000 timesteps, 10 features
        >>> windows = sliding_window(data, window_size=100, stride=10)
        >>> windows.shape
        (91, 100, 10)
    """
    n_timesteps, n_features = data.shape
    n_windows = (n_timesteps - window_size) // stride + 1

    windows = np.zeros((n_windows, window_size, n_features))
    for i in range(n_windows):
        start = i * stride
        windows[i] = data[start : start + window_size]

    return windows


def train_test_split(
    data: np.ndarray, test_ratio: float = 0.2, val_ratio: float = 0.1, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train, validation, and test sets.

    Args:
        data: Data array of shape (n_samples, ...).
        test_ratio: Proportion of data for test set.
        val_ratio: Proportion of training data for validation set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_data, val_data, test_data).
    """
    np.random.seed(seed)
    n_samples = len(data)
    indices = np.random.permutation(n_samples)

    test_size = int(n_samples * test_ratio)
    val_size = int(n_samples * val_ratio)

    test_indices = indices[:test_size]
    val_indices = indices[test_size : test_size + val_size]
    train_indices = indices[test_size + val_size :]

    return data[train_indices], data[val_indices], data[test_indices]
