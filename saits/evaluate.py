"""Evaluation metrics for time series imputation."""

import numpy as np


def evaluate_imputation(
    ground_truth: np.ndarray,
    imputed: np.ndarray,
    missing_mask: np.ndarray,
) -> dict:
    """Evaluate imputation quality using various metrics.

    Args:
        ground_truth: True values array.
        imputed: Imputed values array.
        missing_mask: Binary mask where 1 indicates missing (imputed) values.

    Returns:
        Dictionary containing evaluation metrics:
            - mae: Mean Absolute Error
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - mre: Mean Relative Error

    Example:
        >>> ground_truth = np.random.randn(100, 50, 10)
        >>> imputed = ground_truth + np.random.randn(100, 50, 10) * 0.1
        >>> mask = np.random.random((100, 50, 10)) > 0.8
        >>> metrics = evaluate_imputation(ground_truth, imputed, mask)
        >>> print(f"MAE: {metrics['mae']:.4f}")
    """
    # Ensure arrays have same shape
    assert ground_truth.shape == imputed.shape == missing_mask.shape, (
        f"Shape mismatch: ground_truth {ground_truth.shape}, "
        f"imputed {imputed.shape}, missing_mask {missing_mask.shape}"
    )

    # Only evaluate on missing values
    missing_mask = missing_mask.astype(bool)
    n_missing = missing_mask.sum()

    if n_missing == 0:
        return {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "mre": 0.0}

    true_values = ground_truth[missing_mask]
    pred_values = imputed[missing_mask]

    # Mean Absolute Error
    mae = np.mean(np.abs(true_values - pred_values))

    # Mean Squared Error
    mse = np.mean((true_values - pred_values) ** 2)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Relative Error (avoid division by zero)
    denominator = np.abs(true_values)
    denominator = np.where(denominator < 1e-8, 1e-8, denominator)
    mre = np.mean(np.abs(true_values - pred_values) / denominator)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mre": float(mre),
    }


def compute_reconstruction_error(
    original: np.ndarray, reconstructed: np.ndarray
) -> dict:
    """Compute reconstruction error for all values (observed + imputed).

    Args:
        original: Original complete data.
        reconstructed: Reconstructed/imputed data.

    Returns:
        Dictionary with reconstruction metrics.
    """
    mae = np.mean(np.abs(original - reconstructed))
    mse = np.mean((original - reconstructed) ** 2)
    rmse = np.sqrt(mse)

    return {
        "reconstruction_mae": float(mae),
        "reconstruction_mse": float(mse),
        "reconstruction_rmse": float(rmse),
    }


def compute_feature_wise_metrics(
    ground_truth: np.ndarray,
    imputed: np.ndarray,
    missing_mask: np.ndarray,
) -> dict:
    """Compute metrics for each feature separately.

    Args:
        ground_truth: True values array of shape (..., n_features).
        imputed: Imputed values array of shape (..., n_features).
        missing_mask: Binary mask where 1 indicates missing values.

    Returns:
        Dictionary with per-feature metrics.
    """
    n_features = ground_truth.shape[-1]
    feature_metrics = {}

    for i in range(n_features):
        gt_feature = ground_truth[..., i]
        imp_feature = imputed[..., i]
        mask_feature = missing_mask[..., i]

        metrics = evaluate_imputation(
            gt_feature[..., np.newaxis],
            imp_feature[..., np.newaxis],
            mask_feature[..., np.newaxis],
        )
        feature_metrics[f"feature_{i}"] = metrics

    return feature_metrics
