"""Tests for evaluation metrics."""

import numpy as np
import pytest

from saits.evaluate import (
    evaluate_imputation,
    compute_reconstruction_error,
    compute_feature_wise_metrics,
)


class TestEvaluateImputation:
    """Tests for evaluate_imputation function."""

    def test_perfect_imputation(self):
        """Test that perfect imputation gives zero errors."""
        ground_truth = np.random.randn(100, 50, 10)
        imputed = ground_truth.copy()
        mask = np.random.random((100, 50, 10)) > 0.5

        metrics = evaluate_imputation(ground_truth, imputed, mask)

        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0

    def test_metrics_positive(self):
        """Test that metrics are non-negative."""
        ground_truth = np.random.randn(100, 50, 10)
        imputed = ground_truth + np.random.randn(100, 50, 10) * 0.5
        mask = np.random.random((100, 50, 10)) > 0.5

        metrics = evaluate_imputation(ground_truth, imputed, mask)

        assert metrics["mae"] >= 0
        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["mre"] >= 0

    def test_rmse_greater_than_mae(self):
        """Test that RMSE >= MAE (mathematical property)."""
        ground_truth = np.random.randn(100, 50, 10)
        imputed = ground_truth + np.random.randn(100, 50, 10)
        mask = np.random.random((100, 50, 10)) > 0.5

        metrics = evaluate_imputation(ground_truth, imputed, mask)

        assert metrics["rmse"] >= metrics["mae"]

    def test_shape_mismatch_raises(self):
        """Test that shape mismatch raises AssertionError."""
        ground_truth = np.random.randn(100, 50, 10)
        imputed = np.random.randn(100, 50, 5)  # Wrong shape
        mask = np.random.random((100, 50, 10)) > 0.5

        with pytest.raises(AssertionError):
            evaluate_imputation(ground_truth, imputed, mask)

    def test_empty_mask(self):
        """Test with no missing values (all zeros in mask)."""
        ground_truth = np.random.randn(10, 5, 3)
        imputed = np.random.randn(10, 5, 3)
        mask = np.zeros((10, 5, 3))  # No missing values

        metrics = evaluate_imputation(ground_truth, imputed, mask)

        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0


class TestComputeReconstructionError:
    """Tests for compute_reconstruction_error function."""

    def test_perfect_reconstruction(self):
        """Test that perfect reconstruction gives zero error."""
        original = np.random.randn(100, 50, 10)
        reconstructed = original.copy()

        metrics = compute_reconstruction_error(original, reconstructed)

        assert metrics["reconstruction_mae"] == 0.0
        assert metrics["reconstruction_mse"] == 0.0
        assert metrics["reconstruction_rmse"] == 0.0

    def test_error_increases_with_noise(self):
        """Test that error increases with more noise."""
        original = np.random.randn(100, 50, 10)
        reconstructed_low = original + np.random.randn(100, 50, 10) * 0.1
        reconstructed_high = original + np.random.randn(100, 50, 10) * 1.0

        metrics_low = compute_reconstruction_error(original, reconstructed_low)
        metrics_high = compute_reconstruction_error(original, reconstructed_high)

        assert metrics_low["reconstruction_mae"] < metrics_high["reconstruction_mae"]


class TestComputeFeatureWiseMetrics:
    """Tests for compute_feature_wise_metrics function."""

    def test_returns_metrics_for_all_features(self):
        """Test that metrics are returned for all features."""
        n_features = 5
        ground_truth = np.random.randn(100, 50, n_features)
        imputed = ground_truth + np.random.randn(100, 50, n_features) * 0.1
        mask = np.random.random((100, 50, n_features)) > 0.5

        feature_metrics = compute_feature_wise_metrics(ground_truth, imputed, mask)

        assert len(feature_metrics) == n_features
        for i in range(n_features):
            assert f"feature_{i}" in feature_metrics
            assert "mae" in feature_metrics[f"feature_{i}"]

    def test_metrics_vary_by_feature(self):
        """Test that different features can have different metrics."""
        ground_truth = np.random.randn(100, 50, 3)
        imputed = ground_truth.copy()
        # Add noise only to feature 0
        imputed[:, :, 0] += np.random.randn(100, 50)

        mask = np.ones((100, 50, 3))

        feature_metrics = compute_feature_wise_metrics(ground_truth, imputed, mask)

        # Feature 0 should have higher error
        assert feature_metrics["feature_0"]["mae"] > feature_metrics["feature_1"]["mae"]
