"""Tests for data preprocessing utilities."""

import numpy as np
import pytest
import torch

from saits.data import (
    TimeSeriesDataset,
    create_missing_mask,
    normalize_data,
    denormalize_data,
    sliding_window,
    train_test_split,
)


class TestCreateMissingMask:
    """Tests for create_missing_mask function."""

    def test_mask_shape(self):
        """Test that mask has correct shape."""
        shape = (100, 50, 10)
        mask = create_missing_mask(shape, missing_rate=0.2)
        assert mask.shape == shape

    def test_mask_values_binary(self):
        """Test that mask contains only 0s and 1s."""
        mask = create_missing_mask((100, 50, 10), missing_rate=0.2)
        unique_values = torch.unique(mask)
        assert torch.all((unique_values == 0) | (unique_values == 1))

    def test_missing_rate_approximate(self):
        """Test that missing rate is approximately correct."""
        shape = (100, 50, 10)
        missing_rate = 0.3
        mask = create_missing_mask(shape, missing_rate=missing_rate, seed=42)

        actual_missing_rate = 1 - mask.float().mean().item()
        assert abs(actual_missing_rate - missing_rate) < 0.05

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same mask."""
        mask1 = create_missing_mask((50, 10), missing_rate=0.2, seed=42)
        mask2 = create_missing_mask((50, 10), missing_rate=0.2, seed=42)
        assert torch.equal(mask1, mask2)


class TestNormalizeData:
    """Tests for normalize_data and denormalize_data functions."""

    def test_standard_normalization(self):
        """Test standard normalization produces mean~0 and std~1."""
        data = np.random.randn(100, 50, 10) * 5 + 10
        normalized, stats = normalize_data(data, method="standard")

        # Check approximately normalized
        assert abs(np.mean(normalized)) < 0.1
        assert abs(np.std(normalized) - 1) < 0.1
        assert stats["method"] == "standard"

    def test_minmax_normalization(self):
        """Test minmax normalization produces values in [0, 1]."""
        data = np.random.randn(100, 50, 10) * 5 + 10
        normalized, stats = normalize_data(data, method="minmax")

        # Check values are in [0, 1]
        assert np.all(normalized >= -0.01)  # Small tolerance
        assert np.all(normalized <= 1.01)
        assert stats["method"] == "minmax"

    def test_denormalize_standard(self):
        """Test that denormalization inverts standard normalization."""
        data = np.random.randn(100, 50, 10) * 5 + 10
        normalized, stats = normalize_data(data, method="standard")
        denormalized = denormalize_data(normalized, stats)

        np.testing.assert_array_almost_equal(data, denormalized, decimal=5)

    def test_denormalize_minmax(self):
        """Test that denormalization inverts minmax normalization."""
        data = np.random.randn(100, 50, 10) * 5 + 10
        normalized, stats = normalize_data(data, method="minmax")
        denormalized = denormalize_data(normalized, stats)

        np.testing.assert_array_almost_equal(data, denormalized, decimal=5)

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        data = np.random.randn(10, 5, 3)
        with pytest.raises(ValueError):
            normalize_data(data, method="invalid")


class TestSlidingWindow:
    """Tests for sliding_window function."""

    def test_output_shape(self):
        """Test that sliding window produces correct shape."""
        data = np.random.randn(100, 10)
        window_size = 20
        stride = 5

        windows = sliding_window(data, window_size=window_size, stride=stride)

        expected_n_windows = (100 - window_size) // stride + 1
        assert windows.shape == (expected_n_windows, window_size, 10)

    def test_window_content(self):
        """Test that windows contain correct data."""
        data = np.arange(100).reshape(-1, 1)
        windows = sliding_window(data, window_size=10, stride=5)

        # First window should be 0-9
        np.testing.assert_array_equal(windows[0, :, 0], np.arange(10))
        # Second window should be 5-14
        np.testing.assert_array_equal(windows[1, :, 0], np.arange(5, 15))


class TestTrainTestSplit:
    """Tests for train_test_split function."""

    def test_split_sizes(self):
        """Test that split produces correct sizes."""
        data = np.random.randn(100, 50, 10)
        train, val, test = train_test_split(data, test_ratio=0.2, val_ratio=0.1)

        assert len(test) == 20
        assert len(val) == 10
        assert len(train) == 70

    def test_no_overlap(self):
        """Test that splits don't overlap (using indices)."""
        data = np.arange(100).reshape(-1, 1, 1)
        train, val, test = train_test_split(data, test_ratio=0.2, val_ratio=0.1, seed=42)

        all_indices = np.concatenate([train.flatten(), val.flatten(), test.flatten()])
        assert len(np.unique(all_indices)) == 100

    def test_reproducibility(self):
        """Test that same seed produces same split."""
        data = np.random.randn(100, 50, 10)
        train1, val1, test1 = train_test_split(data, seed=42)
        train2, val2, test2 = train_test_split(data, seed=42)

        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(val1, val2)
        np.testing.assert_array_equal(test1, test2)


class TestTimeSeriesDataset:
    """Tests for TimeSeriesDataset class."""

    def test_dataset_length(self):
        """Test that dataset has correct length."""
        data = np.random.randn(100, 50, 10)
        dataset = TimeSeriesDataset(data)
        assert len(dataset) == 100

    def test_getitem_returns_tuple(self):
        """Test that getitem returns tuple of tensors."""
        data = np.random.randn(100, 50, 10)
        dataset = TimeSeriesDataset(data, mode="train")
        x, mask, original = dataset[0]

        assert isinstance(x, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(original, torch.Tensor)

    def test_train_mode_applies_mask(self):
        """Test that training mode applies artificial masking."""
        data = np.ones((100, 50, 10))
        dataset = TimeSeriesDataset(data, mode="train", artificial_mask_rate=0.5)

        x, mask, original = dataset[0]

        # Original should be all ones
        assert torch.all(original == 1)
        # Some values in mask should be 0 (artificial masking)
        assert mask.float().mean() < 1.0
        # x should have zeros where mask is 0
        assert torch.all(x[mask == 0] == 0)

    def test_eval_mode_no_artificial_mask(self):
        """Test that eval mode doesn't apply artificial masking."""
        data = np.ones((100, 50, 10))
        masks = np.ones((100, 50, 10))
        dataset = TimeSeriesDataset(data, masks=masks, mode="eval")

        x, mask, original = dataset[0]

        # All masks should be 1 (no artificial masking in eval)
        assert torch.all(mask == 1)
