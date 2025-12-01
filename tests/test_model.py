"""Tests for SAITS model."""

import numpy as np
import pytest
import torch

from saits.model import SAITS, SAITSLoss, PositionalEncoding, EncoderLayer


class TestPositionalEncoding:
    """Tests for PositionalEncoding class."""

    def test_output_shape(self):
        """Test that positional encoding preserves input shape."""
        d_model = 64
        seq_len = 50
        batch_size = 8

        pe = PositionalEncoding(d_model=d_model)
        x = torch.randn(seq_len, batch_size, d_model)
        output = pe(x)

        assert output.shape == x.shape

    def test_adds_positional_info(self):
        """Test that positional encoding adds information to input."""
        d_model = 64
        pe = PositionalEncoding(d_model=d_model, dropout=0.0)
        x = torch.zeros(10, 1, d_model)
        output = pe(x)

        # Output should not be all zeros
        assert not torch.allclose(output, x)


class TestEncoderLayer:
    """Tests for EncoderLayer class."""

    def test_output_shape(self):
        """Test that encoder layer preserves input shape."""
        d_model = 64
        n_heads = 4
        d_ff = 128
        batch_size = 8
        seq_len = 50

        layer = EncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        output = layer(x)

        assert output.shape == x.shape


class TestSAITS:
    """Tests for SAITS model."""

    @pytest.fixture
    def model(self):
        """Create a SAITS model for testing."""
        return SAITS(
            n_features=10,
            n_layers=2,
            d_model=64,
            n_heads=4,
            d_ff=128,
            dropout=0.1,
        )

    def test_forward_output_shape(self, model):
        """Test that forward pass produces correct output shape."""
        batch_size = 8
        seq_len = 50
        n_features = 10

        x = torch.randn(batch_size, seq_len, n_features)
        mask = torch.ones(batch_size, seq_len, n_features)

        imputed, loss_components = model(x, mask)

        assert imputed.shape == (batch_size, seq_len, n_features)
        assert "x_hat_1" in loss_components
        assert "x_hat_2" in loss_components
        assert "x_hat_3" in loss_components

    def test_imputation_preserves_observed(self, model):
        """Test that imputation preserves observed values."""
        batch_size = 4
        seq_len = 20
        n_features = 10

        x = torch.randn(batch_size, seq_len, n_features)
        mask = torch.ones(batch_size, seq_len, n_features)
        # Mark some values as missing
        mask[:, :5, :3] = 0

        imputed, _ = model(x, mask)

        # Observed values should be preserved
        observed_mask = mask.bool()
        assert torch.allclose(imputed[observed_mask], x[observed_mask])

    def test_impute_method(self, model):
        """Test the impute convenience method."""
        x = torch.randn(4, 20, 10)
        mask = torch.ones_like(x)
        mask[:, :5, :3] = 0

        imputed = model.impute(x, mask)

        assert imputed.shape == x.shape
        # Model should be in eval mode after impute
        assert not model.training

    def test_gradient_flow(self, model):
        """Test that gradients flow through the model."""
        x = torch.randn(4, 20, 10, requires_grad=False)
        mask = torch.ones_like(x)

        imputed, _ = model(x, mask)
        loss = imputed.sum()
        loss.backward()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestSAITSLoss:
    """Tests for SAITSLoss class."""

    def test_loss_computation(self):
        """Test loss computation with known inputs."""
        criterion = SAITSLoss(mit_weight=1.0, ort_weight=1.0)

        batch_size = 4
        seq_len = 20
        n_features = 10

        x_original = torch.randn(batch_size, seq_len, n_features)
        mask = torch.ones(batch_size, seq_len, n_features)
        mask[:, :5, :3] = 0  # Mark some as missing

        loss_components = {
            "x_hat_1": x_original + torch.randn_like(x_original) * 0.1,
            "x_hat_2": x_original + torch.randn_like(x_original) * 0.1,
            "x_hat_3": x_original + torch.randn_like(x_original) * 0.1,
        }

        losses = criterion(x_original, mask, loss_components)

        assert "total_loss" in losses
        assert "ort_loss" in losses
        assert "mit_loss" in losses
        assert losses["total_loss"] >= 0
        assert losses["ort_loss"] >= 0
        assert losses["mit_loss"] >= 0

    def test_perfect_prediction_zero_loss(self):
        """Test that perfect prediction gives zero loss."""
        criterion = SAITSLoss()

        x_original = torch.randn(4, 20, 10)
        mask = torch.ones_like(x_original)

        loss_components = {
            "x_hat_1": x_original.clone(),
            "x_hat_2": x_original.clone(),
            "x_hat_3": x_original.clone(),
        }

        losses = criterion(x_original, mask, loss_components)

        assert torch.isclose(losses["total_loss"], torch.tensor(0.0), atol=1e-6)
