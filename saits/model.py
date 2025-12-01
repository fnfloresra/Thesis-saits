"""SAITS Model Implementation.

Self-Attention-based Imputation for Time Series (SAITS) model.
Reference: Du et al., "SAITS: Self-Attention-based Imputation for Time Series"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based models."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (seq_len, batch, d_model)

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class DiagonalMaskedSelfAttention(nn.Module):
    """Diagonal-masked self-attention layer for SAITS."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute diagonal-masked self-attention.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            mask: Optional attention mask

        Returns:
            Attention output tensor.
        """
        batch_size = q.size(0)

        # Linear projections
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        return self.w_o(attn_output)


class EncoderLayer(nn.Module):
    """Transformer encoder layer for SAITS."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = DiagonalMaskedSelfAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass through encoder layer.

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            Encoded output tensor.
        """
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class SAITS(nn.Module):
    """Self-Attention-based Imputation for Time Series.

    This model uses a dual-branch self-attention architecture to impute
    missing values in multivariate time series data.

    Args:
        n_features: Number of features in the time series.
        n_layers: Number of encoder layers (default: 2).
        d_model: Dimension of the model (default: 256).
        n_heads: Number of attention heads (default: 4).
        d_ff: Dimension of feed-forward layer (default: 256).
        dropout: Dropout rate (default: 0.1).

    Example:
        >>> model = SAITS(n_features=10, n_layers=2, d_model=128)
        >>> x = torch.randn(32, 100, 10)  # (batch, seq_len, features)
        >>> mask = torch.ones(32, 100, 10)  # 1 = observed, 0 = missing
        >>> imputed, loss_components = model(x, mask)
    """

    def __init__(
        self,
        n_features: int,
        n_layers: int = 2,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model

        # Input embedding
        self.embedding = nn.Linear(n_features * 2, d_model)  # *2 for mask concatenation
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # First DMSA block (encoder layers)
        self.encoder_layers_1 = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Second DMSA block (encoder layers)
        self.encoder_layers_2 = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Output projections
        self.output_projection_1 = nn.Linear(d_model, n_features)
        self.output_projection_2 = nn.Linear(d_model, n_features)

        # Combining weights
        self.weight_combine = nn.Linear(n_features * 2, n_features)

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Forward pass for SAITS model.

        Args:
            x: Input time series tensor of shape (batch, seq_len, n_features).
               Missing values should be set to 0.
            mask: Binary mask tensor of shape (batch, seq_len, n_features).
                  1 indicates observed values, 0 indicates missing values.

        Returns:
            Tuple containing:
                - imputed: Imputed time series tensor of shape (batch, seq_len, n_features)
                - loss_components: Dictionary with intermediate outputs for loss computation
        """
        # Concatenate input with mask
        x_concat = torch.cat([x, mask], dim=-1)

        # Embedding and positional encoding
        h = self.embedding(x_concat)
        h = h.transpose(0, 1)  # (seq_len, batch, d_model)
        h = self.pos_encoding(h)
        h = h.transpose(0, 1)  # (batch, seq_len, d_model)

        # First DMSA block
        for layer in self.encoder_layers_1:
            h = layer(h)
        x_hat_1 = self.output_projection_1(h)

        # Replace observed values for second block input
        x_tilde_1 = mask * x + (1 - mask) * x_hat_1

        # Prepare input for second block
        x_concat_2 = torch.cat([x_tilde_1, mask], dim=-1)
        h2 = self.embedding(x_concat_2)
        h2 = h2.transpose(0, 1)
        h2 = self.pos_encoding(h2)
        h2 = h2.transpose(0, 1)

        # Second DMSA block
        for layer in self.encoder_layers_2:
            h2 = layer(h2)
        x_hat_2 = self.output_projection_2(h2)

        # Combine outputs from both blocks
        combined = torch.cat([x_hat_1, x_hat_2], dim=-1)
        x_hat_3 = self.weight_combine(combined)

        # Final imputation: use observed values where available
        imputed = mask * x + (1 - mask) * x_hat_3

        loss_components = {
            "x_hat_1": x_hat_1,
            "x_hat_2": x_hat_2,
            "x_hat_3": x_hat_3,
        }

        return imputed, loss_components

    def impute(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Impute missing values in time series data.

        Args:
            x: Input time series tensor of shape (batch, seq_len, n_features).
               Missing values should be set to 0.
            mask: Binary mask tensor of shape (batch, seq_len, n_features).
                  1 indicates observed values, 0 indicates missing values.

        Returns:
            Imputed time series tensor.
        """
        self.eval()
        with torch.no_grad():
            imputed, _ = self.forward(x, mask)
        return imputed


class SAITSLoss(nn.Module):
    """Combined loss function for SAITS training.

    The loss combines reconstruction losses from all three outputs:
    - MIT (Masked Imputation Task) loss: for missing values
    - ORT (Observed Reconstruction Task) loss: for observed values
    """

    def __init__(self, mit_weight: float = 1.0, ort_weight: float = 1.0):
        super().__init__()
        self.mit_weight = mit_weight
        self.ort_weight = ort_weight

    def forward(
        self,
        x_original: torch.Tensor,
        mask: torch.Tensor,
        loss_components: dict,
    ) -> dict:
        """Compute SAITS loss.

        Args:
            x_original: Original complete time series (ground truth).
            mask: Binary mask (1 = observed in training, 0 = artificially masked).
            loss_components: Dictionary with model outputs.

        Returns:
            Dictionary containing individual and total losses.
        """
        x_hat_1 = loss_components["x_hat_1"]
        x_hat_2 = loss_components["x_hat_2"]
        x_hat_3 = loss_components["x_hat_3"]

        # ORT loss: reconstruction of observed values
        ort_loss_1 = self._masked_mae(x_original, x_hat_1, mask)
        ort_loss_2 = self._masked_mae(x_original, x_hat_2, mask)
        ort_loss_3 = self._masked_mae(x_original, x_hat_3, mask)

        # MIT loss: reconstruction of missing values
        inverse_mask = 1 - mask
        mit_loss_1 = self._masked_mae(x_original, x_hat_1, inverse_mask)
        mit_loss_2 = self._masked_mae(x_original, x_hat_2, inverse_mask)
        mit_loss_3 = self._masked_mae(x_original, x_hat_3, inverse_mask)

        # Combined losses
        ort_loss = ort_loss_1 + ort_loss_2 + ort_loss_3
        mit_loss = mit_loss_1 + mit_loss_2 + mit_loss_3

        total_loss = self.ort_weight * ort_loss + self.mit_weight * mit_loss

        return {
            "total_loss": total_loss,
            "ort_loss": ort_loss,
            "mit_loss": mit_loss,
        }

    @staticmethod
    def _masked_mae(
        x_true: torch.Tensor, x_pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute masked Mean Absolute Error.

        Args:
            x_true: Ground truth tensor.
            x_pred: Predicted tensor.
            mask: Binary mask (1 = include in loss).

        Returns:
            Masked MAE loss.
        """
        diff = torch.abs(x_true - x_pred) * mask
        n_valid = mask.sum()
        if n_valid > 0:
            return diff.sum() / n_valid
        return torch.tensor(0.0, device=x_true.device)
