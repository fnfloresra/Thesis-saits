"""Training pipeline for SAITS model."""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from saits.model import SAITS, SAITSLoss
from saits.evaluate import evaluate_imputation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SAITSTrainer:
    """Trainer class for SAITS model.

    Args:
        model: SAITS model instance.
        device: Device to train on ('cuda' or 'cpu').
        learning_rate: Learning rate for optimizer (default: 0.001).
        weight_decay: L2 regularization weight (default: 0.0001).
        mit_weight: Weight for MIT loss component (default: 1.0).
        ort_weight: Weight for ORT loss component (default: 1.0).

    Example:
        >>> model = SAITS(n_features=10)
        >>> trainer = SAITSTrainer(model, device='cuda')
        >>> history = trainer.train(train_loader, val_loader, epochs=100)
    """

    def __init__(
        self,
        model: SAITS,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        mit_weight: float = 1.0,
        ort_weight: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )
        self.criterion = SAITSLoss(mit_weight=mit_weight, ort_weight=ort_weight)

        self.best_val_loss = float("inf")
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_rmse": [],
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        checkpoint_path: Optional[str] = None,
    ) -> dict:
        """Train the SAITS model.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data (optional).
            epochs: Number of training epochs.
            early_stopping_patience: Epochs to wait before early stopping.
            checkpoint_path: Path to save model checkpoints.

        Returns:
            Dictionary containing training history.
        """
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.training_history["train_loss"].append(train_loss)

            # Validation phase
            if val_loader is not None:
                val_loss, val_metrics = self._validate_epoch(val_loader)
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_mae"].append(val_metrics["mae"])
                self.training_history["val_rmse"].append(val_metrics["rmse"])

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    if checkpoint_path:
                        self.save_checkpoint(checkpoint_path)
                else:
                    patience_counter += 1

                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f} - "
                    f"Val MAE: {val_metrics['mae']:.4f} - "
                    f"Val RMSE: {val_metrics['rmse']:.4f}"
                )

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

        return self.training_history

    def _train_epoch(self, data_loader: DataLoader) -> float:
        """Run one training epoch.

        Args:
            data_loader: Training data loader.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x, mask, original in tqdm(data_loader, desc="Training", leave=False):
            x = x.to(self.device)
            mask = mask.to(self.device)
            original = original.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            _, loss_components = self.model(x, mask)

            # Compute loss
            losses = self.criterion(original, mask, loss_components)
            loss = losses["total_loss"]

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _validate_epoch(self, data_loader: DataLoader) -> tuple[float, dict]:
        """Run one validation epoch.

        Args:
            data_loader: Validation data loader.

        Returns:
            Tuple of (average loss, metrics dictionary).
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_masks = []
        n_batches = 0

        with torch.no_grad():
            for x, mask, original in tqdm(data_loader, desc="Validation", leave=False):
                x = x.to(self.device)
                mask = mask.to(self.device)
                original = original.to(self.device)

                # Forward pass
                imputed, loss_components = self.model(x, mask)

                # Compute loss
                losses = self.criterion(original, mask, loss_components)
                total_loss += losses["total_loss"].item()

                # Store predictions for metrics
                all_predictions.append(imputed.cpu())
                all_targets.append(original.cpu())
                all_masks.append(mask.cpu())
                n_batches += 1

        # Compute metrics
        predictions = torch.cat(all_predictions, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        masks = torch.cat(all_masks, dim=0).numpy()

        metrics = evaluate_imputation(targets, predictions, 1 - masks)

        return total_loss / n_batches, metrics

    def save_checkpoint(self, path: str):
        """Save model checkpoint.

        Args:
            path: Path to save the checkpoint.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "training_history": self.training_history,
            },
            path,
        )
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint.

        Args:
            path: Path to the checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint["training_history"]
        logger.info(f"Checkpoint loaded from {path}")
