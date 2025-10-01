"""
PyTorch Lightning module for segmentation.

This module encapsulates the loss, metrics, and training logic
around an externally provided segmentation model.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl

from ..losses import get_loss
from ..metrics import get_metric
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SegmentationModel(pl.LightningModule):
    """
    Lightning module for segmentation tasks.

    This class wraps a ready-made segmentation model with Lightning
    training logic, handling loss computation, metric tracking, and
    optimization.

    Args:
        config: Configuration dictionary containing model, training,
                and loss specifications
        model: Instantiated segmentation model (e.g., SMP Unet)

    Example:
        >>> config = load_config("configs/drive.yaml")
        >>> backbone = build_model(config)
        >>> model = SegmentationModel(config, backbone)
        >>> trainer.fit(model, datamodule)
    """

    def __init__(self, config: Dict[str, Any], model: nn.Module):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

        # Extract configurations
        self.model_config = config["model"]
        self.training_config = config["training"]

        # Use externally provided model (e.g., from factory)
        self.model = model

        # Setup loss function
        self.loss_fn = self._create_loss()

        # Setup metrics
        self.train_metrics = self._create_metrics("train")
        self.val_metrics = self._create_metrics("val")

        # Training configuration
        self.learning_rate = self.training_config["learning_rate"]
        self.optimizer_name = self.training_config.get("optimizer", "adamw")
        self.weight_decay = self.training_config.get("weight_decay", 1e-4)
        self.scheduler_config = self.training_config.get("scheduler")

        encoder = self.model_config.get("encoder", "n/a")
        logger.info(
            "Attached model %s (encoder=%s)",
            self.model_config["architecture"],
            encoder,
        )

    def _create_loss(self) -> nn.Module:
        """Create loss function from configuration."""
        loss_config = self.training_config["loss"]
        return get_loss(loss_config["type"], loss_config.get("params", {}))

    def _create_metrics(self, prefix: str) -> nn.ModuleDict:
        """
        Create metrics for the specified prefix.

        Args:
            prefix: Metric prefix (train/val)

        Returns:
            ModuleDict of metrics
        """
        metrics = nn.ModuleDict()
        metric_names = self.training_config.get("metrics", ["dice"])

        for name in metric_names:
            try:
                metric = get_metric(name, threshold=0.5)
                metrics[name] = metric
            except ValueError as e:
                logger.warning(f"Skipping unknown metric '{name}': {e}")

        return metrics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step for one batch."""
        images, masks = batch

        # Forward pass
        logits = self(images)

        # Calculate loss
        loss = self.loss_fn(logits, masks)

        # Calculate and log metrics
        with torch.no_grad():
            preds = torch.sigmoid(logits)

            for name, metric in self.train_metrics.items():
                metric_value = metric(preds, masks)
                self.log(f"train/{name}", metric_value, on_step=False,
                        on_epoch=True, prog_bar=True, sync_dist=True)

        # Log loss
        self.log("train/loss", loss, on_step=True, on_epoch=True,
                prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """Validation step for one batch."""
        images, masks = batch

        # Forward pass
        logits = self(images)

        # Calculate loss
        loss = self.loss_fn(logits, masks)

        # Calculate and log metrics
        preds = torch.sigmoid(logits)

        for name, metric in self.val_metrics.items():
            metric_value = metric(preds, masks)
            self.log(f"val/{name}", metric_value, on_step=False,
                    on_epoch=True, prog_bar=True, sync_dist=True)

        # Log loss
        self.log("val/loss", loss, on_step=False, on_epoch=True,
                prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """Test step for one batch."""
        images, masks = batch

        # Forward pass
        logits = self(images)

        # Calculate loss
        loss = self.loss_fn(logits, masks)

        # Calculate and log metrics
        preds = torch.sigmoid(logits)

        for name, metric in self.val_metrics.items():
            metric_value = metric(preds, masks)
            self.log(f"test/{name}", metric_value, on_step=False,
                    on_epoch=True, sync_dist=True)

        # Log loss
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        """
        Prediction step for inference.

        Args:
            batch: Input batch (images or image-mask pairs)
            batch_idx: Batch index
            dataloader_idx: Dataloader index

        Returns:
            Predictions as probabilities
        """
        # Handle both single images and image-mask pairs
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images = batch[0]
        else:
            images = batch

        # Use SDD-compliant prediction method
        return self.predict(images, strategy="standard")

    def predict(self, x: torch.Tensor, strategy: str = "standard") -> torch.Tensor:
        """
        SDD v4.1 compliant prediction method.

        Args:
            x: Input tensor [B, C, H, W]
            strategy: Prediction strategy ("standard", "tta_hflip", "tta_vflip")

        Returns:
            Predictions with same spatial dims as input

        Raises:
            ValueError: If strategy is unknown
        """
        PREDICTION_STRATEGIES = {
            'standard': 'Single forward pass',
            'tta_hflip': 'Horizontal flip test-time augmentation',
            'tta_vflip': 'Vertical flip test-time augmentation'
        }

        if strategy not in PREDICTION_STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available: {list(PREDICTION_STRATEGIES.keys())}"
            )

        if strategy == "standard":
            # Standard single forward pass
            logits = self.forward(x)
            return torch.sigmoid(logits)

        elif strategy == "tta_hflip":
            # Test-time augmentation with horizontal flip
            logits_orig = self.forward(x)
            logits_flip = self.forward(torch.flip(x, dims=[3]))  # Flip width
            logits_flip = torch.flip(logits_flip, dims=[3])      # Flip back
            logits = (logits_orig + logits_flip) / 2
            return torch.sigmoid(logits)

        elif strategy == "tta_vflip":
            # Test-time augmentation with vertical flip
            logits_orig = self.forward(x)
            logits_flip = self.forward(torch.flip(x, dims=[2]))  # Flip height
            logits_flip = torch.flip(logits_flip, dims=[2])      # Flip back
            logits = (logits_orig + logits_flip) / 2
            return torch.sigmoid(logits)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Setup optimizer
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        # Return optimizer only if no scheduler
        if self.scheduler_config is None:
            return optimizer

        # Setup scheduler
        scheduler_type = self.scheduler_config["type"].lower()

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.scheduler_config.get("min_lr", 1e-6)
            )
            return [optimizer], [scheduler]

        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_config.get("step_size", 10),
                gamma=self.scheduler_config.get("gamma", 0.1)
            )
            return [optimizer], [scheduler]

        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_config.get("factor", 0.5),
                patience=self.scheduler_config.get("patience", 5),
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                }
            }

        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Reset metrics
        for metric in self.train_metrics.values():
            metric.reset()

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        # Reset metrics
        for metric in self.val_metrics.values():
            metric.reset()

    def get_model_summary(self) -> str:
        """
        Get model summary string.

        Returns:
            Model summary as string
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        summary = f"""
Model Summary:
- Architecture: {self.model_config['architecture']}
- Encoder: {self.model_config.get('encoder', 'n/a')}
- Input channels: {self.model_config.get('in_channels', 3)}
- Output channels: {self.model_config.get('out_channels', 1)}
- Total parameters: {total_params:,}
- Trainable parameters: {trainable_params:,}
- Loss function: {self.training_config['loss']['type']}
- Optimizer: {self.optimizer_name}
- Learning rate: {self.learning_rate}
"""
        return summary
