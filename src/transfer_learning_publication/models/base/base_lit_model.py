from typing import Any

import lightning as pl
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ...contracts import Batch, ForecastOutput
from ..layers.rev_in import RevIN
from .base_config import BaseConfig


class BaseLitModel(pl.LightningModule):
    """Base Lightning module for time series forecasting models.

    Provides standard training/validation/test steps with optional RevIN normalization.
    All models should extend this class and implement the forward method.
    """

    def __init__(self, config: BaseConfig | None = None, **kwargs) -> None:
        """Initialize base model with configuration.

        Args:
            config: Model configuration containing hyperparameters.
                    If None, config will be reconstructed from kwargs (for checkpoint loading).
            **kwargs: Individual hyperparameters (used when loading from checkpoint).
        """
        super().__init__()

        # Handle checkpoint loading case where config is None and kwargs are provided
        if config is None:
            if not kwargs:
                raise ValueError("Either config or hyperparameters must be provided")
            # This will be handled by subclasses to create the appropriate config type
            # For now, we'll store kwargs and let subclasses handle config creation
            self._init_from_kwargs = True
            self.config = None  # Will be set by subclass
            # Don't save hyperparameters here - let subclass do it after creating config
        else:
            self._init_from_kwargs = False
            self.config = config
            self.save_hyperparameters(config.to_dict())

        # Only initialize these if we have a config (subclass will call _complete_init)
        if self.config is not None:
            self._complete_init()

    def _complete_init(self) -> None:
        """Complete initialization after config is set.
        
        This is called either directly in __init__ (when config is provided)
        or by subclasses after they create the config from kwargs.
        """
        if self.config is None:
            raise RuntimeError("Config must be set before calling _complete_init")
            
        self.criterion = MSELoss()

        if getattr(self.config, "use_rev_in", True):
            self.rev_in = RevIN(num_features=1, eps=1e-5, affine=True)
        else:
            self.rev_in = None

        self.test_outputs = []
        self._forecast_output: ForecastOutput | None = None

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor | None = None,
        future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass to be implemented by subclasses.

        Args:
            x: Input sequences [batch_size, input_len, input_size]
            static: Static features [batch_size, static_size]
            future: Future features [batch_size, output_len, future_size]

        Returns:
            Predictions [batch_size, output_len, 1]
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Execute training step.

        Args:
            batch: Batch of data
            batch_idx: Index of current batch

        Returns:
            Loss value for backpropagation
        """
        self._validate_batch(batch)

        x_normalized = self._apply_rev_in_normalization(batch.X)

        y_hat = self(x_normalized, batch.static, batch.future)

        y_hat = self._apply_rev_in_denormalization(y_hat)

        loss = self.criterion(y_hat, batch.y.unsqueeze(-1))

        self.log("train_loss", loss, batch_size=batch.batch_size)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Execute validation step.

        Args:
            batch: Batch of data
            batch_idx: Index of current batch

        Returns:
            Loss value for validation
        """
        self._validate_batch(batch)

        x_normalized = self._apply_rev_in_normalization(batch.X)

        y_hat = self(x_normalized, batch.static, batch.future)

        y_hat = self._apply_rev_in_denormalization(y_hat)

        loss = self.criterion(y_hat, batch.y.unsqueeze(-1))

        self.log("val_loss", loss, batch_size=batch.batch_size, prog_bar=True)

        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> dict[str, Any]:
        """Execute test step and collect outputs.

        Args:
            batch: Batch of data
            batch_idx: Index of current batch

        Returns:
            Dictionary with predictions and metadata
        """
        self._validate_batch(batch)

        x_normalized = self._apply_rev_in_normalization(batch.X)

        y_hat = self(x_normalized, batch.static, batch.future)

        y_hat = self._apply_rev_in_denormalization(y_hat)

        loss = self.criterion(y_hat, batch.y.unsqueeze(-1))
        self.log("test_loss", loss, batch_size=batch.batch_size)

        output = {
            "predictions": y_hat.squeeze(-1),
            "observations": batch.y,
            "group_identifiers": batch.group_identifiers,
        }

        if batch.input_end_dates is not None:
            output["input_end_dates"] = batch.input_end_dates

        self.test_outputs.append(output)

        return output

    def on_test_epoch_start(self) -> None:
        """Initialize test output collection."""
        self.test_outputs = []
        self._forecast_output = None

    def on_test_epoch_end(self) -> ForecastOutput:
        """Consolidate test outputs into ForecastOutput contract.

        Returns:
            ForecastOutput object with all test predictions
        """
        if not self.test_outputs:
            raise RuntimeError("No test outputs collected")

        predictions = torch.cat([o["predictions"] for o in self.test_outputs], dim=0)
        observations = torch.cat([o["observations"] for o in self.test_outputs], dim=0)
        group_identifiers = [gid for o in self.test_outputs for gid in o["group_identifiers"]]

        input_end_dates = None
        if "input_end_dates" in self.test_outputs[0]:
            input_end_dates = torch.cat([o["input_end_dates"] for o in self.test_outputs], dim=0)

        # Create output contract
        forecast_output = ForecastOutput(
            predictions=predictions,
            observations=observations,
            group_identifiers=group_identifiers,
            input_end_dates=input_end_dates,
        )

        # Store for later access
        self._forecast_output = forecast_output

        # Clear temporary storage
        self.test_outputs = []

        return forecast_output

    @property
    def forecast_output(self) -> ForecastOutput:
        """Retrieve the stored forecast output from testing.

        Returns:
            ForecastOutput object from last test run

        Raises:
            RuntimeError: If testing has not been run yet
        """
        if self._forecast_output is None:
            raise RuntimeError("No forecast output available. Run trainer.test() first.")
        return self._forecast_output

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = Adam(self.parameters(), lr=self.config.learning_rate)

        scheduler_config = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=getattr(self.config, "scheduler_patience", 5),
                factor=getattr(self.config, "scheduler_factor", 0.5),
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def _apply_rev_in_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RevIN normalization to target feature only.

        Args:
            x: Input tensor [batch_size, input_len, input_size]

        Returns:
            Tensor with normalized target feature (position 0)
        """
        if self.rev_in is None:
            return x

        # Extract and normalize target feature (assumed at position 0)
        x_target = x[:, :, 0:1]  # [batch_size, input_len, 1]
        x_target_normalized = self.rev_in(x_target, mode="norm")

        # Reconstruct tensor with normalized target
        if x.size(-1) > 1:
            # Concatenate normalized target with other features
            x_other = x[:, :, 1:]
            x_normalized = torch.cat([x_target_normalized, x_other], dim=-1)
        else:
            # Only target feature present
            x_normalized = x_target_normalized

        return x_normalized

    def _apply_rev_in_denormalization(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Apply RevIN denormalization to predictions.

        Args:
            y_hat: Model predictions [batch_size, output_len, 1]

        Returns:
            Denormalized predictions
        """
        if self.rev_in is None:
            return y_hat

        return self.rev_in(y_hat, mode="denorm")

    def _validate_batch(self, batch: Batch) -> None:
        """Validate batch is correct type.

        Args:
            batch: Input to validate

        Raises:
            TypeError: If batch is not a Batch instance
        """
        if not isinstance(batch, Batch):
            raise TypeError(f"Expected Batch instance, got {type(batch)}")
