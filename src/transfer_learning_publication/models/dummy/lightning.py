"""
PyTorch Lightning module for the NaiveLastValue model.
"""

from typing import Any

import torch

from ..base.base_lit_model import BaseLitModel
from ..model_factory import register_model
from .config import NaiveLastValueConfig
from .model import NaiveLastValue


@register_model("naive_last_value", config_class=NaiveLastValueConfig)
class LitNaiveLastValue(BaseLitModel):
    """PyTorch Lightning Module implementation of NaiveLastValue.

    This wrapper handles training, validation, and testing procedures for the
    NaiveLastValue model, though the model itself doesn't actually learn.
    """

    def __init__(
        self,
        config: NaiveLastValueConfig | dict[str, Any] | None = None,
        **kwargs,
    ):
        """Initialize the Lightning Module with a NaiveLastValueConfig.

        Args:
            config: Either a NaiveLastValueConfig object or a dictionary of config parameters.
                    If None, config will be reconstructed from kwargs (for checkpoint loading).
            **kwargs: Individual hyperparameters (used when loading from checkpoint).
        """
        # Handle different initialization patterns
        if config is None and kwargs:
            # Loading from checkpoint - reconstruct config from kwargs
            naive_config = NaiveLastValueConfig(**kwargs)
            super().__init__(naive_config)
        elif isinstance(config, dict):
            # Dict config provided
            naive_config = NaiveLastValueConfig.from_dict(config)
            super().__init__(naive_config)
        elif config is not None:
            # NaiveLastValueConfig instance provided
            naive_config = config
            super().__init__(naive_config)
        else:
            raise ValueError("Either config or hyperparameters must be provided")

        # Create the NaiveLastValue model using the config
        self.model = NaiveLastValue(self.config)

    def forward(self, x: torch.Tensor, static: torch.Tensor = None, future: torch.Tensor = None) -> torch.Tensor:
        """Forward pass that delegates to the NaiveLastValue model.

        Args:
            x: Input time series, shape [batch_size, input_len, input_size]
            static: Static features [batch_size, static_size] (not used)
            future: Future forcing data [batch_size, output_len, future_input_size] (not used)

        Returns:
            Predictions [batch_size, output_len, 1]
        """
        return self.model(x, static, future)

    def configure_optimizers(self):
        """Configure optimizer and LR scheduler.

        Overridden to use a minimal learning rate since this model doesn't actually learn.

        Returns:
            Optimizer configured with minimal learning rate
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer
