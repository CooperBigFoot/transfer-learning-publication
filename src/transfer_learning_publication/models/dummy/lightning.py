"""
PyTorch Lightning module for the NaiveLastValue model.
"""

from typing import Any

import torch

from ..base.base_lit_model import BaseLitModel
from .config import NaiveLastValueConfig
from .model import NaiveLastValue


class LitNaiveLastValue(BaseLitModel):
    """PyTorch Lightning Module implementation of NaiveLastValue.

    This wrapper handles training, validation, and testing procedures for the
    NaiveLastValue model, though the model itself doesn't actually learn.
    """

    def __init__(
        self,
        config: NaiveLastValueConfig | dict[str, Any],
    ):
        """Initialize the Lightning Module with a NaiveLastValueConfig.

        Args:
            config: Either a NaiveLastValueConfig object or a dictionary of config parameters
        """
        # Convert dict config to NaiveLastValueConfig if needed
        if isinstance(config, dict):
            config = NaiveLastValueConfig.from_dict(config)

        # Initialize base class with the config
        super().__init__(config)

        # Create the NaiveLastValue model using the config
        self.model = NaiveLastValue(config)

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
