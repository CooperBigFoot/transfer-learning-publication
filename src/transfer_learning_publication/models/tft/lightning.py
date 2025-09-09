"""PyTorch Lightning module for Temporal Fusion Transformer model."""

from typing import Any

import torch

from ..base.base_lit_model import BaseLitModel
from .config import TFTConfig
from .model import TemporalFusionTransformer


class LitTFT(BaseLitModel):
    """PyTorch Lightning Module implementation of TFT.

    This class extends BaseLitModel to provide a standardized interface for training,
    validation, and testing of the Temporal Fusion Transformer within our
    hydrological forecasting framework.
    """

    def __init__(
        self,
        config: TFTConfig | dict[str, Any],
    ) -> None:
        """
        Initialize the LitTFT module.

        Args:
            config: TFT configuration as a TFTConfig instance or dict
        """
        # Convert dict config to TFTConfig if needed
        if isinstance(config, dict):
            config = TFTConfig.from_dict(config)

        # Initialize base lightning model with the config
        super().__init__(config)

        # Create the TFT model
        self.model = TemporalFusionTransformer(config)

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor | None = None,
        future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass that delegates to the TFT model.

        Args:
            x: Historical input features [batch_size, input_len, input_size]
            static: Static features [batch_size, static_size]
            future: Future forcing data [batch_size, output_len, future_input_size]

        Returns:
            Predictions [batch_size, output_len, 1]
        """
        return self.model(x, static, future)