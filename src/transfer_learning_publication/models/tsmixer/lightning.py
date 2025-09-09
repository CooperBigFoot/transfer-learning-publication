"""PyTorch Lightning module for TSMixer model."""

from typing import Any

import torch

from ..base.base_lit_model import BaseLitModel
from .config import TSMixerConfig
from .model import TSMixer


class LitTSMixer(BaseLitModel):
    """PyTorch Lightning Module implementation of TSMixer.

    This class extends BaseLitModel to provide a standardized interface for training,
    validation, and testing of the TSMixer model within our hydrological forecasting
    framework.
    """

    def __init__(
        self,
        config: TSMixerConfig | dict[str, Any],
    ) -> None:
        """
        Initialize the LitTSMixer module.

        Args:
            config: TSMixer configuration as a TSMixerConfig instance or dict
        """
        # Convert dict config to TSMixerConfig if needed
        tsmixer_config = TSMixerConfig.from_dict(config) if isinstance(config, dict) else config

        # Initialize base lightning model with the config
        super().__init__(tsmixer_config)

        # Create the underlying TSMixer model
        self.model = TSMixer(tsmixer_config)

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor | None = None,
        future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass that delegates to the TSMixer model.

        Args:
            x: Historical input features [B, input_len, input_size]
            static: Static features [B, static_size]
            future: Optional future forcing data [B, output_len, future_input_size]

        Returns:
            Predictions [B, output_len, 1]
        """
        return self.model(x, static, future)
