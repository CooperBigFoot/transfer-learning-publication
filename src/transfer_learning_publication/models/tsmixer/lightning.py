"""PyTorch Lightning module for TSMixer model."""

from typing import Any

import torch

from ..base.base_lit_model import BaseLitModel
from ..model_factory import register_model
from .config import TSMixerConfig
from .model import TSMixer


@register_model("tsmixer", config_class=TSMixerConfig)
class LitTSMixer(BaseLitModel):
    """PyTorch Lightning Module implementation of TSMixer.

    This class extends BaseLitModel to provide a standardized interface for training,
    validation, and testing of the TSMixer model within our hydrological forecasting
    framework.
    """

    def __init__(
        self,
        config: TSMixerConfig | dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the LitTSMixer module.

        Args:
            config: TSMixer configuration as a TSMixerConfig instance or dict.
                    If None, config will be reconstructed from kwargs (for checkpoint loading).
            **kwargs: Individual hyperparameters (used when loading from checkpoint).
        """
        # Handle different initialization patterns
        if config is None and kwargs:
            # Loading from checkpoint - reconstruct config from kwargs
            tsmixer_config = TSMixerConfig(**kwargs)
            super().__init__(tsmixer_config)
        elif isinstance(config, dict):
            # Dict config provided
            tsmixer_config = TSMixerConfig.from_dict(config)
            super().__init__(tsmixer_config)
        elif config is not None:
            # TSMixerConfig instance provided
            tsmixer_config = config
            super().__init__(tsmixer_config)
        else:
            raise ValueError("Either config or hyperparameters must be provided")

        # Create the underlying TSMixer model
        self.model = TSMixer(self.config)

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
