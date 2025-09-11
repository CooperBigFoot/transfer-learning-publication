"""PyTorch Lightning module for Temporal Fusion Transformer model."""

from typing import Any

import torch

from ..base.base_lit_model import BaseLitModel
from ..model_factory import register_model
from .config import TFTConfig
from .model import TemporalFusionTransformer


@register_model("tft", config_class=TFTConfig)
class LitTFT(BaseLitModel):
    """PyTorch Lightning Module implementation of TFT.

    This class extends BaseLitModel to provide a standardized interface for training,
    validation, and testing of the Temporal Fusion Transformer within our
    hydrological forecasting framework.
    """

    def __init__(
        self,
        config: TFTConfig | dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the LitTFT module.

        Args:
            config: TFT configuration as a TFTConfig instance or dict.
                    If None, config will be reconstructed from kwargs (for checkpoint loading).
            **kwargs: Individual hyperparameters (used when loading from checkpoint).
        """
        # Handle different initialization patterns
        if config is None and kwargs:
            # Loading from checkpoint - reconstruct config from kwargs
            tft_config = TFTConfig(**kwargs)
            super().__init__(tft_config)
        elif isinstance(config, dict):
            # Dict config provided
            tft_config = TFTConfig.from_dict(config)
            super().__init__(tft_config)
        elif config is not None:
            # TFTConfig instance provided
            tft_config = config
            super().__init__(tft_config)
        else:
            raise ValueError("Either config or hyperparameters must be provided")

        # Create the TFT model
        self.model = TemporalFusionTransformer(self.config)

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
