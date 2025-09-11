"""PyTorch Lightning module for TiDE model."""

from typing import Any

import torch

from ..base.base_lit_model import BaseLitModel
from ..model_factory import register_model
from .config import TiDEConfig
from .model import TiDEModel


@register_model("tide", config_class=TiDEConfig)
class LitTiDE(BaseLitModel):
    """PyTorch Lightning Module implementation of TiDE.

    This class extends BaseLitModel to provide a standardized interface for training,
    validation, and testing of the TiDE model within our hydrological forecasting framework.
    """

    def __init__(
        self,
        config: TiDEConfig | dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the LitTiDE module.

        Args:
            config: TiDE configuration as a TiDEConfig instance or dict.
                    If None, config will be reconstructed from kwargs (for checkpoint loading).
            **kwargs: Individual hyperparameters (used when loading from checkpoint).
        """
        # Handle different initialization patterns
        if config is None and kwargs:
            # Loading from checkpoint - reconstruct config from kwargs
            tide_config = TiDEConfig(**kwargs)
            # Initialize base with the reconstructed config
            super().__init__(tide_config)
        elif isinstance(config, dict):
            # Dict config provided
            tide_config = TiDEConfig.from_dict(config)
            super().__init__(tide_config)
        elif config is not None:
            # TiDEConfig instance provided
            tide_config = config
            super().__init__(tide_config)
        else:
            raise ValueError("Either config or hyperparameters must be provided")

        # Create the TiDE model
        self.model = TiDEModel(self.config)

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor | None = None,
        future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass that delegates to the TiDE model.

        Args:
            x: Historical input features [batch_size, input_len, input_size]
            static: Static features [batch_size, static_size]
            future: Future forcing data [batch_size, output_len, future_input_size]

        Returns:
            Predictions [batch_size, output_len, 1]
        """
        return self.model(x, static, future)
