"""PyTorch Lightning module for EA-LSTM model."""

from typing import Any

import torch

from ..base.base_lit_model import BaseLitModel
from ..model_factory import register_model
from .config import EALSTMConfig
from .model import EALSTM, BiEALSTM


@register_model("ealstm", config_class=EALSTMConfig)
class LitEALSTM(BaseLitModel):
    """PyTorch Lightning Module implementation of EA-LSTM.

    This class extends BaseLitModel to provide a standardized interface for training,
    validation, and testing of the EA-LSTM model within our hydrological forecasting framework.
    """

    def __init__(
        self,
        config: EALSTMConfig | dict[str, Any],
    ) -> None:
        """
        Initialize the LitEALSTM module.

        Args:
            config: EA-LSTM configuration as an EALSTMConfig instance or dict
        """
        # Convert dict config to EALSTMConfig if needed
        ealstm_config = EALSTMConfig.from_dict(config) if isinstance(config, dict) else config

        # Initialize base lightning model with the config
        super().__init__(ealstm_config)

        # Create the appropriate model based on configuration
        if ealstm_config.bidirectional:
            self.model = BiEALSTM(ealstm_config)
        else:
            self.model = EALSTM(ealstm_config)

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor | None = None,
        future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass that delegates to the EA-LSTM model.

        Args:
            x: Historical input features [batch_size, input_len, input_size]
            static: Static features [batch_size, static_size]
            future: Future forcing data [batch_size, output_len, future_input_size]

        Returns:
            Predictions [batch_size, output_len, 1]
        """
        return self.model(x, static, future)
