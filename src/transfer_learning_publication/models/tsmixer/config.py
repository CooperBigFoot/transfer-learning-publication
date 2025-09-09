from typing import ClassVar

from ..base.base_config import BaseConfig


class TSMixerConfig(BaseConfig):
    """Configuration class for TSMixer model.

    TSMixer is a model architecture for time series forecasting that uses MLP-based
    mixing layers to process temporal and feature dimensions separately.
    """

    # Define model-specific parameters
    MODEL_PARAMS: ClassVar[list[str]] = [
        "static_embedding_size",
        "num_mixing_layers",
        "scheduler_patience",
        "scheduler_factor",
        "fusion_method",
        "hidden_size",
        "dropout",
    ]

    def __init__(
        self,
        input_len: int,
        output_len: int,
        input_size: int,
        static_size: int = 0,
        future_input_size: int | None = None,
        hidden_size: int = 64,
        static_embedding_size: int = 10,
        num_mixing_layers: int = 5,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        group_identifier: str = "gauge_id",
        scheduler_patience: int = 2,
        scheduler_factor: float = 0.5,
        fusion_method: str = "add",
        use_rev_in: bool = True,
        **kwargs,
    ):
        """Initialize TSMixer configuration.

        Args:
            input_len: Length of the input sequence
            output_len: Length of the output sequence (forecast horizon)
            input_size: Number of input features
            static_size: Number of static features
            future_input_size: Number of future forcing features (defaults to max(1, input_size - 1))
            hidden_size: Size of hidden layers
            static_embedding_size: Size of static feature embedding
            num_mixing_layers: Number of mixing layers
            dropout: Dropout rate
            learning_rate: Initial learning rate
            group_identifier: Name of the column identifying catchment groups
            scheduler_patience: Patience for learning rate scheduler
            scheduler_factor: Factor for learning rate reduction
            fusion_method: Method to fuse historical and future representations ("add" or "concat")
            use_rev_in: Whether to use RevIN normalization (default True)
            **kwargs: Additional parameters
        """
        # Initialize base config with standard parameters
        super().__init__(
            input_len=input_len,
            output_len=output_len,
            input_size=input_size,
            static_size=static_size,
            future_input_size=future_input_size,
            learning_rate=learning_rate,
            group_identifier=group_identifier,
            use_rev_in=use_rev_in,
            **kwargs,
        )

        # Set model-specific parameters
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.static_embedding_size = static_embedding_size
        self.num_mixing_layers = num_mixing_layers
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.fusion_method = fusion_method

        # Validate parameters
        if fusion_method not in ["add", "concat"]:
            raise ValueError(f"Invalid fusion_method: {fusion_method}. Must be 'add' or 'concat'.")

        if self.num_mixing_layers < 1:
            raise ValueError(f"num_mixing_layers must be at least 1, got {self.num_mixing_layers}")
