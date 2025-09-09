"""
Configuration for the NaiveLastValue model, a simple baseline that repeats
the last observed value for the entire forecast horizon.
"""

from typing import ClassVar

from ..base.base_config import BaseConfig


class NaiveLastValueConfig(BaseConfig):
    """Configuration class for NaiveLastValue model.

    NaiveLastValue is a simple baseline model that forecasts by repeating
    the last observed value of the target variable for the entire forecast horizon.
    """

    # This model has no specific parameters beyond the standard ones
    MODEL_PARAMS: ClassVar[list[str]] = []

    def __init__(
        self,
        input_len: int,
        output_len: int,
        input_size: int,
        static_size: int = 0,
        future_input_size: int | None = None,
        learning_rate: float = 1e-10,  # Very small since this model doesn't learn
        group_identifier: str = "gauge_id",
        use_rev_in: bool = False,  # Disable RevIN for naive baseline
        **kwargs,
    ):
        """Initialize NaiveLastValue configuration.

        Args:
            input_len: Length of the input sequence
            output_len: Length of the output sequence (forecast horizon)
            input_size: Number of input features (target + past features)
            static_size: Number of static features (not used)
            future_input_size: Number of future forcing features (not used)
            learning_rate: Initial learning rate (minimal since model doesn't train)
            group_identifier: Name of the column identifying catchment groups
            use_rev_in: Whether to use RevIN normalization (default False for baseline)
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

        # Validate parameters
        if input_size < 1:
            raise ValueError("input_size must be at least 1")
        if output_len < 1:
            raise ValueError("output_len must be at least 1")
