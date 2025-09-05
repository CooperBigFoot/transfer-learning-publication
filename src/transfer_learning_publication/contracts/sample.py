from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Sample:
    """
    Immutable representation of a single training sample.

    This is the contract between Dataset and DataLoader.
    All tensors are expected to be on CPU at this stage.

    Attributes:
        X: Input time series features. Shape: (input_length, n_input_features)
           For autoregressive models, includes historical target values.
           Features are ordered as specified in DatasetConfig.

        y: Target values to predict. Shape: (output_length,) or (output_length, 1)
           Contains only the target variable values for the prediction window.

        static: Time-invariant features for this group. Shape: (n_static_features,)
                Examples: catchment area, elevation, soil type

        future: Known future covariates. Shape: (output_length, n_future_features)
                Features that are known at prediction time (e.g., weather forecasts)
                Empty tensor with shape (output_length, 0) if no future features.

        group_identifier: Identifier for the group (e.g., gauge_id, basin_id)
                   Useful for analysis, debugging, and grouped operations.

        input_end_date: Timestamp (milliseconds) of the last input timestep.
                       Useful for temporal analysis and date-aware models.
                       None if date tracking is disabled.
    """

    X: torch.Tensor
    y: torch.Tensor
    static: torch.Tensor
    future: torch.Tensor
    group_identifier: str
    input_end_date: int | None = None

    def __post_init__(self):
        """Validate tensor shapes and types."""
        if not isinstance(self.X, torch.Tensor):
            raise TypeError(f"X must be torch.Tensor, got {type(self.X)}")
        if not isinstance(self.y, torch.Tensor):
            raise TypeError(f"y must be torch.Tensor, got {type(self.y)}")
        if not isinstance(self.static, torch.Tensor):
            raise TypeError(f"static must be torch.Tensor, got {type(self.static)}")
        if not isinstance(self.future, torch.Tensor):
            raise TypeError(f"future must be torch.Tensor, got {type(self.future)}")

        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D (time, features), got shape {self.X.shape}")

        if self.y.ndim not in (1, 2):
            raise ValueError(f"y must be 1D or 2D, got shape {self.y.shape}")

        if self.static.ndim != 1:
            raise ValueError(f"static must be 1D, got shape {self.static.shape}")

        if self.future.ndim != 2:
            raise ValueError(f"future must be 2D (time, features), got shape {self.future.shape}")

        if self.future.shape[0] != self.y.shape[0]:
            raise ValueError(f"future length ({self.future.shape[0]}) must match output length ({self.y.shape[0]})")
