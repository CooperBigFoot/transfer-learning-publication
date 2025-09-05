from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Batch:
    """
    Immutable representation of a batch of samples for training.

    This is the contract between DataLoader and LightningModule.
    All tensors should have batch dimension as the first dimension.

    Attributes:
        X: Batched input sequences. Shape: (batch_size, input_length, n_input_features)

        y: Batched target values. Shape: (batch_size, output_length) or
           (batch_size, output_length, 1) depending on model requirements

        static: Batched static features. Shape: (batch_size, n_static_features)

        future: Batched future covariates. Shape: (batch_size, output_length, n_future_features)
                Can be empty with shape (batch_size, output_length, 0)

        group_identifiers: List of group identifiers for each sample in the batch.
                    Length: batch_size. Useful for debugging and analysis.

        input_end_dates: Tensor of input end timestamps for each sample.
                        Shape: (batch_size,). None if date tracking is disabled.
    """

    X: torch.Tensor
    y: torch.Tensor
    static: torch.Tensor
    future: torch.Tensor
    group_identifiers: list[str]
    input_end_dates: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return self.X.shape[0]

    @property
    def input_length(self) -> int:
        """Get the input sequence length."""
        return self.X.shape[1]

    @property
    def output_length(self) -> int:
        """Get the output sequence length."""
        return self.y.shape[1]

    @property
    def n_input_features(self) -> int:
        """Get the number of input features."""
        return self.X.shape[2]

    @property
    def n_static_features(self) -> int:
        """Get the number of static features."""
        return self.static.shape[1]

    @property
    def n_future_features(self) -> int:
        """Get the number of future features."""
        return self.future.shape[2]

    def __post_init__(self):
        """Validate batch consistency."""
        batch_size = self.X.shape[0]

        # Check all tensors have same batch size
        if self.y.shape[0] != batch_size:
            raise ValueError(f"y batch size ({self.y.shape[0]}) doesn't match X ({batch_size})")

        if self.static.shape[0] != batch_size:
            raise ValueError(f"static batch size ({self.static.shape[0]}) doesn't match X ({batch_size})")

        if self.future.shape[0] != batch_size:
            raise ValueError(f"future batch size ({self.future.shape[0]}) doesn't match X ({batch_size})")

        if len(self.group_identifiers) != batch_size:
            raise ValueError(
                f"group_identifiers length ({len(self.group_identifiers)}) doesn't match batch size ({batch_size})"
            )

        if self.input_end_dates is not None and self.input_end_dates.shape[0] != batch_size:
            raise ValueError(
                f"input_end_dates size ({self.input_end_dates.shape[0]}) doesn't match batch size ({batch_size})"
            )

        # Validate dimensions
        if self.X.ndim != 3:
            raise ValueError(f"X must be 3D (batch, time, features), got shape {self.X.shape}")

        if self.y.ndim not in (2, 3):
            raise ValueError(f"y must be 2D or 3D, got shape {self.y.shape}")

        if self.static.ndim != 2:
            raise ValueError(f"static must be 2D (batch, features), got shape {self.static.shape}")

        if self.future.ndim != 3:
            raise ValueError(f"future must be 3D (batch, time, features), got shape {self.future.shape}")

    def to(self, device: torch.device) -> "Batch":
        """
        Move all tensors to specified device.

        Args:
            device: Target device

        Returns:
            New Batch with tensors on target device
        """
        return Batch(
            X=self.X.to(device),
            y=self.y.to(device),
            static=self.static.to(device),
            future=self.future.to(device),
            group_identifiers=self.group_identifiers,  # Keep as list of strings
            input_end_dates=self.input_end_dates.to(device) if self.input_end_dates is not None else None,
        )

    def as_dict(self) -> dict[str, torch.Tensor | list[str] | None]:
        """
        Convert to dictionary format for backward compatibility.

        Useful during migration from dict-based batches.
        """
        return {
            "X": self.X,
            "y": self.y,
            "static": self.static,
            "future": self.future,
            "group_identifiers": self.group_identifiers,
            "input_end_dates": self.input_end_dates,
        }
