from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ForecastOutput:
    """Contract between model test output and evaluation framework.

    Immutable container for model predictions and associated metadata
    from the test phase.
    """

    predictions: torch.Tensor  # [n_samples, output_len]
    observations: torch.Tensor  # [n_samples, output_len]
    group_identifiers: list[str]  # Basin/gauge IDs
    input_end_dates: torch.Tensor | None = None  # Timestamps if available

    def __post_init__(self):
        """Validate output consistency."""
        n_predictions = self.predictions.shape[0]
        n_observations = self.observations.shape[0]
        n_groups = len(self.group_identifiers)

        if n_predictions != n_observations:
            raise ValueError(
                f"Predictions ({n_predictions}) and observations ({n_observations}) must have same number of samples"
            )

        if n_predictions != n_groups:
            raise ValueError(
                f"Number of predictions ({n_predictions}) must match number of group identifiers ({n_groups})"
            )

        if self.input_end_dates is not None and self.input_end_dates.shape[0] != n_predictions:
            raise ValueError(
                f"Input end dates ({self.input_end_dates.shape[0]}) must match number of predictions ({n_predictions})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for compatibility."""
        return {
            "predictions": self.predictions,
            "observations": self.observations,
            "group_identifiers": self.group_identifiers,
            "input_end_dates": self.input_end_dates,
        }
