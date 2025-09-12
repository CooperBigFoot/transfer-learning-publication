# src/transfer_learning_publication/containers/time_series.py

import logging
from datetime import datetime, timedelta

import torch

logger = logging.getLogger(__name__)


class TimeSeriesCollection:
    """
    Immutable collection of time series data for multiple groups.

    This class stores time series data in memory as tensors.
    All data is validated at construction time to ensure no NaNs and consistent structure.

    Data Organization:
    - Each group has a continuous time series with consistent features
    - Features are in the same order for all groups
    - Temporal coverage is continuous (no gaps in dates)
    - Date ranges are stored as metadata, positions are used for indexing
    - Groups are accessed by integer indices for performance
    """

    def __init__(
        self,
        tensors: list[torch.Tensor],  # List of tensors, one per group
        feature_names: list[str],  # Ordered list of feature names
        date_ranges: list[tuple[datetime, datetime]],  # List of (start, end) tuples
        group_identifiers: list[str],  # Ordered list of group identifiers
        target_was_filled: list[torch.Tensor] | None = None,  # Optional target filled flags
        validate: bool = True,
    ):
        """
        Args:
            tensors: List of time series tensors, one per group.
                    Each tensor shape: (n_timesteps, n_features)
            feature_names: Ordered list of feature names corresponding to tensor columns
            date_ranges: List of (start_date, end_date) tuples (same order as tensors)
            group_identifiers: Ordered list of group identifiers (same order as tensors)
            target_was_filled: Optional list of 1D binary tensors (one per group).
                              Each tensor has shape (n_timesteps,) with values:
                              0 = original data, 1 = filled/imputed data
            validate: If True, validate data integrity (no NaNs, consistent shapes, etc.)

        Raises:
            ValueError: If validation fails (NaNs present, inconsistent features, etc.)
        """
        self._tensors = tensors
        self._feature_names = list(feature_names)  # Copy to ensure immutability
        self._date_ranges = list(date_ranges)  # Copy to ensure immutability
        self._group_identifiers = list(group_identifiers)  # Copy to ensure immutability
        self._target_was_filled = target_was_filled  # Store target filled flags

        # Validate input consistency
        if len(self._tensors) != len(self._group_identifiers):
            raise ValueError(
                f"Number of tensors ({len(self._tensors)}) must match number of group identifiers "
                f"({len(self._group_identifiers)})"
            )
        if len(self._tensors) != len(self._date_ranges):
            raise ValueError(
                f"Number of tensors ({len(self._tensors)}) must match number of date ranges ({len(self._date_ranges)})"
            )
        
        # Validate target_was_filled if provided
        if self._target_was_filled is not None:
            if len(self._target_was_filled) != len(self._tensors):
                raise ValueError(
                    f"Number of target_was_filled tensors ({len(self._target_was_filled)}) must match "
                    f"number of tensors ({len(self._tensors)})"
                )

        # Build index mappings
        self._group_to_idx = {group: idx for idx, group in enumerate(self._group_identifiers)}
        self._feature_indices = {name: idx for idx, name in enumerate(self._feature_names)}

        # Cache for computed properties
        self._n_features = len(self._feature_names)
        self._n_groups = len(self._tensors)

        if validate:
            self.validate()

    def get_group_series_by_idx(self, group_idx: int, start_idx: int, end_idx: int) -> torch.Tensor:
        """
        Get time series slice for a group by index (fast path).

        Args:
            group_idx: Integer index of the group
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)

        Returns:
            Tensor of shape (end_idx - start_idx, n_features)

        Raises:
            IndexError: If group_idx or time indices are out of bounds
            ValueError: If start_idx >= end_idx
        """
        if group_idx < 0 or group_idx >= self._n_groups:
            raise IndexError(f"Group index {group_idx} out of bounds [0, {self._n_groups})")

        tensor = self._tensors[group_idx]

        # Validate indices
        if start_idx < 0 or end_idx > tensor.shape[0]:
            raise IndexError(
                f"Index range [{start_idx}:{end_idx}) out of bounds for group at index {group_idx} "
                f"with length {tensor.shape[0]}"
            )

        if start_idx >= end_idx:
            raise ValueError(f"start_idx ({start_idx}) must be less than end_idx ({end_idx})")

        return tensor[start_idx:end_idx]

    def get_group_series(self, group_identifier: str, start_idx: int, end_idx: int) -> torch.Tensor:
        """
        Get time series slice for a group.

        Args:
            group_identifier: Group identifier
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)

        Returns:
            Tensor of shape (end_idx - start_idx, n_features)

        Raises:
            KeyError: If group_identifier not found
            IndexError: If indices are out of bounds
        """
        if group_identifier not in self._group_to_idx:
            raise KeyError(f"Group '{group_identifier}' not found in collection")

        group_idx = self._group_to_idx[group_identifier]
        tensor = self._tensors[group_idx]

        # Validate indices
        if start_idx < 0 or end_idx > tensor.shape[0]:
            raise IndexError(
                f"Index range [{start_idx}:{end_idx}) out of bounds for group '{group_identifier}' "
                f"with length {tensor.shape[0]}"
            )

        if start_idx >= end_idx:
            raise ValueError(f"start_idx ({start_idx}) must be less than end_idx ({end_idx})")

        return tensor[start_idx:end_idx]

    def get_group_feature_by_idx(self, group_idx: int, feature: str, start_idx: int, end_idx: int) -> torch.Tensor:
        """
        Get specific feature slice for a group by index (fast path).

        Args:
            group_idx: Integer index of the group
            feature: Feature name
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)

        Returns:
            Tensor of shape (end_idx - start_idx,)

        Raises:
            IndexError: If group_idx or time indices are out of bounds
            KeyError: If feature not found
        """
        if feature not in self._feature_indices:
            raise KeyError(f"Feature '{feature}' not found. Available: {self._feature_names}")

        feature_idx = self._feature_indices[feature]
        series = self.get_group_series_by_idx(group_idx, start_idx, end_idx)
        return series[:, feature_idx]

    def get_group_feature(self, group_identifier: str, feature: str, start_idx: int, end_idx: int) -> torch.Tensor:
        """
        Get specific feature slice for a group.

        Args:
            group_identifier: Group identifier
            feature: Feature name
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)

        Returns:
            Tensor of shape (end_idx - start_idx,)

        Raises:
            KeyError: If group_identifier or feature not found
            IndexError: If indices are out of bounds
        """
        if feature not in self._feature_indices:
            raise KeyError(f"Feature '{feature}' not found. Available: {self._feature_names}")

        feature_idx = self._feature_indices[feature]
        series = self.get_group_series(group_identifier, start_idx, end_idx)
        return series[:, feature_idx]

    @property
    def group_identifiers(self) -> list[str]:
        """Get list of all group identifiers."""
        return self._group_identifiers.copy()

    @property
    def feature_names(self) -> list[str]:
        """Get ordered list of feature names."""
        return self._feature_names.copy()

    @property
    def feature_indices(self) -> dict[str, int]:
        """Get mapping from feature names to column indices."""
        return self._feature_indices.copy()

    @property
    def group_to_idx(self) -> dict[str, int]:
        """Get mapping from group identifiers to indices."""
        return self._group_to_idx.copy()

    @property
    def date_ranges(self) -> dict[str, tuple[datetime, datetime]]:
        """Get date ranges for each group."""
        # Reconstruct as dictionary for compatibility
        return {self._group_identifiers[i]: self._date_ranges[i] for i in range(self._n_groups)}
    
    @property
    def target_was_filled(self) -> list[torch.Tensor] | None:
        """Get target filled flags if available."""
        return self._target_was_filled

    def get_group_length_by_idx(self, group_idx: int) -> int:
        """
        Get number of timesteps for a group by index (fast path).

        Args:
            group_idx: Integer index of the group

        Returns:
            Number of timesteps

        Raises:
            IndexError: If group_idx is out of bounds
        """
        if group_idx < 0 or group_idx >= self._n_groups:
            raise IndexError(f"Group index {group_idx} out of bounds [0, {self._n_groups})")
        return self._tensors[group_idx].shape[0]

    def get_group_length(self, group_identifier: str) -> int:
        """
        Get number of timesteps for a group.

        Args:
            group_identifier: Group identifier

        Returns:
            Number of timesteps

        Raises:
            KeyError: If group not found
        """
        if group_identifier not in self._group_to_idx:
            raise KeyError(f"Group '{group_identifier}' not found")
        group_idx = self._group_to_idx[group_identifier]
        return self._tensors[group_idx].shape[0]

    def get_n_features(self) -> int:
        """Get number of features (consistent across all groups)."""
        return self._n_features

    def get_total_timesteps(self) -> int:
        """Get total timesteps across all groups."""
        return sum(tensor.shape[0] for tensor in self._tensors)

    def index_to_date(self, group_identifier: str, index: int) -> datetime:
        """
        Convert positional index to actual date for a group.

        Args:
            group_identifier: Group identifier
            index: Positional index in the group's time series

        Returns:
            Corresponding datetime

        Note:
            Assumes daily frequency and continuous coverage.

        Raises:
            KeyError: If group not found
            IndexError: If index out of bounds
        """
        if group_identifier not in self._group_to_idx:
            raise KeyError(f"Group '{group_identifier}' not found")

        group_idx = self._group_to_idx[group_identifier]
        start_date, _ = self._date_ranges[group_idx]
        group_length = self.get_group_length(group_identifier)

        if index < 0 or index >= group_length:
            raise IndexError(f"Index {index} out of bounds for group with length {group_length}")

        return start_date + timedelta(days=index)

    def date_to_index(self, group_identifier: str, date: datetime) -> int:
        """
        Convert date to positional index for a group.

        Args:
            group_identifier: Group identifier
            date: Date to convert

        Returns:
            Positional index in the group's time series

        Raises:
            ValueError: If date is outside group's date range
            KeyError: If group not found
        """
        if group_identifier not in self._group_to_idx:
            raise KeyError(f"Group '{group_identifier}' not found")

        group_idx = self._group_to_idx[group_identifier]
        start_date, end_date = self._date_ranges[group_idx]

        # Normalize to date (remove time component) for consistent comparison
        from datetime import date as date_type

        if isinstance(date, datetime):
            date = date.date()
        elif not isinstance(date, date_type):
            raise TypeError(f"Expected datetime or date object, got {type(date)}")

        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()

        if date < start_date or date > end_date:
            raise ValueError(f"Date {date} outside range [{start_date}, {end_date}] for group '{group_identifier}'")

        delta = date - start_date
        return delta.days

    def validate(self) -> None:
        """
        Validate data integrity.

        Checks:
        - No NaN values in any tensor
        - All tensors have correct number of features
        - Date ranges match tensor lengths (assuming daily frequency)

        Raises:
            ValueError: If any validation check fails
        """
        if not self._tensors:
            logger.warning("TimeSeriesCollection is empty")
            return

        # Validate each group
        for idx, tensor in enumerate(self._tensors):
            group_id = self._group_identifiers[idx]

            # Check for NaNs
            if torch.isnan(tensor).any():
                nan_count = torch.isnan(tensor).sum().item()
                raise ValueError(
                    f"Group '{group_id}' contains {nan_count} NaN values. "
                    "This indicates a problem in upstream data processing."
                )

            # Check number of features
            if tensor.shape[1] != self._n_features:
                raise ValueError(f"Group '{group_id}' has {tensor.shape[1]} features, expected {self._n_features}")

            # Check date range consistency (assuming daily frequency)
            start_date, end_date = self._date_ranges[idx]
            expected_days = (end_date - start_date).days + 1  # +1 because end is inclusive

            if tensor.shape[0] != expected_days:
                raise ValueError(
                    f"Group '{group_id}' has {tensor.shape[0]} timesteps but date range "
                    f"[{start_date}, {end_date}] implies {expected_days} days"
                )
            
            # Check target_was_filled consistency if provided
            if self._target_was_filled is not None:
                flag_tensor = self._target_was_filled[idx]
                
                # Check shape - should be 1D with same length as timesteps
                if flag_tensor.dim() != 1:
                    raise ValueError(
                        f"Group '{group_id}' target_was_filled tensor must be 1D, got {flag_tensor.dim()}D"
                    )
                
                if flag_tensor.shape[0] != tensor.shape[0]:
                    raise ValueError(
                        f"Group '{group_id}' target_was_filled length ({flag_tensor.shape[0]}) must match "
                        f"number of timesteps ({tensor.shape[0]})"
                    )
                
                # Check values are binary (0 or 1)
                unique_vals = torch.unique(flag_tensor)
                if not torch.all((unique_vals == 0) | (unique_vals == 1)):
                    raise ValueError(
                        f"Group '{group_id}' target_was_filled must contain only 0 or 1, "
                        f"found values: {unique_vals.tolist()}"
                    )

        logger.info(f"Validation passed for {self._n_groups} groups")

    def summary(self) -> dict:
        """
        Get summary statistics about the collection.

        Returns:
            Dictionary with keys:
            - n_groups: Number of groups
            - n_features: Number of features
            - total_timesteps: Total timesteps across all groups
            - min_length: Shortest group time series
            - max_length: Longest group time series
            - avg_length: Average group time series length
            - memory_mb: Approximate memory usage in MB
            - date_range: Overall min and max dates
        """
        if not self._tensors:
            return {
                "n_groups": 0,
                "n_features": self._n_features,
                "total_timesteps": 0,
                "min_length": 0,
                "max_length": 0,
                "avg_length": 0.0,
                "memory_mb": 0.0,
                "date_range": (None, None),
            }

        lengths = [tensor.shape[0] for tensor in self._tensors]

        # Calculate memory usage (using actual tensor dtypes)
        memory_bytes = sum(tensor.numel() * tensor.element_size() for tensor in self._tensors)
        memory_mb = memory_bytes / (1024 * 1024)

        # Find overall date range
        all_start_dates = [start for start, _ in self._date_ranges]
        all_end_dates = [end for _, end in self._date_ranges]

        return {
            "n_groups": self._n_groups,
            "n_features": self._n_features,
            "total_timesteps": sum(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_length": sum(lengths) / len(lengths),
            "memory_mb": round(memory_mb, 2),
            "date_range": (min(all_start_dates), max(all_end_dates)),
            "has_target_filled_flags": self._target_was_filled is not None,
        }

    def __repr__(self) -> str:
        """Readable representation showing key stats."""
        stats = self.summary()
        return (
            f"TimeSeriesCollection("
            f"n_groups={stats['n_groups']}, "
            f"n_features={stats['n_features']}, "
            f"total_timesteps={stats['total_timesteps']:,}, "
            f"memory_mb={stats['memory_mb']:.1f}"
            f"{', has_target_filled_flags=True' if self._target_was_filled is not None else ''})"
        )

    def __len__(self) -> int:
        """Number of groups in the collection."""
        return self._n_groups

    def __contains__(self, group_identifier: str) -> bool:
        """Check if group exists in collection."""
        return group_identifier in self._group_to_idx
