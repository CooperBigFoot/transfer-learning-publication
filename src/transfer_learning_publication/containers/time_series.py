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
    """

    def __init__(
        self,
        group_tensors: dict[str, torch.Tensor],  # {group_identifier: Tensor[timesteps, features]}
        feature_names: list[str],  # Ordered list of feature names
        date_ranges: dict[str, tuple[datetime, datetime]],  # {group_identifier: (start, end)}
        validate: bool = True,
    ):
        """
        Args:
            group_tensors: Dictionary mapping group identifiers to their time series tensors.
                          Each tensor shape: (n_timesteps, n_features)
            feature_names: Ordered list of feature names corresponding to tensor columns
            date_ranges: Dictionary mapping group identifiers to (start_date, end_date) tuples
            validate: If True, validate data integrity (no NaNs, consistent shapes, etc.)

        Raises:
            ValueError: If validation fails (NaNs present, inconsistent features, etc.)
        """
        self._group_tensors = group_tensors
        self._feature_names = list(feature_names)  # Copy to ensure immutability
        self._date_ranges = dict(date_ranges)  # Copy to ensure immutability

        # Build feature index mapping
        self._feature_indices = {name: idx for idx, name in enumerate(self._feature_names)}

        # Cache for computed properties
        self._n_features = len(self._feature_names)
        self._group_ids = sorted(self._group_tensors.keys())

        if validate:
            self.validate()

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
        if group_identifier not in self._group_tensors:
            raise KeyError(f"Group '{group_identifier}' not found in collection")

        tensor = self._group_tensors[group_identifier]

        # Validate indices
        if start_idx < 0 or end_idx > tensor.shape[0]:
            raise IndexError(
                f"Index range [{start_idx}:{end_idx}) out of bounds for group '{group_identifier}' "
                f"with length {tensor.shape[0]}"
            )

        if start_idx >= end_idx:
            raise ValueError(f"start_idx ({start_idx}) must be less than end_idx ({end_idx})")

        return tensor[start_idx:end_idx]

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
        return self._group_ids.copy()

    @property
    def feature_names(self) -> list[str]:
        """Get ordered list of feature names."""
        return self._feature_names.copy()

    @property
    def feature_indices(self) -> dict[str, int]:
        """Get mapping from feature names to column indices."""
        return self._feature_indices.copy()

    @property
    def date_ranges(self) -> dict[str, tuple[datetime, datetime]]:
        """Get date ranges for each group."""
        return self._date_ranges.copy()

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
        if group_identifier not in self._group_tensors:
            raise KeyError(f"Group '{group_identifier}' not found")
        return self._group_tensors[group_identifier].shape[0]

    def get_n_features(self) -> int:
        """Get number of features (consistent across all groups)."""
        return self._n_features

    def get_total_timesteps(self) -> int:
        """Get total timesteps across all groups."""
        return sum(tensor.shape[0] for tensor in self._group_tensors.values())

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
        if group_identifier not in self._date_ranges:
            raise KeyError(f"Group '{group_identifier}' not found in date_ranges")

        start_date, _ = self._date_ranges[group_identifier]
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
        if group_identifier not in self._date_ranges:
            raise KeyError(f"Group '{group_identifier}' not found in date_ranges")

        start_date, end_date = self._date_ranges[group_identifier]

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
        - All groups in date_ranges have corresponding tensors
        - All groups have both tensor and date range

        Raises:
            ValueError: If any validation check fails
        """
        # Check that all groups have both tensor and date range
        tensor_groups = set(self._group_tensors.keys())
        date_groups = set(self._date_ranges.keys())

        if tensor_groups != date_groups:
            missing_dates = tensor_groups - date_groups
            missing_tensors = date_groups - tensor_groups
            msg = []
            if missing_dates:
                msg.append(f"Groups with tensors but no dates: {missing_dates}")
            if missing_tensors:
                msg.append(f"Groups with dates but no tensors: {missing_tensors}")
            raise ValueError(". ".join(msg))

        # Validate each group
        for group_id in self._group_tensors:
            tensor = self._group_tensors[group_id]

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
            start_date, end_date = self._date_ranges[group_id]
            expected_days = (end_date - start_date).days + 1  # +1 because end is inclusive

            if tensor.shape[0] != expected_days:
                raise ValueError(
                    f"Group '{group_id}' has {tensor.shape[0]} timesteps but date range "
                    f"[{start_date}, {end_date}] implies {expected_days} days"
                )

        logger.info(f"Validation passed for {len(self._group_tensors)} groups")

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
        if not self._group_tensors:
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

        lengths = [tensor.shape[0] for tensor in self._group_tensors.values()]

        # Calculate memory usage (using actual tensor dtypes)
        memory_bytes = sum(tensor.numel() * tensor.element_size() for tensor in self._group_tensors.values())
        memory_mb = memory_bytes / (1024 * 1024)

        # Find overall date range
        all_start_dates = [start for start, _ in self._date_ranges.values()]
        all_end_dates = [end for _, end in self._date_ranges.values()]

        return {
            "n_groups": len(self._group_tensors),
            "n_features": self._n_features,
            "total_timesteps": sum(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_length": sum(lengths) / len(lengths),
            "memory_mb": round(memory_mb, 2),
            "date_range": (min(all_start_dates), max(all_end_dates)),
        }

    def __repr__(self) -> str:
        """Readable representation showing key stats."""
        stats = self.summary()
        return (
            f"TimeSeriesCollection("
            f"n_groups={stats['n_groups']}, "
            f"n_features={stats['n_features']}, "
            f"total_timesteps={stats['total_timesteps']:,}, "
            f"memory_mb={stats['memory_mb']:.1f})"
        )

    def __len__(self) -> int:
        """Number of groups in the collection."""
        return len(self._group_tensors)

    def __contains__(self, group_identifier: str) -> bool:
        """Check if group exists in collection."""
        return group_identifier in self._group_tensors
