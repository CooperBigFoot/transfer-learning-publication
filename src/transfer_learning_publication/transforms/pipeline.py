import copy
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .base import BaseTransform


class BasePipeline(ABC):
    """Abstract base class for transformation pipelines.

    Provides common functionality for pipeline implementations that apply
    transforms to data with group identifiers.

    Convention: Last column of input array is the group identifier.
    Features to transform are all columns except the last one.
    """

    def __init__(self, transforms: list[BaseTransform]):
        """Initialize pipeline with list of transforms to chain.

        Args:
            transforms: List of BaseTransform instances to apply in sequence
        """
        self.transforms = transforms
        self._is_fitted = False

    def _validate_input(self, X: np.ndarray) -> None:
        """Validate input array format."""
        if not isinstance(X, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if X.ndim != 2:
            raise ValueError(f"Input must be 2D array, got {X.ndim}D")
        if X.shape[1] < 2:
            raise ValueError("Input must have at least 2 columns (features + group identifier)")

    def _clone_transforms(self) -> list[BaseTransform]:
        """Create deep copies of all transforms."""
        return [copy.deepcopy(transform) for transform in self.transforms]

    def _extract_features_and_groups(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract features and group identifiers from input array.

        Args:
            X: Input array where X[:, :-1] are features and X[:, -1] is group ID

        Returns:
            Tuple of (features, groups)
        """
        return X[:, :-1], X[:, -1]

    def _check_fitted_state(self, method_name: str) -> None:
        """Check if pipeline is fitted before applying transforms.

        Args:
            method_name: Name of the method being called (for error message)

        Raises:
            RuntimeError: If pipeline hasn't been fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError(f"Pipeline must be fitted before {method_name}()")

    def _apply_transforms_sequence(self, data: np.ndarray, transforms: list[BaseTransform]) -> np.ndarray:
        """Apply a sequence of transforms in order.

        Args:
            data: Input data to transform
            transforms: List of fitted transforms to apply

        Returns:
            Transformed data
        """
        current_data = data
        for transform in transforms:
            current_data = transform.transform(current_data)
        return current_data

    def _apply_inverse_transforms_sequence(self, data: np.ndarray, transforms: list[BaseTransform]) -> np.ndarray:
        """Apply a sequence of inverse transforms in reverse order.

        Args:
            data: Input data to inverse transform
            transforms: List of fitted transforms to apply in reverse

        Returns:
            Inverse transformed data
        """
        current_data = data
        for transform in reversed(transforms):
            current_data = transform.inverse_transform(current_data)
        return current_data

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BasePipeline":
        """Fit the pipeline to training data."""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted pipeline."""
        pass

    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data using fitted pipeline."""
        pass


class PerBasinPipeline(BasePipeline):
    """Pipeline that applies transforms separately to each basin/group.

    Convention: Last column of input array is the group identifier.
    Features to transform are all columns except the last one.
    """

    def __init__(self, transforms: list[BaseTransform]):
        """Initialize pipeline with list of transforms to chain.

        Args:
            transforms: List of BaseTransform instances to apply in sequence
        """
        super().__init__(transforms)
        self.fitted_pipelines: dict[Any, list[BaseTransform]] = {}

    def _get_unique_groups(self, X: np.ndarray) -> np.ndarray:
        groups = X[:, -1]
        return np.unique(groups)

    def fit(self, X: np.ndarray) -> "PerBasinPipeline":
        """Fit separate transform chains for each group.

        Args:
            X: Input array where X[:, :-1] are features and X[:, -1] is group ID

        Returns:
            Self for method chaining
        """
        self._validate_input(X)

        # Extract features and group identifiers
        X_features, groups = self._extract_features_and_groups(X)
        unique_groups = self._get_unique_groups(X)

        # Fit separate pipeline for each group
        self.fitted_pipelines = {}

        for group_id in unique_groups:
            # Get data for this group
            group_mask = groups == group_id
            group_features = X_features[group_mask]

            # Clone transforms for this group
            group_transforms = self._clone_transforms()

            # Fit transforms in sequence
            current_data = group_features
            for transform in group_transforms:
                transform.fit(current_data)
                current_data = transform.transform(current_data)

            self.fitted_pipelines[group_id] = group_transforms

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted pipelines.

        Only groups that were seen during fit() will be transformed.
        Unseen groups will pass through with zeros in feature columns.

        Args:
            X: Input array where X[:, :-1] are features and X[:, -1] is group ID

        Returns:
            Transformed array with same shape as input
        """
        self._check_fitted_state("transform")
        self._validate_input(X)

        # Extract features and groups
        X_features, groups = self._extract_features_and_groups(X)
        unique_groups = np.unique(groups)

        # Check for unfitted groups
        unfitted_groups = set(unique_groups) - set(self.fitted_pipelines.keys())
        if unfitted_groups:
            warnings.warn(
                f"Groups {unfitted_groups} were not seen during fit and will not be transformed",
                RuntimeWarning,
                stacklevel=2,
            )

        # Initialize output array
        X_transformed = np.zeros_like(X)
        X_transformed[:, -1] = groups  # Copy group column unchanged

        # Transform each group
        for group_id, group_transforms in self.fitted_pipelines.items():
            group_mask = groups == group_id
            if not np.any(group_mask):
                continue

            group_features = X_features[group_mask]
            transformed_features = self._apply_transforms_sequence(group_features, group_transforms)
            X_transformed[group_mask, :-1] = transformed_features

        return X_transformed

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data using fitted pipelines.

        Only groups that were seen during fit() will be inverse transformed.
        Unseen groups will pass through with zeros in feature columns.

        Args:
            X: Transformed array where X[:, :-1] are features and X[:, -1] is group ID

        Returns:
            Inverse transformed array with same shape as input
        """
        self._check_fitted_state("inverse_transform")
        self._validate_input(X)

        # Extract features and groups
        X_features, groups = self._extract_features_and_groups(X)
        unique_groups = np.unique(groups)

        # Check for unfitted groups
        unfitted_groups = set(unique_groups) - set(self.fitted_pipelines.keys())
        if unfitted_groups:
            warnings.warn(
                f"Groups {unfitted_groups} were not seen during fit and will not be inverse transformed",
                RuntimeWarning,
                stacklevel=2,
            )

        # Initialize output array
        X_inverse = np.zeros_like(X)
        X_inverse[:, -1] = groups  # Copy group column unchanged

        # Inverse transform each group
        for group_id, group_transforms in self.fitted_pipelines.items():
            group_mask = groups == group_id
            if not np.any(group_mask):
                continue

            group_features = X_features[group_mask]
            inverse_features = self._apply_inverse_transforms_sequence(group_features, group_transforms)
            X_inverse[group_mask, :-1] = inverse_features

        return X_inverse

    def get_group_pipeline(self, group_id: Any) -> list[BaseTransform]:
        """Get fitted transform chain for a specific group.

        Args:
            group_id: Group identifier

        Returns:
            List of fitted transforms for the group
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before accessing group pipelines")

        if group_id not in self.fitted_pipelines:
            available_groups = list(self.fitted_pipelines.keys())
            raise ValueError(f"Group {group_id} not found. Available groups: {available_groups}")

        return self.fitted_pipelines[group_id]


class GlobalPipeline(BasePipeline):
    """Pipeline that applies transforms globally across all groups.

    Pools all data together during fitting, ignoring group boundaries.

    Convention: Last column of input array is the group identifier.
    Features to transform are all columns except the last one.
    """

    def __init__(self, transforms: list[BaseTransform]):
        """Initialize pipeline with list of transforms to chain.

        Args:
            transforms: List of BaseTransform instances to apply in sequence
        """
        super().__init__(transforms)
        self.fitted_transforms: list[BaseTransform] = []

    def fit(self, X: np.ndarray) -> "GlobalPipeline":
        """Fit transforms on pooled data from all groups.

        Args:
            X: Input array where X[:, :-1] are features and X[:, -1] is group ID

        Returns:
            Self for method chaining
        """
        self._validate_input(X)

        # Extract features (ignore group identifiers for fitting)
        X_features, _ = self._extract_features_and_groups(X)

        # Clone transforms for fitting
        self.fitted_transforms = self._clone_transforms()

        # Fit transforms in sequence on pooled data
        current_data = X_features
        for transform in self.fitted_transforms:
            transform.fit(current_data)
            current_data = transform.transform(current_data)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using globally fitted transforms.

        Args:
            X: Input array where X[:, :-1] are features and X[:, -1] is group ID

        Returns:
            Transformed array with same shape as input
        """
        self._check_fitted_state("transform")
        self._validate_input(X)

        # Extract features and groups
        X_features, groups = self._extract_features_and_groups(X)

        # Apply transforms in sequence
        transformed_features = self._apply_transforms_sequence(X_features, self.fitted_transforms)

        # Reconstruct array with transformed features and original groups
        X_transformed = np.column_stack([transformed_features, groups])

        return X_transformed

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data using globally fitted transforms.

        Args:
            X: Transformed array where X[:, :-1] are features and X[:, -1] is group ID

        Returns:
            Inverse transformed array with same shape as input
        """
        self._check_fitted_state("inverse_transform")
        self._validate_input(X)

        # Extract features and groups
        X_features, groups = self._extract_features_and_groups(X)

        # Apply inverse transforms in reverse order
        inverse_features = self._apply_inverse_transforms_sequence(X_features, self.fitted_transforms)

        # Reconstruct array with inverse transformed features and original groups
        X_inverse = np.column_stack([inverse_features, groups])

        return X_inverse

    def get_fitted_transforms(self) -> list[BaseTransform]:
        """Get the fitted transform chain.

        Returns:
            List of fitted transforms

        Raises:
            RuntimeError: If pipeline hasn't been fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before accessing fitted transforms")

        return self.fitted_transforms
