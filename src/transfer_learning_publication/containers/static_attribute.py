import logging

import torch

logger = logging.getLogger(__name__)


class StaticAttributeCollection:
    """
    Immutable collection of static attributes for multiple groups.

    This class stores static (non-time-varying) attributes in memory as tensors.
    All data is validated at construction time to ensure no NaNs and consistent structure.

    Data Organization:
    - Each group has a fixed set of attributes
    - Attributes are in the same order for all groups
    - All groups must have the same number of attributes
    - Groups are accessed by integer indices for performance
    """

    def __init__(
        self,
        tensors: list[torch.Tensor],  # List of tensors, one per group
        attribute_names: list[str],  # Ordered list of attribute names
        group_identifiers: list[str],  # Ordered list of group identifiers
        validate: bool = True,
    ):
        """
        Args:
            tensors: List of attribute tensors, one per group.
                    Each tensor shape: (n_attributes,)
            attribute_names: Ordered list of attribute names corresponding to tensor values
            group_identifiers: Ordered list of group identifiers (same order as tensors)
            validate: If True, validate data integrity (no NaNs, consistent shapes, etc.)

        Raises:
            ValueError: If validation fails (NaNs present, inconsistent shapes, etc.)
        """
        self._tensors = tensors
        self._attribute_names = list(attribute_names)  # Copy to ensure immutability
        self._group_identifiers = list(group_identifiers)  # Copy to ensure immutability

        # Validate input consistency
        if len(self._tensors) != len(self._group_identifiers):
            raise ValueError(
                f"Number of tensors ({len(self._tensors)}) must match number of group identifiers "
                f"({len(self._group_identifiers)})"
            )

        # Build index mappings
        self._group_to_idx = {group: idx for idx, group in enumerate(self._group_identifiers)}
        self._attribute_indices = {name: idx for idx, name in enumerate(self._attribute_names)}

        # Cache for computed properties
        self._n_attributes = len(self._attribute_names)
        self._n_groups = len(self._tensors)

        if validate:
            self.validate()

    def get_group_attributes_by_idx(self, group_idx: int) -> torch.Tensor:
        """
        Get all attributes for a group by index (fast path).

        Args:
            group_idx: Integer index of the group

        Returns:
            Tensor of shape (n_attributes,)

        Raises:
            IndexError: If group_idx is out of bounds
        """
        if group_idx < 0 or group_idx >= self._n_groups:
            raise IndexError(f"Group index {group_idx} out of bounds [0, {self._n_groups})")

        return self._tensors[group_idx]

    def get_group_attribute_by_idx(self, group_idx: int, attribute: str) -> torch.Tensor:
        """
        Get specific attribute value for a group by index (fast path).

        Args:
            group_idx: Integer index of the group
            attribute: Attribute name

        Returns:
            Scalar tensor with the attribute value

        Raises:
            IndexError: If group_idx is out of bounds
            KeyError: If attribute not found
        """
        if attribute not in self._attribute_indices:
            raise KeyError(f"Attribute '{attribute}' not found. Available: {self._attribute_names}")

        attribute_idx = self._attribute_indices[attribute]
        attributes = self.get_group_attributes_by_idx(group_idx)
        return attributes[attribute_idx]

    def get_group_attributes(self, group_identifier: str) -> torch.Tensor:
        """
        Get all attributes for a group by identifier.

        Args:
            group_identifier: Group identifier

        Returns:
            Tensor of shape (n_attributes,)

        Raises:
            KeyError: If group_identifier not found
        """
        if group_identifier not in self._group_to_idx:
            raise KeyError(f"Group '{group_identifier}' not found in collection")

        group_idx = self._group_to_idx[group_identifier]
        return self._tensors[group_idx]

    def get_group_attribute(self, group_identifier: str, attribute: str) -> torch.Tensor:
        """
        Get specific attribute value for a group.

        Args:
            group_identifier: Group identifier
            attribute: Attribute name

        Returns:
            Scalar tensor with the attribute value

        Raises:
            KeyError: If group_identifier or attribute not found
        """
        if attribute not in self._attribute_indices:
            raise KeyError(f"Attribute '{attribute}' not found. Available: {self._attribute_names}")

        attribute_idx = self._attribute_indices[attribute]
        attributes = self.get_group_attributes(group_identifier)
        return attributes[attribute_idx]

    @property
    def group_identifiers(self) -> list[str]:
        """Get list of all group identifiers."""
        return self._group_identifiers.copy()

    @property
    def attribute_names(self) -> list[str]:
        """Get ordered list of attribute names."""
        return self._attribute_names.copy()

    @property
    def attribute_indices(self) -> dict[str, int]:
        """Get mapping from attribute names to indices."""
        return self._attribute_indices.copy()

    @property
    def group_to_idx(self) -> dict[str, int]:
        """Get mapping from group identifiers to indices."""
        return self._group_to_idx.copy()

    def get_n_attributes(self) -> int:
        """Get number of attributes (consistent across all groups)."""
        return self._n_attributes

    def get_n_groups(self) -> int:
        """Get number of groups."""
        return self._n_groups

    def validate(self) -> None:
        """
        Validate data integrity.

        Checks:
        - No NaN values in any tensor
        - All tensors have correct number of attributes
        - All tensors are 1-dimensional

        Raises:
            ValueError: If any validation check fails
        """
        if not self._tensors:
            logger.warning("StaticAttributeCollection is empty")
            return

        # Validate each group
        for idx, tensor in enumerate(self._tensors):
            group_id = self._group_identifiers[idx]

            # Check tensor dimension
            if tensor.ndim != 1:
                raise ValueError(
                    f"Group '{group_id}' tensor has {tensor.ndim} dimensions, expected 1. Shape: {tensor.shape}"
                )

            # Check for NaNs
            if torch.isnan(tensor).any():
                nan_count = torch.isnan(tensor).sum().item()
                nan_attrs = [self._attribute_names[i] for i in range(len(tensor)) if torch.isnan(tensor[i])]
                raise ValueError(
                    f"Group '{group_id}' contains {nan_count} NaN values in attributes: {nan_attrs}. "
                    "This indicates a problem in upstream data processing."
                )

            # Check number of attributes
            if tensor.shape[0] != self._n_attributes:
                raise ValueError(f"Group '{group_id}' has {tensor.shape[0]} attributes, expected {self._n_attributes}")

        logger.info(f"Validation passed for {self._n_groups} groups with {self._n_attributes} attributes")

    def summary(self) -> dict:
        """
        Get summary statistics about the collection.

        Returns:
            Dictionary with keys:
            - n_groups: Number of groups
            - n_attributes: Number of attributes
            - memory_mb: Approximate memory usage in MB
            - attribute_names: List of attribute names
            - has_missing_values: Whether any NaN values exist
        """
        if not self._tensors:
            return {
                "n_groups": 0,
                "n_attributes": self._n_attributes,
                "memory_mb": 0.0,
                "attribute_names": self._attribute_names,
                "has_missing_values": False,
            }

        # Calculate memory usage
        memory_bytes = sum(tensor.numel() * tensor.element_size() for tensor in self._tensors)
        memory_mb = memory_bytes / (1024 * 1024)

        # Check for any NaN values (without raising exception)
        has_missing = any(torch.isnan(tensor).any().item() for tensor in self._tensors)

        return {
            "n_groups": self._n_groups,
            "n_attributes": self._n_attributes,
            "memory_mb": round(memory_mb, 4),
            "attribute_names": self._attribute_names,
            "has_missing_values": has_missing,
        }

    def __repr__(self) -> str:
        """Readable representation showing key stats."""
        stats = self.summary()
        return (
            f"StaticAttributeCollection("
            f"n_groups={stats['n_groups']}, "
            f"n_attributes={stats['n_attributes']}, "
            f"memory_mb={stats['memory_mb']:.4f})"
        )

    def __len__(self) -> int:
        """Number of groups in the collection."""
        return self._n_groups

    def __contains__(self, group_identifier: str) -> bool:
        """Check if group exists in collection."""
        return group_identifier in self._group_to_idx
