import logging

import torch

from .time_series import TimeSeriesCollection

logger = logging.getLogger(__name__)


class SequenceIndex:
    """
    Immutable index of valid sequences in time series data.

    Maps from linear sequence indices (0 to n_sequences-1) to concrete sequence
    locations defined by (group_idx, start_idx, end_idx). This enables efficient
    random access to sequences during training without scanning for validity.

    Data Organization:
    - Each row represents one valid sequence
    - Sequences can overlap within the same group
    - Only valid sequences are stored (sparse representation)
    - All indices are integers for maximum performance
    - End indices are exclusive (Python slicing convention)

    The sequence finding logic is separated from storage - this class only
    stores and retrieves pre-computed sequence locations.
    """

    def __init__(
        self,
        sequences: torch.LongTensor,
        n_groups: int,
        input_length: int,
        output_length: int,
        validate: bool = True,
    ):
        """
        Initialize the sequence index with pre-computed valid sequences.

        Args:
            sequences: Tensor of shape (n_sequences, 3) where each row is
                      [group_idx, start_idx, end_idx]. End indices are exclusive.
            n_groups: Total number of groups in the dataset (for validation)
            input_length: Length of input sequences (for metadata/validation)
            output_length: Length of output sequences (for metadata/validation)
            validate: If True, validate index integrity

        Raises:
            ValueError: If validation fails
            TypeError: If sequences is not a LongTensor
        """
        # Ensure we have a LongTensor for index operations
        if not isinstance(sequences, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(sequences)}")

        if sequences.dtype != torch.long:
            logger.info(f"Converting sequences tensor from {sequences.dtype} to torch.long")
            sequences = sequences.long()

        self._sequences = sequences
        self._n_groups = n_groups
        self._input_length = input_length
        self._output_length = output_length
        self._total_length = input_length + output_length

        # Cache frequently accessed properties
        self._n_sequences = sequences.shape[0] if sequences.numel() > 0 else 0

        if validate:
            self._validate()

        # Compute group statistics for summary (after validation)
        self._compute_group_stats()

    def get_sequence_info(self, idx: int) -> tuple[int, int, int]:
        """
        Get sequence location by index.

        Args:
            idx: Sequence index in range [0, n_sequences)

        Returns:
            Tuple of (group_idx, start_idx, end_idx) where end_idx is exclusive

        Raises:
            IndexError: If idx is out of bounds
        """
        if not 0 <= idx < self._n_sequences:
            raise IndexError(f"Sequence index {idx} out of bounds [0, {self._n_sequences})")

        # Direct tensor indexing - single memory access
        sequence_info = self._sequences[idx]

        # Convert to Python ints for downstream use
        return (sequence_info[0].item(), sequence_info[1].item(), sequence_info[2].item())

    def get_sequence_info_batch(self, indices: list[int] | torch.LongTensor) -> torch.LongTensor:
        """
        Get multiple sequence locations at once (for potential batch operations).

        Args:
            indices: List or tensor of sequence indices

        Returns:
            Tensor of shape (len(indices), 3) with sequence information

        Raises:
            IndexError: If any index is out of bounds
        """
        if isinstance(indices, list):
            indices = torch.tensor(indices, dtype=torch.long)

        # Validate bounds
        if len(indices) > 0:
            min_idx = indices.min().item()
            max_idx = indices.max().item()
            if min_idx < 0 or max_idx >= self._n_sequences:
                raise IndexError(f"Index out of bounds. Range [{min_idx}, {max_idx}] not in [0, {self._n_sequences})")

        return self._sequences[indices]

    @property
    def n_sequences(self) -> int:
        """Total number of valid sequences."""
        return self._n_sequences

    @property
    def n_groups(self) -> int:
        """Total number of groups in the dataset."""
        return self._n_groups

    @property
    def input_length(self) -> int:
        """Length of input sequences."""
        return self._input_length

    @property
    def output_length(self) -> int:
        """Length of output sequences."""
        return self._output_length

    @property
    def total_length(self) -> int:
        """Total sequence length (input + output)."""
        return self._total_length

    def _validate(self) -> None:
        """
        Validate index integrity.

        Checks:
        - Correct tensor shape
        - Group indices in valid range [0, n_groups)
        - Start indices are non-negative
        - End indices are greater than start indices
        - Sequence lengths match expected total_length

        Raises:
            ValueError: If validation fails
        """
        if self._sequences.numel() == 0:
            logger.info("SequenceIndex is empty - no sequences to validate")
            return

        # Check tensor shape
        if self._sequences.ndim != 2 or self._sequences.shape[1] != 3:
            raise ValueError(f"Expected sequences tensor of shape (n_sequences, 3), got {self._sequences.shape}")

        # Extract columns for validation
        group_indices = self._sequences[:, 0]
        start_indices = self._sequences[:, 1]
        end_indices = self._sequences[:, 2]

        # Check group indices are in valid range
        min_group = group_indices.min().item()
        max_group = group_indices.max().item()
        if min_group < 0 or max_group >= self._n_groups:
            raise ValueError(
                f"Group indices out of range. Found range [{min_group}, {max_group}], expected [0, {self._n_groups})"
            )

        # Check start indices are non-negative
        min_start = start_indices.min().item()
        if min_start < 0:
            raise ValueError(f"Found negative start index: {min_start}")

        # Check end > start (exclusive end)
        sequence_lengths = end_indices - start_indices
        min_length = sequence_lengths.min().item()
        max_length = sequence_lengths.max().item()

        if min_length <= 0:
            invalid_mask = sequence_lengths <= 0
            invalid_count = invalid_mask.sum().item()
            raise ValueError(
                f"Found {invalid_count} sequences with non-positive length "
                f"(end_idx must be > start_idx for exclusive indexing)"
            )

        # Check sequence lengths match expected total_length
        if min_length != self._total_length or max_length != self._total_length:
            raise ValueError(
                f"Sequence lengths inconsistent. Expected all sequences to have "
                f"length {self._total_length}, but found range [{min_length}, {max_length}]"
            )

        logger.info(f"Validation passed: {self._n_sequences} sequences across {len(group_indices.unique())} groups")

    def _compute_group_stats(self) -> None:
        """Compute per-group statistics for summary."""
        if self._sequences.numel() == 0:
            self._sequences_per_group = {}
            return

        group_indices = self._sequences[:, 0]
        unique_groups, counts = torch.unique(group_indices, return_counts=True)

        self._sequences_per_group = {
            int(group_idx): int(count) for group_idx, count in zip(unique_groups.tolist(), counts.tolist(), strict=True)
        }

    def get_sequences_for_group(self, group_idx: int) -> torch.LongTensor:
        """
        Get all sequence indices that belong to a specific group.

        Args:
            group_idx: Group index to filter by

        Returns:
            1D tensor of sequence indices for this group

        Raises:
            ValueError: If group_idx is out of bounds
        """
        if not 0 <= group_idx < self._n_groups:
            raise ValueError(f"Group index {group_idx} out of bounds [0, {self._n_groups})")

        # Find all sequences for this group
        mask = self._sequences[:, 0] == group_idx
        sequence_indices = torch.where(mask)[0]

        return sequence_indices

    def summary(self) -> dict:
        """
        Get summary statistics about the index.

        Returns:
            Dictionary with keys:
            - n_sequences: Total number of sequences
            - n_groups_with_sequences: Number of groups that have at least one sequence
            - sequences_per_group: Dict mapping group_idx to sequence count
            - avg_sequences_per_group: Average sequences per group (excluding groups with 0)
            - min_sequences_per_group: Minimum sequences in any group with data
            - max_sequences_per_group: Maximum sequences in any group
            - memory_mb: Approximate memory usage in MB
        """
        if self._n_sequences == 0:
            return {
                "n_sequences": 0,
                "n_groups_with_sequences": 0,
                "sequences_per_group": {},
                "avg_sequences_per_group": 0.0,
                "min_sequences_per_group": 0,
                "max_sequences_per_group": 0,
                "memory_mb": 0.0,
            }

        # Memory usage
        memory_bytes = self._sequences.numel() * self._sequences.element_size()
        memory_mb = memory_bytes / (1024 * 1024)

        # Group statistics
        counts = list(self._sequences_per_group.values())

        return {
            "n_sequences": self._n_sequences,
            "n_groups_with_sequences": len(self._sequences_per_group),
            "sequences_per_group": self._sequences_per_group.copy(),
            "avg_sequences_per_group": sum(counts) / len(counts) if counts else 0.0,
            "min_sequences_per_group": min(counts) if counts else 0,
            "max_sequences_per_group": max(counts) if counts else 0,
            "memory_mb": round(memory_mb, 4),
        }

    def __repr__(self) -> str:
        """Readable representation showing key stats."""
        stats = self.summary()
        return (
            f"SequenceIndex("
            f"n_sequences={stats['n_sequences']:,}, "
            f"n_groups_with_sequences={stats['n_groups_with_sequences']}, "
            f"total_length={self._total_length}, "
            f"memory_mb={stats['memory_mb']:.4f})"
        )

    def __len__(self) -> int:
        """Number of valid sequences in the index."""
        return self._n_sequences

    @staticmethod
    def find_valid_sequences(
        time_series: "TimeSeriesCollection",
        target_feature: str,
        forcing_features: list[str],
        input_length: int,
        output_length: int,
    ) -> torch.LongTensor:
        """
        Find all valid sequences in the time series data.

        This method will be implemented to scan through the TimeSeriesCollection
        and identify all positions where valid sequences of the required length
        can be extracted (no NaN values in the relevant features).

        Args:
            time_series: TimeSeriesCollection to search for valid sequences
            target_feature: Name of the target feature to check for NaNs
            forcing_features: List of forcing features to check for NaNs
            input_length: Required length of input sequences
            output_length: Required length of output sequences

        Returns:
            Tensor of shape (n_valid_sequences, 3) with [group_idx, start, end] for each sequence

        Note:
            Implementation to follow after confirming the interface.
        """
        # TODO: Implement sequence finding logic
        raise NotImplementedError("Sequence finding logic to be implemented")
