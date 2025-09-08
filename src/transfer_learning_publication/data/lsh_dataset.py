import logging
from datetime import datetime

import torch
from torch.utils.data import Dataset

from ..containers.lsh_data_container import LSHDataContainer
from ..contracts.sample import Sample

logger = logging.getLogger(__name__)


class LSHDataset(Dataset):
    """
    High-performance dataset for time series forecasting.

    This dataset is a pure data access layer that extracts samples from
    a pre-validated LSHDataContainer. It performs no validation or data
    processing - just fast, direct tensor access.

    Design Principles:
    - Trust upstream validation (no defensive programming)
    - Integer-based indexing for performance
    - Direct tensor access in the hot path
    - String lookups only for final Sample creation

    Args:
        container: Pre-validated data container with all collections and config
    """

    def __init__(self, container: LSHDataContainer):
        """Initialize dataset with pre-validated container."""
        super().__init__()
        self.container = container

        # Cache frequently accessed attributes for performance
        self._n_sequences = len(container.sequence_index)
        self._config = container.config
        self._sequence_index = container.sequence_index

        # Cache direct tensor access for hot path
        self._time_tensors = container.time_series._tensors
        self._static_tensors = container.static_attributes._tensors

        # Cache group identifiers list for string lookup
        self._group_ids = container.time_series.group_identifiers

        # Pre-compute frequently used values
        self._input_length = self._config.input_length
        self._output_length = self._config.output_length

        # Log dataset summary
        logger.info(
            f"LSHDataset initialized: {self._n_sequences:,} sequences, "
            f"input_length={self._input_length}, output_length={self._output_length}"
        )

    def __len__(self) -> int:
        """Return number of valid sequences in the dataset."""
        return self._n_sequences

    def __getitem__(self, idx: int) -> Sample:
        """
        Get a sample by index.

        This is the hot path - optimized for performance with direct
        tensor access and minimal overhead.

        Args:
            idx: Sample index in range [0, n_sequences)

        Returns:
            Sample object with input, output, static, and future tensors

        Raises:
            IndexError: If idx is out of bounds (propagated from sequence_index)
        """
        # Step 1: Get sequence location (single integer lookup)
        group_idx, start, end = self._sequence_index.get_sequence_info(idx)
        input_end = start + self._input_length

        # Step 2: Direct tensor access (fast path)
        time_tensor = self._time_tensors[group_idx]
        static_tensor = self._static_tensors[group_idx]

        # Step 3: Extract features using advanced indexing
        if self._config.input_feature_indices:
            X = time_tensor[start:input_end, self._config.input_feature_indices]
        else:
            X = time_tensor[start:input_end]

        y = time_tensor[input_end:end, self._config.target_idx]

        if self._config.future_indices:
            future = time_tensor[input_end:end, self._config.future_indices]
        else:
            future = torch.empty((self._output_length, 0), dtype=time_tensor.dtype)

        # Step 4: String lookup
        group_id = self._group_ids[group_idx]

        # Step 5: Optional date tracking
        input_end_date = None
        if self._config.include_dates:
            try:
                end_date = self.container.time_series.index_to_date(group_id, input_end - 1)
                if isinstance(end_date, datetime):
                    input_end_date = int(end_date.timestamp() * 1000)
                else:
                    input_end_date = int(datetime.combine(end_date, datetime.min.time()).timestamp() * 1000)
            except Exception as e:
                logger.warning(f"Failed to compute input_end_date for group {group_id}, sequence {idx}: {e}")
                input_end_date = None

        # Step 6: Create and return Sample
        return Sample(
            X=X, y=y, static=static_tensor, future=future, group_identifier=group_id, input_end_date=input_end_date
        )
