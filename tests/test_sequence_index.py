import pytest
import torch

from transfer_learning_publication.containers import TimeSeriesCollection
from transfer_learning_publication.containers.sequence_index import SequenceIndex


class TestSequenceIndex:
    """Test SequenceIndex class functionality."""

    @pytest.fixture
    def basic_sequences(self):
        """Create basic test sequences data."""
        # Create sequences: [group_idx, start_idx, end_idx]
        # Group 0: sequences at positions 0-5, 2-7, 5-10
        # Group 1: sequences at positions 0-5, 3-8
        sequences = torch.tensor(
            [
                [0, 0, 5],
                [0, 2, 7],
                [0, 5, 10],
                [1, 0, 5],
                [1, 3, 8],
            ],
            dtype=torch.long,
        )
        n_groups = 3  # 0, 1, 2 (group 2 has no sequences)
        input_length = 3
        output_length = 2
        return sequences, n_groups, input_length, output_length

    @pytest.fixture
    def empty_sequences(self):
        """Create empty sequences data."""
        sequences = torch.tensor([], dtype=torch.long).reshape(0, 3)
        n_groups = 5
        input_length = 3
        output_length = 2
        return sequences, n_groups, input_length, output_length

    @pytest.fixture
    def single_sequence(self):
        """Create single sequence data."""
        sequences = torch.tensor([[0, 10, 15]], dtype=torch.long)
        n_groups = 1
        input_length = 3
        output_length = 2
        return sequences, n_groups, input_length, output_length

    def test_basic_construction(self, basic_sequences):
        """Test basic construction with valid data."""
        sequences, n_groups, input_length, output_length = basic_sequences
        index = SequenceIndex(sequences, n_groups, input_length, output_length)

        assert len(index) == 5
        assert index.n_sequences == 5
        assert index.n_groups == 3
        assert index.input_length == 3
        assert index.output_length == 2
        assert index.total_length == 5

    def test_construction_without_validation(self, basic_sequences):
        """Test construction with validation disabled."""
        sequences, n_groups, input_length, output_length = basic_sequences
        index = SequenceIndex(sequences, n_groups, input_length, output_length, validate=False)

        assert len(index) == 5
        assert index.n_sequences == 5

    def test_empty_construction(self, empty_sequences):
        """Test construction with empty sequences."""
        sequences, n_groups, input_length, output_length = empty_sequences
        index = SequenceIndex(sequences, n_groups, input_length, output_length)

        assert len(index) == 0
        assert index.n_sequences == 0
        assert index.n_groups == 5

    def test_dtype_conversion(self, basic_sequences):
        """Test automatic conversion to LongTensor."""
        sequences, n_groups, input_length, output_length = basic_sequences
        # Convert to float tensor
        float_sequences = sequences.float()

        index = SequenceIndex(float_sequences, n_groups, input_length, output_length)
        assert index._sequences.dtype == torch.long

    def test_non_tensor_input_error(self, basic_sequences):
        """Test that non-tensor input raises TypeError."""
        _, n_groups, input_length, output_length = basic_sequences
        sequences_list = [[0, 0, 5], [0, 2, 7]]

        with pytest.raises(TypeError, match="Expected torch.Tensor"):
            SequenceIndex(sequences_list, n_groups, input_length, output_length)

    def test_get_sequence_info(self, basic_sequences):
        """Test retrieving individual sequence information."""
        sequences, n_groups, input_length, output_length = basic_sequences
        index = SequenceIndex(sequences, n_groups, input_length, output_length)

        # Test first sequence
        group_idx, start_idx, end_idx = index.get_sequence_info(0)
        assert group_idx == 0
        assert start_idx == 0
        assert end_idx == 5

        # Test last sequence
        group_idx, start_idx, end_idx = index.get_sequence_info(4)
        assert group_idx == 1
        assert start_idx == 3
        assert end_idx == 8

    def test_get_sequence_info_out_of_bounds(self, basic_sequences):
        """Test that out-of-bounds access raises IndexError."""
        sequences, n_groups, input_length, output_length = basic_sequences
        index = SequenceIndex(sequences, n_groups, input_length, output_length)

        with pytest.raises(IndexError, match="out of bounds"):
            index.get_sequence_info(-1)

        with pytest.raises(IndexError, match="out of bounds"):
            index.get_sequence_info(5)

    def test_get_sequence_info_batch(self, basic_sequences):
        """Test batch retrieval of sequence information."""
        sequences, n_groups, input_length, output_length = basic_sequences
        index = SequenceIndex(sequences, n_groups, input_length, output_length)

        # Test with list input
        indices = [0, 2, 4]
        batch = index.get_sequence_info_batch(indices)
        assert batch.shape == (3, 3)
        assert batch[0, 0].item() == 0  # group_idx of sequence 0
        assert batch[1, 0].item() == 0  # group_idx of sequence 2
        assert batch[2, 0].item() == 1  # group_idx of sequence 4

        # Test with tensor input
        indices_tensor = torch.tensor([1, 3], dtype=torch.long)
        batch = index.get_sequence_info_batch(indices_tensor)
        assert batch.shape == (2, 3)

    def test_get_sequence_info_batch_out_of_bounds(self, basic_sequences):
        """Test that batch retrieval with out-of-bounds indices raises IndexError."""
        sequences, n_groups, input_length, output_length = basic_sequences
        index = SequenceIndex(sequences, n_groups, input_length, output_length)

        with pytest.raises(IndexError, match="Index out of bounds"):
            index.get_sequence_info_batch([0, 10])

    def test_get_sequences_for_group(self, basic_sequences):
        """Test retrieving sequences for a specific group."""
        sequences, n_groups, input_length, output_length = basic_sequences
        index = SequenceIndex(sequences, n_groups, input_length, output_length)

        # Group 0 should have 3 sequences
        group0_indices = index.get_sequences_for_group(0)
        assert len(group0_indices) == 3
        assert group0_indices.tolist() == [0, 1, 2]

        # Group 1 should have 2 sequences
        group1_indices = index.get_sequences_for_group(1)
        assert len(group1_indices) == 2
        assert group1_indices.tolist() == [3, 4]

        # Group 2 should have no sequences
        group2_indices = index.get_sequences_for_group(2)
        assert len(group2_indices) == 0

    def test_get_sequences_for_group_out_of_bounds(self, basic_sequences):
        """Test that invalid group index raises ValueError."""
        sequences, n_groups, input_length, output_length = basic_sequences
        index = SequenceIndex(sequences, n_groups, input_length, output_length)

        with pytest.raises(ValueError, match="Group index"):
            index.get_sequences_for_group(-1)

        with pytest.raises(ValueError, match="Group index"):
            index.get_sequences_for_group(3)

    def test_validation_invalid_shape(self):
        """Test validation catches invalid tensor shape."""
        sequences = torch.tensor([1, 2, 3], dtype=torch.long)  # 1D tensor
        with pytest.raises(ValueError, match="Expected sequences tensor of shape"):
            SequenceIndex(sequences, n_groups=1, input_length=3, output_length=2)

        sequences = torch.tensor([[1, 2]], dtype=torch.long)  # Wrong column count
        with pytest.raises(ValueError, match="Expected sequences tensor of shape"):
            SequenceIndex(sequences, n_groups=1, input_length=3, output_length=2)

    def test_validation_group_indices_out_of_range(self):
        """Test validation catches group indices out of valid range."""
        # Group index 5 when n_groups=3
        sequences = torch.tensor([[5, 0, 5]], dtype=torch.long)
        with pytest.raises(ValueError, match="Group indices out of range"):
            SequenceIndex(sequences, n_groups=3, input_length=3, output_length=2)

        # Negative group index
        sequences = torch.tensor([[-1, 0, 5]], dtype=torch.long)
        with pytest.raises(ValueError, match="Group indices out of range"):
            SequenceIndex(sequences, n_groups=3, input_length=3, output_length=2)

    def test_validation_negative_start_indices(self):
        """Test validation catches negative start indices."""
        sequences = torch.tensor([[0, -1, 5]], dtype=torch.long)
        with pytest.raises(ValueError, match="negative start index"):
            SequenceIndex(sequences, n_groups=1, input_length=3, output_length=2)

    def test_validation_invalid_sequence_lengths(self):
        """Test validation catches invalid sequence lengths."""
        # End <= start (non-positive length)
        sequences = torch.tensor([[0, 5, 5]], dtype=torch.long)
        with pytest.raises(ValueError, match="non-positive length"):
            SequenceIndex(sequences, n_groups=1, input_length=3, output_length=2)

        sequences = torch.tensor([[0, 5, 3]], dtype=torch.long)
        with pytest.raises(ValueError, match="non-positive length"):
            SequenceIndex(sequences, n_groups=1, input_length=3, output_length=2)

    def test_validation_inconsistent_sequence_lengths(self):
        """Test validation catches inconsistent sequence lengths."""
        # One sequence has length 5, another has length 6
        sequences = torch.tensor([[0, 0, 5], [0, 0, 6]], dtype=torch.long)
        with pytest.raises(ValueError, match="Sequence lengths inconsistent"):
            SequenceIndex(sequences, n_groups=1, input_length=3, output_length=2)

    def test_summary(self, basic_sequences):
        """Test summary statistics."""
        sequences, n_groups, input_length, output_length = basic_sequences
        index = SequenceIndex(sequences, n_groups, input_length, output_length)

        summary = index.summary()
        assert summary["n_sequences"] == 5
        assert summary["n_groups_with_sequences"] == 2
        assert summary["sequences_per_group"] == {0: 3, 1: 2}
        assert summary["avg_sequences_per_group"] == 2.5
        assert summary["min_sequences_per_group"] == 2
        assert summary["max_sequences_per_group"] == 3
        assert summary["memory_mb"] > 0

    def test_summary_empty(self, empty_sequences):
        """Test summary with empty index."""
        sequences, n_groups, input_length, output_length = empty_sequences
        index = SequenceIndex(sequences, n_groups, input_length, output_length)

        summary = index.summary()
        assert summary["n_sequences"] == 0
        assert summary["n_groups_with_sequences"] == 0
        assert summary["sequences_per_group"] == {}
        assert summary["avg_sequences_per_group"] == 0.0
        assert summary["memory_mb"] == 0.0

    def test_repr(self, basic_sequences):
        """Test string representation."""
        sequences, n_groups, input_length, output_length = basic_sequences
        index = SequenceIndex(sequences, n_groups, input_length, output_length)

        repr_str = repr(index)
        assert "SequenceIndex" in repr_str
        assert "n_sequences=5" in repr_str
        assert "n_groups_with_sequences=2" in repr_str
        assert "total_length=5" in repr_str

    def test_find_valid_sequences_not_implemented(self):
        """Test that find_valid_sequences raises NotImplementedError."""
        # This is a placeholder test for the unimplemented method
        with pytest.raises(NotImplementedError, match="Sequence finding logic"):
            SequenceIndex.find_valid_sequences(
                time_series=None,
                target_feature="target",
                forcing_features=["forcing1"],
                input_length=3,
                output_length=2,
            )


class TestSequenceIndexIntegration:
    """Integration tests with TimeSeriesCollection."""

    @pytest.fixture
    def time_series_with_nans(self):
        """Create TimeSeriesCollection with some NaN values."""
        # Group 0: 10 timesteps, NaNs at indices 3-4
        tensor0 = torch.tensor(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0],
                [float("nan"), 40.0, 400.0],
                [float("nan"), 50.0, 500.0],
                [6.0, 60.0, 600.0],
                [7.0, 70.0, 700.0],
                [8.0, 80.0, 800.0],
                [9.0, 90.0, 900.0],
                [10.0, 100.0, 1000.0],
            ],
            dtype=torch.float32,
        )

        # Group 1: 8 timesteps, NaN in forcing feature at index 5
        tensor1 = torch.tensor(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0],
                [4.0, 40.0, 400.0],
                [5.0, 50.0, 500.0],
                [6.0, float("nan"), 600.0],
                [7.0, 70.0, 700.0],
                [8.0, 80.0, 800.0],
            ],
            dtype=torch.float32,
        )

        feature_names = ["target", "forcing1", "forcing2"]
        from datetime import datetime

        date_ranges = [
            (datetime(2020, 1, 1), datetime(2020, 1, 10)),
            (datetime(2020, 1, 1), datetime(2020, 1, 8)),
        ]
        group_identifiers = ["group0", "group1"]

        # Don't validate to allow NaNs
        return TimeSeriesCollection(
            tensors=[tensor0, tensor1],
            feature_names=feature_names,
            date_ranges=date_ranges,
            group_identifiers=group_identifiers,
            validate=False,
        )

    def test_sequence_index_with_find_valid_sequences_placeholder(self, time_series_with_nans):
        """Test SequenceIndex with manually created sequences (until find_valid_sequences is implemented)."""
        # Manually create valid sequences for testing
        # For input_length=3, output_length=2 (total=5):
        # Group 0: Valid sequences at 0-5 (before NaN), 5-10 (after NaN)
        # Group 1: Valid sequences at 0-5 (before NaN in forcing)
        sequences = torch.tensor(
            [
                [0, 0, 5],  # Group 0, indices 0-4
                [0, 5, 10],  # Group 0, indices 5-9
                [1, 0, 5],  # Group 1, indices 0-4
            ],
            dtype=torch.long,
        )

        index = SequenceIndex(
            sequences=sequences,
            n_groups=2,
            input_length=3,
            output_length=2,
        )

        assert index.n_sequences == 3
        assert index.get_sequences_for_group(0).tolist() == [0, 1]
        assert index.get_sequences_for_group(1).tolist() == [2]

        # Verify sequence access
        group_idx, start_idx, end_idx = index.get_sequence_info(0)
        assert group_idx == 0
        assert start_idx == 0
        assert end_idx == 5
