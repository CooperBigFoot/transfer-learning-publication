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

    def test_find_valid_sequences_basic(self):
        """Test find_valid_sequences with basic time series."""
        # Create simple time series without NaNs
        tensor0 = torch.randn(10, 3, dtype=torch.float32)  # 10 timesteps
        tensor1 = torch.randn(8, 3, dtype=torch.float32)  # 8 timesteps
        tensor2 = torch.randn(3, 3, dtype=torch.float32)  # 3 timesteps (too short)

        from datetime import datetime

        time_series = TimeSeriesCollection(
            tensors=[tensor0, tensor1, tensor2],
            feature_names=["feature1", "feature2", "feature3"],
            date_ranges=[
                (datetime(2020, 1, 1), datetime(2020, 1, 10)),
                (datetime(2020, 1, 1), datetime(2020, 1, 8)),
                (datetime(2020, 1, 1), datetime(2020, 1, 3)),
            ],
            group_identifiers=["group0", "group1", "group2"],
        )

        sequences = SequenceIndex.find_valid_sequences(
            time_series=time_series,
            input_length=3,
            output_length=2,
        )

        # Group 0: length 10, total_length 5 -> 6 sequences (0-5, 1-6, ..., 5-10)
        # Group 1: length 8, total_length 5 -> 4 sequences (0-5, 1-6, 2-7, 3-8)
        # Group 2: length 3, total_length 5 -> 0 sequences (too short)
        assert sequences.shape == (10, 3)

        # Check first group sequences
        group0_sequences = sequences[sequences[:, 0] == 0]
        assert len(group0_sequences) == 6
        assert group0_sequences[0].tolist() == [0, 0, 5]
        assert group0_sequences[-1].tolist() == [0, 5, 10]

        # Check second group sequences
        group1_sequences = sequences[sequences[:, 0] == 1]
        assert len(group1_sequences) == 4
        assert group1_sequences[0].tolist() == [1, 0, 5]
        assert group1_sequences[-1].tolist() == [1, 3, 8]

        # Check third group (should have no sequences)
        group2_sequences = sequences[sequences[:, 0] == 2]
        assert len(group2_sequences) == 0

    def test_find_valid_sequences_with_target_filled_no_filled(self):
        """Test find_valid_sequences with target_was_filled where no values are filled."""
        # Create time series
        tensor0 = torch.randn(10, 2, dtype=torch.float32)
        tensor1 = torch.randn(8, 2, dtype=torch.float32)
        
        from datetime import datetime
        
        time_series = TimeSeriesCollection(
            tensors=[tensor0, tensor1],
            feature_names=["feature1", "feature2"],
            date_ranges=[
                (datetime(2020, 1, 1), datetime(2020, 1, 10)),
                (datetime(2020, 1, 1), datetime(2020, 1, 8)),
            ],
            group_identifiers=["group0", "group1"],
        )
        
        # Create flag tensors with all zeros (no filled values)
        target_filled = [
            torch.zeros(10, dtype=torch.uint8),
            torch.zeros(8, dtype=torch.uint8),
        ]
        
        sequences = SequenceIndex.find_valid_sequences(
            time_series=time_series,
            input_length=3,
            output_length=2,
            target_was_filled=target_filled,
        )
        
        # Should get the same sequences as without flags
        # Group 0: length 10, total 5 -> 6 sequences
        # Group 1: length 8, total 5 -> 4 sequences
        assert sequences.shape[0] == 10
        
    def test_find_valid_sequences_with_target_filled_all_filled(self):
        """Test find_valid_sequences with target_was_filled where all values are filled."""
        # Create time series
        tensor0 = torch.randn(10, 2, dtype=torch.float32)
        tensor1 = torch.randn(8, 2, dtype=torch.float32)
        
        from datetime import datetime
        
        time_series = TimeSeriesCollection(
            tensors=[tensor0, tensor1],
            feature_names=["feature1", "feature2"],
            date_ranges=[
                (datetime(2020, 1, 1), datetime(2020, 1, 10)),
                (datetime(2020, 1, 1), datetime(2020, 1, 8)),
            ],
            group_identifiers=["group0", "group1"],
        )
        
        # Create flag tensors with all ones (all filled values)
        target_filled = [
            torch.ones(10, dtype=torch.uint8),
            torch.ones(8, dtype=torch.uint8),
        ]
        
        sequences = SequenceIndex.find_valid_sequences(
            time_series=time_series,
            input_length=3,
            output_length=2,
            target_was_filled=target_filled,
        )
        
        # Should get no sequences since all target values are filled
        assert sequences.shape[0] == 0
        
    def test_find_valid_sequences_with_target_filled_mixed(self):
        """Test find_valid_sequences with mixed filled/original values."""
        # Create time series
        tensor0 = torch.randn(10, 2, dtype=torch.float32)
        
        from datetime import datetime
        
        time_series = TimeSeriesCollection(
            tensors=[tensor0],
            feature_names=["feature1", "feature2"],
            date_ranges=[(datetime(2020, 1, 1), datetime(2020, 1, 10))],
            group_identifiers=["group0"],
        )
        
        # Create flag with pattern: [0,0,0,0,0,1,1,0,0,0]
        # This means positions 5 and 6 have filled values
        target_filled = [
            torch.tensor([0, 0, 0, 0, 0, 1, 1, 0, 0, 0], dtype=torch.uint8),
        ]
        
        sequences = SequenceIndex.find_valid_sequences(
            time_series=time_series,
            input_length=3,
            output_length=2,
            target_was_filled=target_filled,
        )
        
        # With input_length=3, output_length=2, total_length=5
        # Possible windows: [0:5], [1:6], [2:7], [3:8], [4:9], [5:10]
        # Output windows:   [3:5], [4:6], [5:7], [6:8], [7:9], [8:10]
        # 
        # [0:5] -> output [3:5] = positions 3,4 -> flags [0,0] -> VALID
        # [1:6] -> output [4:6] = positions 4,5 -> flags [0,1] -> INVALID (has filled)
        # [2:7] -> output [5:7] = positions 5,6 -> flags [1,1] -> INVALID (has filled)
        # [3:8] -> output [6:8] = positions 6,7 -> flags [1,0] -> INVALID (has filled)
        # [4:9] -> output [7:9] = positions 7,8 -> flags [0,0] -> VALID
        # [5:10] -> output [8:10] = positions 8,9 -> flags [0,0] -> VALID
        
        assert sequences.shape[0] == 3  # Only 3 valid sequences
        
        # Check the valid sequences
        valid_starts = sequences[:, 1].tolist()
        assert 0 in valid_starts  # [0:5] is valid
        assert 4 in valid_starts  # [4:9] is valid
        assert 5 in valid_starts  # [5:10] is valid
        
    def test_find_valid_sequences_with_target_filled_output_window_only(self):
        """Test that filled values in input window don't affect sequence validity."""
        # Create time series
        tensor0 = torch.randn(10, 2, dtype=torch.float32)
        
        from datetime import datetime
        
        time_series = TimeSeriesCollection(
            tensors=[tensor0],
            feature_names=["feature1", "feature2"],
            date_ranges=[(datetime(2020, 1, 1), datetime(2020, 1, 10))],
            group_identifiers=["group0"],
        )
        
        # Create flag with filled values only in positions that would be in input windows
        # Pattern: [1,1,1,0,0,0,0,0,0,0]
        # Filled values at positions 0,1,2 (will be in input windows only)
        target_filled = [
            torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint8),
        ]
        
        sequences = SequenceIndex.find_valid_sequences(
            time_series=time_series,
            input_length=3,
            output_length=2,
            target_was_filled=target_filled,
        )
        
        # With input_length=3, output_length=2:
        # [0:5] -> input [0:3] = [1,1,1], output [3:5] = [0,0] -> VALID (output has no filled)
        # [1:6] -> input [1:4] = [1,1,0], output [4:6] = [0,0] -> VALID
        # [2:7] -> input [2:5] = [1,0,0], output [5:7] = [0,0] -> VALID
        # [3:8] -> input [3:6] = [0,0,0], output [6:8] = [0,0] -> VALID
        # [4:9] -> input [4:7] = [0,0,0], output [7:9] = [0,0] -> VALID
        # [5:10] -> input [5:8] = [0,0,0], output [8:10] = [0,0] -> VALID
        
        # All sequences should be valid since filled values are only in input windows
        assert sequences.shape[0] == 6
        
    def test_find_valid_sequences_backward_compatibility(self):
        """Test that find_valid_sequences works without target_was_filled (backward compatibility)."""
        # Create time series
        tensor0 = torch.randn(10, 2, dtype=torch.float32)
        
        from datetime import datetime
        
        time_series = TimeSeriesCollection(
            tensors=[tensor0],
            feature_names=["feature1", "feature2"],
            date_ranges=[(datetime(2020, 1, 1), datetime(2020, 1, 10))],
            group_identifiers=["group0"],
        )
        
        # Call without target_was_filled
        sequences = SequenceIndex.find_valid_sequences(
            time_series=time_series,
            input_length=3,
            output_length=2,
        )
        
        # Should get all possible sequences
        assert sequences.shape[0] == 6  # 10 - 5 + 1 = 6 sequences

    def test_find_valid_sequences_empty(self):
        """Test find_valid_sequences with empty time series."""

        time_series = TimeSeriesCollection(
            tensors=[],
            feature_names=["feature1"],
            date_ranges=[],
            group_identifiers=[],
        )

        sequences = SequenceIndex.find_valid_sequences(
            time_series=time_series,
            input_length=3,
            output_length=2,
        )

        assert sequences.shape == (0, 3)
        assert sequences.dtype == torch.long

    def test_find_valid_sequences_exact_length(self):
        """Test find_valid_sequences when group length exactly matches total_length."""
        # Create time series with exact length match
        tensor = torch.randn(5, 2, dtype=torch.float32)  # Exactly 5 timesteps

        from datetime import datetime

        time_series = TimeSeriesCollection(
            tensors=[tensor],
            feature_names=["feature1", "feature2"],
            date_ranges=[(datetime(2020, 1, 1), datetime(2020, 1, 5))],
            group_identifiers=["group0"],
        )

        sequences = SequenceIndex.find_valid_sequences(
            time_series=time_series,
            input_length=3,
            output_length=2,
        )

        # Should have exactly one sequence
        assert sequences.shape == (1, 3)
        assert sequences[0].tolist() == [0, 0, 5]

    def test_find_valid_sequences_all_too_short(self):
        """Test find_valid_sequences when all groups are too short."""
        tensor0 = torch.randn(2, 3, dtype=torch.float32)  # Too short
        tensor1 = torch.randn(3, 3, dtype=torch.float32)  # Too short

        from datetime import datetime

        time_series = TimeSeriesCollection(
            tensors=[tensor0, tensor1],
            feature_names=["feature1", "feature2", "feature3"],
            date_ranges=[
                (datetime(2020, 1, 1), datetime(2020, 1, 2)),
                (datetime(2020, 1, 1), datetime(2020, 1, 3)),
            ],
            group_identifiers=["group0", "group1"],
        )

        sequences = SequenceIndex.find_valid_sequences(
            time_series=time_series,
            input_length=3,
            output_length=2,
        )

        # Should return empty tensor with correct shape
        assert sequences.shape == (0, 3)
        assert sequences.dtype == torch.long


class TestSequenceIndexIntegration:
    """Integration tests with TimeSeriesCollection."""

    @pytest.fixture
    def time_series_no_nans(self):
        """Create TimeSeriesCollection without NaN values."""
        # Group 0: 10 timesteps
        tensor0 = torch.tensor(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0],
                [4.0, 40.0, 400.0],
                [5.0, 50.0, 500.0],
                [6.0, 60.0, 600.0],
                [7.0, 70.0, 700.0],
                [8.0, 80.0, 800.0],
                [9.0, 90.0, 900.0],
                [10.0, 100.0, 1000.0],
            ],
            dtype=torch.float32,
        )

        # Group 1: 8 timesteps
        tensor1 = torch.tensor(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0],
                [4.0, 40.0, 400.0],
                [5.0, 50.0, 500.0],
                [6.0, 60.0, 600.0],
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

        return TimeSeriesCollection(
            tensors=[tensor0, tensor1],
            feature_names=feature_names,
            date_ranges=date_ranges,
            group_identifiers=group_identifiers,
        )

    def test_sequence_index_with_find_valid_sequences(self, time_series_no_nans):
        """Test SequenceIndex with find_valid_sequences method."""
        # Use find_valid_sequences to generate sequences
        sequences = SequenceIndex.find_valid_sequences(
            time_series=time_series_no_nans,
            input_length=3,
            output_length=2,
        )

        # Create index from generated sequences
        index = SequenceIndex(
            sequences=sequences,
            n_groups=2,
            input_length=3,
            output_length=2,
        )

        # Group 0: 10 timesteps, total_length=5 -> 6 valid sequences
        # Group 1: 8 timesteps, total_length=5 -> 4 valid sequences
        assert index.n_sequences == 10
        assert len(index.get_sequences_for_group(0)) == 6
        assert len(index.get_sequences_for_group(1)) == 4

        # Verify first sequence of group 0
        group_idx, start_idx, end_idx = index.get_sequence_info(0)
        assert group_idx == 0
        assert start_idx == 0
        assert end_idx == 5

        # Verify last sequence of group 0
        group_idx, start_idx, end_idx = index.get_sequence_info(5)
        assert group_idx == 0
        assert start_idx == 5
        assert end_idx == 10

        # Verify first sequence of group 1
        group_idx, start_idx, end_idx = index.get_sequence_info(6)
        assert group_idx == 1
        assert start_idx == 0
        assert end_idx == 5

        # Verify last sequence of group 1
        group_idx, start_idx, end_idx = index.get_sequence_info(9)
        assert group_idx == 1
        assert start_idx == 3
        assert end_idx == 8

    def test_find_valid_sequences_with_mixed_lengths(self):
        """Test find_valid_sequences with groups of varying lengths."""
        # Create time series with very different lengths
        tensor0 = torch.randn(20, 2, dtype=torch.float32)  # Long series
        tensor1 = torch.randn(5, 2, dtype=torch.float32)  # Exactly minimum length
        tensor2 = torch.randn(4, 2, dtype=torch.float32)  # Too short
        tensor3 = torch.randn(7, 2, dtype=torch.float32)  # Slightly longer

        from datetime import datetime

        time_series = TimeSeriesCollection(
            tensors=[tensor0, tensor1, tensor2, tensor3],
            feature_names=["feature1", "feature2"],
            date_ranges=[
                (datetime(2020, 1, 1), datetime(2020, 1, 20)),
                (datetime(2020, 1, 1), datetime(2020, 1, 5)),
                (datetime(2020, 1, 1), datetime(2020, 1, 4)),
                (datetime(2020, 1, 1), datetime(2020, 1, 7)),
            ],
            group_identifiers=["group0", "group1", "group2", "group3"],
        )

        sequences = SequenceIndex.find_valid_sequences(
            time_series=time_series,
            input_length=3,
            output_length=2,
        )

        # Group 0: 20 timesteps -> 16 sequences
        # Group 1: 5 timesteps -> 1 sequence
        # Group 2: 4 timesteps -> 0 sequences
        # Group 3: 7 timesteps -> 3 sequences
        # Total: 20 sequences
        assert sequences.shape[0] == 20

        # Verify sequences are ordered by group
        group_indices = sequences[:, 0].tolist()
        assert group_indices == [0] * 16 + [1] * 1 + [3] * 3

        # Create index and verify
        index = SequenceIndex(
            sequences=sequences,
            n_groups=4,
            input_length=3,
            output_length=2,
        )

        assert len(index.get_sequences_for_group(0)) == 16
        assert len(index.get_sequences_for_group(1)) == 1
        assert len(index.get_sequences_for_group(2)) == 0
        assert len(index.get_sequences_for_group(3)) == 3
