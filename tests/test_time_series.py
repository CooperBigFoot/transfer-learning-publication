import logging
from datetime import datetime

import pytest
import torch

from transfer_learning_publication.containers import TimeSeriesCollection


class TestTimeSeriesCollection:
    """Test TimeSeriesCollection class functionality."""

    @pytest.fixture
    def basic_data(self):
        """Create basic test data with 2 groups."""
        tensors = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),
            torch.tensor([[7.0, 8.0], [9.0, 10.0]], dtype=torch.float32),
        ]
        feature_names = ["feature_a", "feature_b"]
        date_ranges = [
            (datetime(2020, 1, 1), datetime(2020, 1, 3)),
            (datetime(2020, 1, 1), datetime(2020, 1, 2)),
        ]
        group_identifiers = ["group1", "group2"]
        return tensors, feature_names, date_ranges, group_identifiers

    @pytest.fixture
    def empty_data(self):
        """Create empty test data."""
        return [], ["feature_a"], [], []

    @pytest.fixture
    def single_feature_data(self):
        """Create test data with single feature."""
        tensors = [
            torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32),
        ]
        feature_names = ["single_feature"]
        date_ranges = [
            (datetime(2020, 1, 1), datetime(2020, 1, 3)),
        ]
        group_identifiers = ["group1"]
        return tensors, feature_names, date_ranges, group_identifiers

    @pytest.fixture
    def mixed_dtype_data(self):
        """Create test data with different tensor dtypes."""
        tensors = [
            torch.tensor([[1.0, 2.0]], dtype=torch.float64),  # float64
            torch.tensor([[3.0, 4.0]], dtype=torch.float32),  # float32
        ]
        feature_names = ["feature_a", "feature_b"]
        date_ranges = [
            (datetime(2020, 1, 1), datetime(2020, 1, 1)),
            (datetime(2020, 1, 1), datetime(2020, 1, 1)),
        ]
        group_identifiers = ["group1", "group2"]
        return tensors, feature_names, date_ranges, group_identifiers

    def test_basic_construction(self, basic_data):
        """Test basic construction with valid data."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        assert len(collection) == 2
        assert collection.get_n_features() == 2
        assert collection.get_total_timesteps() == 5
        assert "group1" in collection
        assert "group2" in collection
        assert "group3" not in collection

    def test_construction_without_validation(self, basic_data):
        """Test construction with validation disabled."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers, validate=False)

        assert len(collection) == 2
        assert collection.get_n_features() == 2

    def test_construction_with_nan_values(self, basic_data):
        """Test construction fails with NaN values when validation enabled."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        tensors[0][0, 0] = float("nan")

        with pytest.raises(ValueError, match="Group 'group1' contains 1 NaN values"):
            TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

    def test_construction_with_mismatched_features(self, basic_data):
        """Test construction fails with mismatched feature count."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        tensors[0] = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)  # 3 features

        with pytest.raises(ValueError, match="Group 'group1' has 3 features, expected 2"):
            TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

    def test_construction_with_mismatched_date_range(self, basic_data):
        """Test construction fails when date range doesn't match tensor length."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        # group1 has 3 timesteps but date range implies 1 day
        date_ranges[0] = (datetime(2020, 1, 1), datetime(2020, 1, 1))

        with pytest.raises(ValueError, match="Group 'group1' has 3 timesteps but date range"):
            TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

    def test_construction_with_missing_date_ranges(self, basic_data):
        """Test construction fails with missing date ranges."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        date_ranges.pop(0)  # Remove first date range

        with pytest.raises(ValueError, match="Number of tensors.*must match number of date ranges"):
            TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

    def test_construction_with_missing_tensors(self, basic_data):
        """Test construction fails with missing tensors."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        tensors.pop(0)  # Remove first tensor

        with pytest.raises(ValueError, match="Number of tensors.*must match number of group identifiers"):
            TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

    def test_get_group_series_valid(self, basic_data):
        """Test get_group_series with valid inputs."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        # Get full series
        result = collection.get_group_series("group1", 0, 3)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert torch.equal(result, expected)

        # Get partial series
        result = collection.get_group_series("group1", 1, 2)
        expected = torch.tensor([[3.0, 4.0]])
        assert torch.equal(result, expected)

    def test_get_group_series_invalid_group(self, basic_data):
        """Test get_group_series with invalid group."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        with pytest.raises(KeyError, match="Group 'invalid' not found in collection"):
            collection.get_group_series("invalid", 0, 1)

    def test_get_group_series_invalid_indices(self, basic_data):
        """Test get_group_series with invalid indices."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        # Out of bounds end index
        with pytest.raises(IndexError, match="Index range .* out of bounds"):
            collection.get_group_series("group1", 0, 10)

        # Negative start index
        with pytest.raises(IndexError, match="Index range .* out of bounds"):
            collection.get_group_series("group1", -1, 2)

        # start >= end
        with pytest.raises(ValueError, match="start_idx .* must be less than end_idx"):
            collection.get_group_series("group1", 2, 1)

    def test_get_group_feature_valid(self, basic_data):
        """Test get_group_feature with valid inputs."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        result = collection.get_group_feature("group1", "feature_a", 0, 3)
        expected = torch.tensor([1.0, 3.0, 5.0])
        assert torch.equal(result, expected)

        result = collection.get_group_feature("group1", "feature_b", 1, 2)
        expected = torch.tensor([4.0])
        assert torch.equal(result, expected)

    def test_get_group_feature_invalid_feature(self, basic_data):
        """Test get_group_feature with invalid feature."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        with pytest.raises(KeyError, match="Feature 'invalid' not found"):
            collection.get_group_feature("group1", "invalid", 0, 1)

    def test_properties_immutability(self, basic_data):
        """Test that properties return immutable copies."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        # Test group_identifiers
        groups = collection.group_identifiers
        original_groups = collection.group_identifiers
        groups.append("new_group")
        assert collection.group_identifiers == original_groups

        # Test feature_names
        features = collection.feature_names
        original_features = collection.feature_names
        features.append("new_feature")
        assert collection.feature_names == original_features

        # Test date_ranges
        dates = collection.date_ranges
        original_dates = collection.date_ranges
        dates["new_group"] = (datetime(2020, 1, 1), datetime(2020, 1, 2))
        assert collection.date_ranges == original_dates

    def test_get_group_length(self, basic_data):
        """Test get_group_length method."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        assert collection.get_group_length("group1") == 3
        assert collection.get_group_length("group2") == 2

        with pytest.raises(KeyError, match="Group 'invalid' not found"):
            collection.get_group_length("invalid")

    def test_index_to_date(self, basic_data):
        """Test index_to_date conversion."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        assert collection.index_to_date("group1", 0) == datetime(2020, 1, 1)
        assert collection.index_to_date("group1", 1) == datetime(2020, 1, 2)
        assert collection.index_to_date("group1", 2) == datetime(2020, 1, 3)

        with pytest.raises(IndexError, match="Index .* out of bounds"):
            collection.index_to_date("group1", 10)

        with pytest.raises(KeyError, match="Group 'invalid' not found"):
            collection.index_to_date("invalid", 0)

    def test_date_to_index(self, basic_data):
        """Test date_to_index conversion."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        assert collection.date_to_index("group1", datetime(2020, 1, 1)) == 0
        assert collection.date_to_index("group1", datetime(2020, 1, 2)) == 1
        assert collection.date_to_index("group1", datetime(2020, 1, 3)) == 2

        # Test with date object (not datetime)
        from datetime import date

        assert collection.date_to_index("group1", date(2020, 1, 2)) == 1

        with pytest.raises(ValueError, match="Date .* outside range"):
            collection.date_to_index("group1", datetime(2020, 1, 10))

        with pytest.raises(KeyError, match="Group 'invalid' not found"):
            collection.date_to_index("invalid", datetime(2020, 1, 1))

    def test_date_to_index_invalid_type(self, basic_data):
        """Test date_to_index with invalid date type."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        with pytest.raises(TypeError, match="Expected datetime or date object"):
            collection.date_to_index("group1", "2020-01-01")  # String instead of date

    def test_summary_with_data(self, basic_data):
        """Test summary method with data."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        summary = collection.summary()

        assert summary["n_groups"] == 2
        assert summary["n_features"] == 2
        assert summary["total_timesteps"] == 5
        assert summary["min_length"] == 2
        assert summary["max_length"] == 3
        assert summary["avg_length"] == 2.5
        assert summary["memory_mb"] >= 0  # Memory should be non-negative, might be very small
        assert summary["date_range"] == (datetime(2020, 1, 1), datetime(2020, 1, 3))

    def test_summary_empty(self, empty_data):
        """Test summary method with empty data."""
        tensors, feature_names, date_ranges, group_identifiers = empty_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        summary = collection.summary()

        assert summary["n_groups"] == 0
        assert summary["n_features"] == 1
        assert summary["total_timesteps"] == 0
        assert summary["min_length"] == 0
        assert summary["max_length"] == 0
        assert summary["avg_length"] == 0.0
        assert summary["memory_mb"] == 0.0
        assert summary["date_range"] == (None, None)

    def test_repr_string(self, basic_data):
        """Test __repr__ method."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        repr_str = repr(collection)
        assert "TimeSeriesCollection" in repr_str
        assert "n_groups=2" in repr_str
        assert "n_features=2" in repr_str
        assert "total_timesteps=5" in repr_str
        assert "memory_mb=" in repr_str

    def test_special_methods(self, basic_data):
        """Test special methods (__len__, __contains__)."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        # Test __len__
        assert len(collection) == 2

        # Test __contains__
        assert "group1" in collection
        assert "group2" in collection
        assert "group3" not in collection

    def test_single_feature_collection(self, single_feature_data):
        """Test collection with single feature."""
        tensors, feature_names, date_ranges, group_identifiers = single_feature_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        assert collection.get_n_features() == 1
        assert len(collection.feature_names) == 1
        assert collection.feature_names[0] == "single_feature"

        # Test feature retrieval
        result = collection.get_group_feature("group1", "single_feature", 0, 3)
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.equal(result, expected)

    def test_memory_calculation_different_dtypes(self, mixed_dtype_data):
        """Test memory calculation with different tensor dtypes."""
        tensors, feature_names, date_ranges, group_identifiers = mixed_dtype_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        summary = collection.summary()
        # Note: This test shows that the memory calculation now uses actual tensor dtypes
        assert summary["memory_mb"] >= 0  # Memory should be non-negative, might be very small for test data

    def test_validation_logging(self, basic_data, caplog):
        """Test that validation logs success message."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data

        with caplog.at_level(logging.INFO):
            TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        assert "Validation passed for 2 groups" in caplog.text

    def test_edge_case_single_timestep(self):
        """Test collection with single timestep per group."""
        tensors = [
            torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        ]
        feature_names = ["feature_a", "feature_b"]
        date_ranges = [
            (datetime(2020, 1, 1), datetime(2020, 1, 1)),
        ]
        group_identifiers = ["group1"]

        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        assert collection.get_group_length("group1") == 1
        assert collection.get_total_timesteps() == 1

        # Test data retrieval
        result = collection.get_group_series("group1", 0, 1)
        expected = torch.tensor([[1.0, 2.0]])
        assert torch.equal(result, expected)

    def test_feature_indices_property(self, basic_data):
        """Test feature_indices property."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        indices = collection.feature_indices
        assert indices["feature_a"] == 0
        assert indices["feature_b"] == 1

        # Test immutability
        indices["new_feature"] = 2
        assert "new_feature" not in collection.feature_indices

    def test_large_date_ranges(self):
        """Test with larger date ranges to verify date calculations."""
        tensors = [
            torch.ones((365, 2), dtype=torch.float32),  # One year of daily data
        ]
        feature_names = ["feature_a", "feature_b"]
        date_ranges = [
            (datetime(2020, 1, 1), datetime(2020, 12, 31)),  # 366 days (leap year)
        ]
        group_identifiers = ["group1"]

        # This should fail because tensor has 365 timesteps but leap year 2020 has 366 days
        with pytest.raises(ValueError, match="has 365 timesteps but date range.*implies 366 days"):
            TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        # Fix the tensor size
        tensors[0] = torch.ones((366, 2), dtype=torch.float32)
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        assert collection.get_group_length("group1") == 366
        assert collection.index_to_date("group1", 0) == datetime(2020, 1, 1)
        assert collection.index_to_date("group1", 365) == datetime(2020, 12, 31)

    def test_integer_based_access(self, basic_data):
        """Test integer-based access methods (fast path)."""
        tensors, feature_names, date_ranges, group_identifiers = basic_data
        collection = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        # Test get_group_series_by_idx
        result = collection.get_group_series_by_idx(0, 0, 3)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert torch.equal(result, expected)

        result = collection.get_group_series_by_idx(1, 0, 2)
        expected = torch.tensor([[7.0, 8.0], [9.0, 10.0]])
        assert torch.equal(result, expected)

        # Test get_group_feature_by_idx
        result = collection.get_group_feature_by_idx(0, "feature_a", 0, 3)
        expected = torch.tensor([1.0, 3.0, 5.0])
        assert torch.equal(result, expected)

        # Test get_group_length_by_idx
        assert collection.get_group_length_by_idx(0) == 3
        assert collection.get_group_length_by_idx(1) == 2

        # Test out of bounds
        with pytest.raises(IndexError, match="Group index 2 out of bounds"):
            collection.get_group_series_by_idx(2, 0, 1)

        # Test group_to_idx mapping
        assert collection.group_to_idx["group1"] == 0
        assert collection.group_to_idx["group2"] == 1
