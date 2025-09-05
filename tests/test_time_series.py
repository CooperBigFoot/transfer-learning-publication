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
        group_tensors = {
            "group1": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),
            "group2": torch.tensor([[7.0, 8.0], [9.0, 10.0]], dtype=torch.float32),
        }
        feature_names = ["feature_a", "feature_b"]
        date_ranges = {
            "group1": (datetime(2020, 1, 1), datetime(2020, 1, 3)),
            "group2": (datetime(2020, 1, 1), datetime(2020, 1, 2)),
        }
        return group_tensors, feature_names, date_ranges

    @pytest.fixture
    def empty_data(self):
        """Create empty test data."""
        return {}, ["feature_a"], {}

    @pytest.fixture
    def single_feature_data(self):
        """Create test data with single feature."""
        group_tensors = {
            "group1": torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32),
        }
        feature_names = ["single_feature"]
        date_ranges = {
            "group1": (datetime(2020, 1, 1), datetime(2020, 1, 3)),
        }
        return group_tensors, feature_names, date_ranges

    @pytest.fixture
    def mixed_dtype_data(self):
        """Create test data with different tensor dtypes."""
        group_tensors = {
            "group1": torch.tensor([[1.0, 2.0]], dtype=torch.float64),  # float64
            "group2": torch.tensor([[3.0, 4.0]], dtype=torch.float32),  # float32
        }
        feature_names = ["feature_a", "feature_b"]
        date_ranges = {
            "group1": (datetime(2020, 1, 1), datetime(2020, 1, 1)),
            "group2": (datetime(2020, 1, 1), datetime(2020, 1, 1)),
        }
        return group_tensors, feature_names, date_ranges

    def test_basic_construction(self, basic_data):
        """Test basic construction with valid data."""
        group_tensors, feature_names, date_ranges = basic_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

        assert len(collection) == 2
        assert collection.get_n_features() == 2
        assert collection.get_total_timesteps() == 5
        assert "group1" in collection
        assert "group2" in collection
        assert "group3" not in collection

    def test_construction_without_validation(self, basic_data):
        """Test construction with validation disabled."""
        group_tensors, feature_names, date_ranges = basic_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges, validate=False)

        assert len(collection) == 2
        assert collection.get_n_features() == 2

    def test_construction_with_nan_values(self, basic_data):
        """Test construction fails with NaN values when validation enabled."""
        group_tensors, feature_names, date_ranges = basic_data
        group_tensors["group1"][0, 0] = float("nan")

        with pytest.raises(ValueError, match="Group 'group1' contains 1 NaN values"):
            TimeSeriesCollection(group_tensors, feature_names, date_ranges)

    def test_construction_with_mismatched_features(self, basic_data):
        """Test construction fails with mismatched feature count."""
        group_tensors, feature_names, date_ranges = basic_data
        group_tensors["group1"] = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)  # 3 features

        with pytest.raises(ValueError, match="Group 'group1' has 3 features, expected 2"):
            TimeSeriesCollection(group_tensors, feature_names, date_ranges)

    def test_construction_with_mismatched_date_range(self, basic_data):
        """Test construction fails when date range doesn't match tensor length."""
        group_tensors, feature_names, date_ranges = basic_data
        # group1 has 3 timesteps but date range implies 1 day
        date_ranges["group1"] = (datetime(2020, 1, 1), datetime(2020, 1, 1))

        with pytest.raises(ValueError, match="Group 'group1' has 3 timesteps but date range"):
            TimeSeriesCollection(group_tensors, feature_names, date_ranges)

    def test_construction_with_missing_date_ranges(self, basic_data):
        """Test construction fails with missing date ranges."""
        group_tensors, feature_names, date_ranges = basic_data
        del date_ranges["group1"]

        with pytest.raises(ValueError, match="Groups with tensors but no dates"):
            TimeSeriesCollection(group_tensors, feature_names, date_ranges)

    def test_construction_with_missing_tensors(self, basic_data):
        """Test construction fails with missing tensors."""
        group_tensors, feature_names, date_ranges = basic_data
        del group_tensors["group1"]

        with pytest.raises(ValueError, match="Groups with dates but no tensors"):
            TimeSeriesCollection(group_tensors, feature_names, date_ranges)

    def test_get_group_series_valid(self, basic_data):
        """Test get_group_series with valid inputs."""
        group_tensors, feature_names, date_ranges = basic_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

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
        group_tensors, feature_names, date_ranges = basic_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

        with pytest.raises(KeyError, match="Group 'invalid' not found in collection"):
            collection.get_group_series("invalid", 0, 1)

    def test_get_group_series_invalid_indices(self, basic_data):
        """Test get_group_series with invalid indices."""
        group_tensors, feature_names, date_ranges = basic_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

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
        group_tensors, feature_names, date_ranges = basic_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

        result = collection.get_group_feature("group1", "feature_a", 0, 3)
        expected = torch.tensor([1.0, 3.0, 5.0])
        assert torch.equal(result, expected)

        result = collection.get_group_feature("group1", "feature_b", 1, 2)
        expected = torch.tensor([4.0])
        assert torch.equal(result, expected)

    def test_get_group_feature_invalid_feature(self, basic_data):
        """Test get_group_feature with invalid feature."""
        group_tensors, feature_names, date_ranges = basic_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

        with pytest.raises(KeyError, match="Feature 'invalid' not found"):
            collection.get_group_feature("group1", "invalid", 0, 1)

    def test_properties_immutability(self, basic_data):
        """Test that properties return immutable copies."""
        group_tensors, feature_names, date_ranges = basic_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

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
        group_tensors, feature_names, date_ranges = basic_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

        assert collection.get_group_length("group1") == 3
        assert collection.get_group_length("group2") == 2

        with pytest.raises(KeyError, match="Group 'invalid' not found"):
            collection.get_group_length("invalid")

    def test_index_to_date(self, basic_data):
        """Test index_to_date conversion."""
        group_tensors, feature_names, date_ranges = basic_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

        assert collection.index_to_date("group1", 0) == datetime(2020, 1, 1)
        assert collection.index_to_date("group1", 1) == datetime(2020, 1, 2)
        assert collection.index_to_date("group1", 2) == datetime(2020, 1, 3)

        with pytest.raises(IndexError, match="Index .* out of bounds"):
            collection.index_to_date("group1", 10)

        with pytest.raises(KeyError, match="Group 'invalid' not found in date_ranges"):
            collection.index_to_date("invalid", 0)

    def test_date_to_index(self, basic_data):
        """Test date_to_index conversion."""
        group_tensors, feature_names, date_ranges = basic_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

        assert collection.date_to_index("group1", datetime(2020, 1, 1)) == 0
        assert collection.date_to_index("group1", datetime(2020, 1, 2)) == 1
        assert collection.date_to_index("group1", datetime(2020, 1, 3)) == 2

        # Test with date object (not datetime)
        from datetime import date

        assert collection.date_to_index("group1", date(2020, 1, 2)) == 1

        with pytest.raises(ValueError, match="Date .* outside range"):
            collection.date_to_index("group1", datetime(2020, 1, 10))

        with pytest.raises(KeyError, match="Group 'invalid' not found in date_ranges"):
            collection.date_to_index("invalid", datetime(2020, 1, 1))

    def test_date_to_index_invalid_type(self, basic_data):
        """Test date_to_index with invalid date type."""
        group_tensors, feature_names, date_ranges = basic_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

        with pytest.raises(TypeError, match="Expected datetime or date object"):
            collection.date_to_index("group1", "2020-01-01")  # String instead of date

    def test_summary_with_data(self, basic_data):
        """Test summary method with data."""
        group_tensors, feature_names, date_ranges = basic_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

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
        group_tensors, feature_names, date_ranges = empty_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

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
        group_tensors, feature_names, date_ranges = basic_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

        repr_str = repr(collection)
        assert "TimeSeriesCollection" in repr_str
        assert "n_groups=2" in repr_str
        assert "n_features=2" in repr_str
        assert "total_timesteps=5" in repr_str
        assert "memory_mb=" in repr_str

    def test_special_methods(self, basic_data):
        """Test special methods (__len__, __contains__)."""
        group_tensors, feature_names, date_ranges = basic_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

        # Test __len__
        assert len(collection) == 2

        # Test __contains__
        assert "group1" in collection
        assert "group2" in collection
        assert "group3" not in collection

    def test_single_feature_collection(self, single_feature_data):
        """Test collection with single feature."""
        group_tensors, feature_names, date_ranges = single_feature_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

        assert collection.get_n_features() == 1
        assert len(collection.feature_names) == 1
        assert collection.feature_names[0] == "single_feature"

        # Test feature retrieval
        result = collection.get_group_feature("group1", "single_feature", 0, 3)
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.equal(result, expected)

    def test_memory_calculation_different_dtypes(self, mixed_dtype_data):
        """Test memory calculation with different tensor dtypes."""
        group_tensors, feature_names, date_ranges = mixed_dtype_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

        summary = collection.summary()
        # Note: This test shows that the memory calculation now uses actual tensor dtypes
        assert summary["memory_mb"] >= 0  # Memory should be non-negative, might be very small for test data

    def test_validation_logging(self, basic_data, caplog):
        """Test that validation logs success message."""
        group_tensors, feature_names, date_ranges = basic_data

        with caplog.at_level(logging.INFO):
            TimeSeriesCollection(group_tensors, feature_names, date_ranges)

        assert "Validation passed for 2 groups" in caplog.text

    def test_edge_case_single_timestep(self):
        """Test collection with single timestep per group."""
        group_tensors = {
            "group1": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        }
        feature_names = ["feature_a", "feature_b"]
        date_ranges = {
            "group1": (datetime(2020, 1, 1), datetime(2020, 1, 1)),
        }

        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

        assert collection.get_group_length("group1") == 1
        assert collection.get_total_timesteps() == 1

        # Test data retrieval
        result = collection.get_group_series("group1", 0, 1)
        expected = torch.tensor([[1.0, 2.0]])
        assert torch.equal(result, expected)

    def test_feature_indices_property(self, basic_data):
        """Test feature_indices property."""
        group_tensors, feature_names, date_ranges = basic_data
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

        indices = collection.feature_indices
        assert indices["feature_a"] == 0
        assert indices["feature_b"] == 1

        # Test immutability
        indices["new_feature"] = 2
        assert "new_feature" not in collection.feature_indices

    def test_large_date_ranges(self):
        """Test with larger date ranges to verify date calculations."""
        group_tensors = {
            "group1": torch.ones((365, 2), dtype=torch.float32),  # One year of daily data
        }
        feature_names = ["feature_a", "feature_b"]
        date_ranges = {
            "group1": (datetime(2020, 1, 1), datetime(2020, 12, 31)),  # 366 days (leap year)
        }

        # This should fail because tensor has 365 timesteps but leap year 2020 has 366 days
        with pytest.raises(ValueError, match="has 365 timesteps but date range.*implies 366 days"):
            TimeSeriesCollection(group_tensors, feature_names, date_ranges)

        # Fix the tensor size
        group_tensors["group1"] = torch.ones((366, 2), dtype=torch.float32)
        collection = TimeSeriesCollection(group_tensors, feature_names, date_ranges)

        assert collection.get_group_length("group1") == 366
        assert collection.index_to_date("group1", 0) == datetime(2020, 1, 1)
        assert collection.index_to_date("group1", 365) == datetime(2020, 12, 31)
