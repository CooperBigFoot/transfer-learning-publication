import numpy as np
import polars as pl
import pytest

from transfer_learning_publication.cleaners import trim_to_column


class TestTrimToColumn:
    """Test trim_to_column function from cleaners module."""

    @pytest.fixture
    def basic_data(self):
        """Basic test data with nulls at start and end."""
        return pl.LazyFrame(
            {
                "priority_col": [None, None, 1.0, 2.0, 3.0, None, None],
                "other_col": [10, 20, 30, 40, 50, 60, 70],
                "string_col": ["a", "b", "c", "d", "e", "f", "g"],
            }
        )

    @pytest.fixture
    def interleaved_nulls_data(self):
        """Data with interleaved nulls/NaNs in priority column."""
        return pl.LazyFrame(
            {
                "priority_col": [1.0, 2.0, None, 3.0, float("nan"), 4.0, None, 5.0, None, None],
                "other_col": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "string_col": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            }
        )

    @pytest.fixture
    def nan_only_data(self):
        """Data with only NaNs (no nulls) in priority column."""
        return pl.LazyFrame(
            {
                "priority_col": [float("nan"), float("nan"), 1.0, 2.0, 3.0, float("nan"), float("nan")],
                "other_col": [10, 20, 30, 40, 50, 60, 70],
                "string_col": ["a", "b", "c", "d", "e", "f", "g"],
            }
        )

    @pytest.fixture
    def time_series_data(self):
        """Time series data scenario."""
        return pl.LazyFrame(
            {
                "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
                "temperature": [20.0, 21.0, 22.0, 23.0, 24.0],
                "precipitation": [5.0, 6.0, 7.0, 8.0, 9.0],
                "streamflow": [None, 100.0, 110.0, 120.0, None],
            }
        )

    @pytest.fixture
    def special_values_data(self):
        """Data with infinity values."""
        return pl.LazyFrame(
            {
                "priority_col": [float("inf"), 1.0, 2.0, float("-inf"), None],
                "other_col": [10, 20, 30, 40, 50],
                "string_col": ["a", "b", "c", "d", "e"],
            }
        )

    @pytest.fixture
    def all_null_data(self):
        """Data where priority column is all nulls."""
        return pl.LazyFrame({"priority_col": [None, None, None], "other_col": [10, 20, 30]})

    @pytest.fixture
    def no_null_data(self):
        """Data with no nulls in priority column."""
        return pl.LazyFrame(
            {
                "priority_col": [1.0, 2.0, 3.0, 4.0, 5.0],
                "other_col": [10, 20, 30, 40, 50],
                "string_col": ["a", "b", "c", "d", "e"],
            }
        )

    @pytest.fixture
    def single_valid_data(self):
        """Data with only one valid value."""
        return pl.LazyFrame({"priority_col": [None, None, 42.0, None, None], "other_col": [10, 20, 30, 40, 50]})

    @pytest.fixture
    def empty_data(self):
        """Empty LazyFrame."""
        return pl.LazyFrame(
            {"priority_col": [], "other_col": []}, schema={"priority_col": pl.Float64, "other_col": pl.Int64}
        )

    @pytest.fixture
    def large_data(self):
        """Large dataset for performance testing."""
        n_rows = 100_000
        priority_data = [None] * 1000 + list(range(n_rows - 2000)) + [None] * 1000
        return pl.LazyFrame({"priority_col": priority_data, "other_col": list(range(n_rows))})

    def test_basic_trimming(self, basic_data):
        """Test basic trimming with nulls at start and end."""
        result = trim_to_column(basic_data, "priority_col")
        collected = result.collect()

        # Should keep rows 2, 3, 4 (indices where priority_col has values 1.0, 2.0, 3.0)
        assert collected.shape[0] == 3
        assert collected["priority_col"].to_list() == [1.0, 2.0, 3.0]
        assert collected["other_col"].to_list() == [30, 40, 50]
        assert collected["string_col"].to_list() == ["c", "d", "e"]

    def test_interleaved_nulls_nans(self, interleaved_nulls_data):
        """Test trimming with interleaved nulls and NaNs."""
        result = trim_to_column(interleaved_nulls_data, "priority_col")
        collected = result.collect()

        # Should keep rows 0-7 (trimming only trailing nulls at end)
        assert collected.shape[0] == 8
        assert collected["other_col"].to_list() == [10, 20, 30, 40, 50, 60, 70, 80]
        assert collected["string_col"].to_list() == ["a", "b", "c", "d", "e", "f", "g", "h"]

        # Check that first and last values in priority col are valid
        priority_values = collected["priority_col"].to_list()
        assert priority_values[0] == 1.0
        assert priority_values[-1] == 5.0

    def test_nan_only_trimming(self, nan_only_data):
        """Test trimming with only NaNs (no nulls)."""
        result = trim_to_column(nan_only_data, "priority_col")
        collected = result.collect()

        # Should keep rows 2, 3, 4 (where priority_col has values 1.0, 2.0, 3.0)
        assert collected.shape[0] == 3
        assert collected["priority_col"].to_list() == [1.0, 2.0, 3.0]
        assert collected["other_col"].to_list() == [30, 40, 50]
        assert collected["string_col"].to_list() == ["c", "d", "e"]

    def test_time_series_scenario(self, time_series_data):
        """Test time series trimming scenario."""
        result = trim_to_column(time_series_data, "streamflow")
        collected = result.collect()

        # Should keep rows 1, 2, 3 (where streamflow has values)
        assert collected.shape[0] == 3
        assert collected["date"].to_list() == ["2020-01-02", "2020-01-03", "2020-01-04"]
        assert collected["temperature"].to_list() == [21.0, 22.0, 23.0]
        assert collected["precipitation"].to_list() == [6.0, 7.0, 8.0]
        assert collected["streamflow"].to_list() == [100.0, 110.0, 120.0]

    def test_special_infinity_values(self, special_values_data):
        """Test that infinity values are NOT treated as null/NaN."""
        result = trim_to_column(special_values_data, "priority_col")
        collected = result.collect()

        # Should keep rows 0, 1, 2, 3 (infinity values are valid, only null at end is trimmed)
        assert collected.shape[0] == 4
        assert collected["other_col"].to_list() == [10, 20, 30, 40]
        assert collected["string_col"].to_list() == ["a", "b", "c", "d"]

        # Check infinity values are preserved
        priority_values = collected["priority_col"].to_list()
        assert np.isinf(priority_values[0]) and priority_values[0] > 0  # +inf
        assert np.isinf(priority_values[3]) and priority_values[3] < 0  # -inf

    def test_all_nulls_error_or_empty(self, all_null_data):
        """Test behavior when all values in priority column are null."""
        result = trim_to_column(all_null_data, "priority_col")
        collected = result.collect()

        # Should return empty result (no valid range found)
        assert collected.shape[0] == 0

    def test_no_nulls_unchanged(self, no_null_data):
        """Test that data with no nulls returns unchanged."""
        result = trim_to_column(no_null_data, "priority_col")
        collected = result.collect()
        original = no_null_data.collect()

        # Should be identical to original
        assert collected.shape == original.shape
        assert collected["priority_col"].to_list() == original["priority_col"].to_list()
        assert collected["other_col"].to_list() == original["other_col"].to_list()
        assert collected["string_col"].to_list() == original["string_col"].to_list()

    def test_single_valid_value(self, single_valid_data):
        """Test with only one valid value in priority column."""
        result = trim_to_column(single_valid_data, "priority_col")
        collected = result.collect()

        # Should keep only row 2 (where value is 42.0)
        assert collected.shape[0] == 1
        assert collected["priority_col"].to_list() == [42.0]
        assert collected["other_col"].to_list() == [30]

    def test_invalid_column_error(self, basic_data):
        """Test error when priority column doesn't exist."""
        with pytest.raises(ValueError, match="Column 'nonexistent' not found in LazyFrame"):
            trim_to_column(basic_data, "nonexistent")

    def test_empty_dataframe(self, empty_data):
        """Test with empty LazyFrame."""
        result = trim_to_column(empty_data, "priority_col")
        collected = result.collect()

        # Should return empty result
        assert collected.shape[0] == 0
        assert list(collected.columns) == ["priority_col", "other_col"]

    def test_column_order_preservation(self, basic_data):
        """Test that column order is preserved."""
        original_columns = basic_data.collect_schema().names()
        result = trim_to_column(basic_data, "priority_col")
        result_columns = result.collect_schema().names()

        assert result_columns == original_columns

    def test_column_types_preservation(self, basic_data):
        """Test that column data types are preserved."""
        original_schema = basic_data.collect_schema()
        result = trim_to_column(basic_data, "priority_col")
        result_schema = result.collect_schema()

        assert result_schema == original_schema

    def test_lazy_evaluation_maintained(self, large_data):
        """Test that LazyFrame remains lazy during operation."""
        result = trim_to_column(large_data, "priority_col")

        # Should return LazyFrame without collecting
        assert isinstance(result, pl.LazyFrame)

        # When collected, should have correct trimmed size
        collected = result.collect()
        assert collected.shape[0] == 98_000  # 100k - 2k nulls

    def test_different_numeric_types(self):
        """Test with different numeric data types."""
        data = pl.LazyFrame(
            {
                "int_col": [None, 1, 2, 3, None],
                "float32_col": [10.0, 20.0, 30.0, 40.0, 50.0],
                "float64_col": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        ).with_columns(
            [
                pl.col("float32_col").cast(pl.Float32),
                pl.col("float64_col").cast(pl.Float64),
                pl.col("int_col").cast(pl.Int64),
            ]
        )

        result = trim_to_column(data, "int_col")
        collected = result.collect()

        # Should keep rows 1, 2, 3
        assert collected.shape[0] == 3
        assert collected["int_col"].to_list() == [1, 2, 3]
        assert collected["float32_col"].to_list() == [20.0, 30.0, 40.0]
        assert collected["float64_col"].to_list() == [200.0, 300.0, 400.0]

        # Check types are preserved
        assert collected.schema["int_col"] == pl.Int64
        assert collected.schema["float32_col"] == pl.Float32
        assert collected.schema["float64_col"] == pl.Float64

    def test_string_column_with_nulls(self):
        """Test with string priority column containing nulls."""
        data = pl.LazyFrame({"string_priority": [None, "a", "b", "c", None], "numeric_col": [10, 20, 30, 40, 50]})

        result = trim_to_column(data, "string_priority")
        collected = result.collect()

        # Should keep rows 1, 2, 3
        assert collected.shape[0] == 3
        assert collected["string_priority"].to_list() == ["a", "b", "c"]
        assert collected["numeric_col"].to_list() == [20, 30, 40]

    def test_mixed_null_nan_comprehensive(self):
        """Comprehensive test with various null/NaN patterns."""
        data = pl.LazyFrame(
            {
                "priority": [None, float("nan"), 1.0, None, 2.0, float("nan"), 3.0, None, float("nan")],
                "other": [10, 20, 30, 40, 50, 60, 70, 80, 90],
            }
        )

        result = trim_to_column(data, "priority")
        collected = result.collect()

        # Should keep rows 2-6 (from first valid value 1.0 to last valid value 3.0)
        assert collected.shape[0] == 5
        assert collected["other"].to_list() == [30, 40, 50, 60, 70]

        # Check priority column values
        priority_values = collected["priority"].to_list()
        assert priority_values[0] == 1.0  # First valid value
        assert priority_values[1] is None  # Null in middle
        assert priority_values[2] == 2.0  # Second valid value
        assert np.isnan(priority_values[3])  # NaN in middle
        assert priority_values[4] == 3.0  # Last valid value
