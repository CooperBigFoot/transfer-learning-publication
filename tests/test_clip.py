from math import isnan

import polars as pl
import pytest

from transfer_learning_publication.cleaners import clip_columns


class TestClipColumns:
    """Test suite for the clip_columns function."""

    def test_clip_single_column_with_min_max(self):
        """Test clipping a single column with both min and max values."""
        # Create test data with values below, within, and above the range
        df = pl.LazyFrame({"values": [-5.0, 0.0, 5.0, 10.0, 15.0]})

        result = clip_columns(df, "values", min_value=2.0, max_value=8.0).collect()
        expected = pl.DataFrame({"values": [2.0, 2.0, 5.0, 8.0, 8.0]})

        assert result.equals(expected)

    def test_clip_multiple_columns(self):
        """Test clipping multiple columns."""
        df = pl.LazyFrame(
            {
                "col1": [-10.0, 0.0, 5.0, 20.0],
                "col2": [-5.0, 2.0, 8.0, 15.0],
                "col3": [1.0, 2.0, 3.0, 4.0],  # This column won't be clipped
            }
        )

        result = clip_columns(df, ["col1", "col2"], min_value=0.0, max_value=10.0).collect()
        expected = pl.DataFrame(
            {"col1": [0.0, 0.0, 5.0, 10.0], "col2": [0.0, 2.0, 8.0, 10.0], "col3": [1.0, 2.0, 3.0, 4.0]}
        )

        assert result.equals(expected)

    def test_clip_with_only_min_value(self):
        """Test clipping with only minimum value specified."""
        df = pl.LazyFrame({"values": [-10.0, -1.0, 0.0, 5.0, 10.0]})

        result = clip_columns(df, "values", min_value=0.0).collect()
        expected = pl.DataFrame({"values": [0.0, 0.0, 0.0, 5.0, 10.0]})

        assert result.equals(expected)

    def test_clip_with_only_max_value(self):
        """Test clipping with only maximum value specified."""
        df = pl.LazyFrame({"values": [-5.0, 0.0, 5.0, 10.0, 15.0]})

        result = clip_columns(df, "values", max_value=8.0).collect()
        expected = pl.DataFrame({"values": [-5.0, 0.0, 5.0, 8.0, 8.0]})

        assert result.equals(expected)

    def test_clip_with_nulls_and_nans(self):
        """Test that nulls and NaNs are preserved unchanged."""
        df = pl.LazyFrame({"values": [-5.0, None, 5.0, float("nan"), 15.0]})

        result = clip_columns(df, "values", min_value=0.0, max_value=10.0).collect()

        # Check non-null/nan values
        assert result["values"][0] == 0.0  # -5.0 clipped to 0.0
        assert result["values"][1] is None  # null preserved
        assert result["values"][2] == 5.0  # 5.0 unchanged
        assert isnan(result["values"][3])  # NaN preserved
        assert result["values"][4] == 10.0  # 15.0 clipped to 10.0

    def test_clip_different_numeric_types(self):
        """Test clipping works with different numeric types."""
        df = pl.LazyFrame(
            {
                "int_col": pl.Series([1, 5, 10, 15], dtype=pl.Int32),
                "float_col": pl.Series([1.5, 5.5, 10.5, 15.5], dtype=pl.Float64),
                "uint_col": pl.Series([1, 5, 10, 15], dtype=pl.UInt32),
            }
        )

        result = clip_columns(df, ["int_col", "float_col", "uint_col"], min_value=3, max_value=12).collect()

        expected = pl.DataFrame(
            {
                "int_col": pl.Series([3, 5, 10, 12], dtype=pl.Int32),
                "float_col": pl.Series([3.0, 5.5, 10.5, 12.0], dtype=pl.Float64),
                "uint_col": pl.Series([3, 5, 10, 12], dtype=pl.UInt32),
            }
        )

        assert result.equals(expected)

    def test_no_bounds_provided_warning(self):
        """Test that a warning is issued when no bounds are provided."""
        df = pl.LazyFrame({"values": [1.0, 2.0, 3.0]})

        with pytest.warns(UserWarning, match="No min or max value provided"):
            result = clip_columns(df, "values").collect()

        # Should return unchanged
        expected = pl.DataFrame({"values": [1.0, 2.0, 3.0]})
        assert result.equals(expected)

    def test_column_not_found_error(self):
        """Test error when specified column doesn't exist."""
        df = pl.LazyFrame({"existing_col": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            clip_columns(df, "nonexistent", min_value=0.0).collect()

    def test_non_numeric_column_error(self):
        """Test error when trying to clip non-numeric column."""
        df = pl.LazyFrame({"string_col": ["a", "b", "c"]})

        with pytest.raises(ValueError, match="Column 'string_col' is not numeric"):
            clip_columns(df, "string_col", min_value=0.0).collect()

    def test_mixed_numeric_and_non_numeric_columns(self):
        """Test error when some columns in list are non-numeric."""
        df = pl.LazyFrame({"numeric_col": [1.0, 2.0, 3.0], "string_col": ["a", "b", "c"]})

        with pytest.raises(ValueError, match="Column 'string_col' is not numeric"):
            clip_columns(df, ["numeric_col", "string_col"], min_value=0.0).collect()

    def test_string_column_name_normalization(self):
        """Test that single string column name is properly normalized to list."""
        df = pl.LazyFrame({"values": [-5.0, 0.0, 5.0, 15.0]})

        # Test with string input
        result1 = clip_columns(df, "values", min_value=0.0, max_value=10.0).collect()
        # Test with list input (should be equivalent)
        result2 = clip_columns(df, ["values"], min_value=0.0, max_value=10.0).collect()

        assert result1.equals(result2)

    def test_edge_case_min_equals_max(self):
        """Test clipping when min_value equals max_value."""
        df = pl.LazyFrame({"values": [-5.0, 0.0, 5.0, 10.0]})

        result = clip_columns(df, "values", min_value=3.0, max_value=3.0).collect()
        expected = pl.DataFrame({"values": [3.0, 3.0, 3.0, 3.0]})

        assert result.equals(expected)

    def test_no_clipping_needed(self):
        """Test when all values are within bounds."""
        df = pl.LazyFrame({"values": [2.0, 3.0, 4.0, 5.0]})

        result = clip_columns(df, "values", min_value=1.0, max_value=6.0).collect()
        expected = pl.DataFrame({"values": [2.0, 3.0, 4.0, 5.0]})

        assert result.equals(expected)

    def test_empty_dataframe(self):
        """Test clipping on empty dataframe."""
        df = pl.LazyFrame({"values": []}, schema={"values": pl.Float64})

        result = clip_columns(df, "values", min_value=0.0, max_value=10.0).collect()
        expected = pl.DataFrame({"values": []}, schema={"values": pl.Float64})

        assert result.equals(expected)

    def test_preserves_other_columns(self):
        """Test that columns not being clipped are preserved unchanged."""
        df = pl.LazyFrame(
            {
                "to_clip": [-5.0, 15.0],
                "preserve_int": [100, 200],
                "preserve_str": ["hello", "world"],
                "preserve_bool": [True, False],
            }
        )

        result = clip_columns(df, "to_clip", min_value=0.0, max_value=10.0).collect()

        # Check clipped column
        assert result["to_clip"].to_list() == [0.0, 10.0]
        # Check preserved columns
        assert result["preserve_int"].to_list() == [100, 200]
        assert result["preserve_str"].to_list() == ["hello", "world"]
        assert result["preserve_bool"].to_list() == [True, False]
