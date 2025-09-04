
import polars as pl
import pytest

from transfer_learning_publication.cleaners import fill_na_columns


class TestFillNaColumns:
    """Test suite for the fill_na_columns function."""

    def test_fill_single_integer_column(self):
        """Test filling nulls in a single integer column."""
        df = pl.LazyFrame({"values": [1, None, 3, None, 5]})

        result = fill_na_columns(df, "values", fill_value=-999).collect()
        expected = pl.DataFrame({"values": [1, -999, 3, -999, 5]})

        assert result.equals(expected)

    def test_fill_single_float_column_with_nulls_and_nans(self):
        """Test filling both nulls and NaNs in a single float column."""
        df = pl.LazyFrame({"values": [1.0, None, 3.0, float("nan"), 5.0]})

        result = fill_na_columns(df, "values", fill_value=0.0).collect()

        # Check non-null/nan values are preserved
        assert result["values"][0] == 1.0
        assert result["values"][2] == 3.0
        assert result["values"][4] == 5.0
        # Check nulls and NaNs are filled
        assert result["values"][1] == 0.0
        assert result["values"][3] == 0.0

    def test_fill_multiple_columns(self):
        """Test filling nulls in multiple columns."""
        df = pl.LazyFrame(
            {
                "col1": [1, None, 3, None],
                "col2": [1.0, None, float("nan"), 4.0],
                "col3": [10, 20, 30, 40],  # This column won't be filled
            }
        )

        result = fill_na_columns(df, ["col1", "col2"], fill_value=-1).collect()
        expected = pl.DataFrame({"col1": [1, -1, 3, -1], "col2": [1.0, -1.0, -1.0, 4.0], "col3": [10, 20, 30, 40]})

        assert result.equals(expected)

    def test_string_column_name_normalization(self):
        """Test that single string column name is properly normalized to list."""
        df = pl.LazyFrame({"values": [1, None, 3, None, 5]})

        # Test with string input
        result1 = fill_na_columns(df, "values", fill_value=42).collect()
        # Test with list input (should be equivalent)
        result2 = fill_na_columns(df, ["values"], fill_value=42).collect()

        assert result1.equals(result2)

    def test_preserve_non_null_values(self):
        """Test that non-null values are preserved unchanged."""
        df = pl.LazyFrame({"values": [1.5, 2.7, 3.14, 4.0, 5.9]})

        result = fill_na_columns(df, "values", fill_value=99.0).collect()
        expected = pl.DataFrame({"values": [1.5, 2.7, 3.14, 4.0, 5.9]})

        assert result.equals(expected)

    def test_flag_columns_false_default(self):
        """Test that no flag columns are added when add_binary_flag=False (default)."""
        df = pl.LazyFrame({"values": [1, None, 3, None, 5]})

        result = fill_na_columns(df, "values", fill_value=0).collect()

        # Should only have the original column
        assert list(result.columns) == ["values"]
        assert result.shape[1] == 1

    def test_flag_columns_single_column(self):
        """Test flag column creation for single column when add_binary_flag=True."""
        df = pl.LazyFrame({"values": [1, None, 3, None, 5]})

        result = fill_na_columns(df, "values", fill_value=0, add_binary_flag=True).collect()

        # Should have original column plus flag column
        assert list(result.columns) == ["values", "values_was_filled"]
        assert result["values"].to_list() == [1, 0, 3, 0, 5]
        assert result["values_was_filled"].to_list() == [0, 1, 0, 1, 0]

        # Flag column should be UInt8
        assert result.schema["values_was_filled"] == pl.UInt8

    def test_flag_columns_multiple_columns(self):
        """Test flag column creation for multiple columns."""
        df = pl.LazyFrame(
            {"col1": [1, None, 3, None], "col2": [1.0, None, float("nan"), 4.0], "col3": [100, 200, 300, 400]}
        )

        result = fill_na_columns(df, ["col1", "col2"], fill_value=-1, add_binary_flag=True).collect()

        # Should have original columns plus flag columns at the end
        expected_columns = ["col1", "col2", "col3", "col1_was_filled", "col2_was_filled"]
        assert list(result.columns) == expected_columns

        # Check flag values
        assert result["col1_was_filled"].to_list() == [0, 1, 0, 1]
        assert result["col2_was_filled"].to_list() == [0, 1, 1, 0]

        # Check flag data types
        assert result.schema["col1_was_filled"] == pl.UInt8
        assert result.schema["col2_was_filled"] == pl.UInt8

    def test_flag_columns_with_float_nans(self):
        """Test that flag columns correctly identify both nulls and NaNs in float columns."""
        df = pl.LazyFrame({"values": [1.0, None, 3.0, float("nan"), 5.0]})

        result = fill_na_columns(df, "values", fill_value=0.0, add_binary_flag=True).collect()

        assert result["values_was_filled"].to_list() == [0, 1, 0, 1, 0]

    def test_different_numeric_types(self):
        """Test filling works with different numeric types."""
        df = pl.LazyFrame(
            {
                "int32_col": pl.Series([1, None, 3], dtype=pl.Int32),
                "float64_col": pl.Series([1.0, None, 3.0], dtype=pl.Float64),
                "uint32_col": pl.Series([1, None, 3], dtype=pl.UInt32),
            }
        )

        result = fill_na_columns(df, ["int32_col", "float64_col", "uint32_col"], fill_value=42).collect()

        expected = pl.DataFrame(
            {
                "int32_col": pl.Series([1, 42, 3], dtype=pl.Int32),
                "float64_col": pl.Series([1.0, 42.0, 3.0], dtype=pl.Float64),
                "uint32_col": pl.Series([1, 42, 3], dtype=pl.UInt32),
            }
        )

        assert result.equals(expected)

    def test_empty_columns_list_warning(self):
        """Test that a warning is issued when no columns are provided."""
        df = pl.LazyFrame({"values": [1, 2, 3]})

        with pytest.warns(UserWarning, match="No columns provided"):
            result = fill_na_columns(df, [], fill_value=99).collect()

        # Should return unchanged
        expected = pl.DataFrame({"values": [1, 2, 3]})
        assert result.equals(expected)

    def test_column_not_found_error(self):
        """Test error when specified column doesn't exist."""
        df = pl.LazyFrame({"existing_col": [1, 2, 3]})

        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            fill_na_columns(df, "nonexistent", fill_value=0).collect()

    def test_non_numeric_column_error(self):
        """Test error when trying to fill non-numeric column."""
        df = pl.LazyFrame({"string_col": ["a", "b", "c"]})

        with pytest.raises(ValueError, match="Column 'string_col' is not numeric"):
            fill_na_columns(df, "string_col", fill_value=0).collect()

    def test_mixed_numeric_and_non_numeric_columns(self):
        """Test error when some columns in list are non-numeric."""
        df = pl.LazyFrame({"numeric_col": [1.0, 2.0, 3.0], "string_col": ["a", "b", "c"]})

        with pytest.raises(ValueError, match="Column 'string_col' is not numeric"):
            fill_na_columns(df, ["numeric_col", "string_col"], fill_value=0).collect()

    def test_empty_dataframe(self):
        """Test filling on empty dataframe."""
        df = pl.LazyFrame({"values": []}, schema={"values": pl.Float64})

        result = fill_na_columns(df, "values", fill_value=0.0).collect()
        expected = pl.DataFrame({"values": []}, schema={"values": pl.Float64})

        assert result.equals(expected)

    def test_empty_dataframe_with_flags(self):
        """Test filling on empty dataframe with flag columns."""
        df = pl.LazyFrame({"values": []}, schema={"values": pl.Int64})

        result = fill_na_columns(df, "values", fill_value=0, add_binary_flag=True).collect()

        assert list(result.columns) == ["values", "values_was_filled"]
        assert result.shape[0] == 0
        assert result.schema["values_was_filled"] == pl.UInt8

    def test_all_null_values(self):
        """Test filling when all values are null."""
        df = pl.LazyFrame({"values": [None, None, None]})

        result = fill_na_columns(df, "values", fill_value=42, add_binary_flag=True).collect()

        assert result["values"].to_list() == [42, 42, 42]
        assert result["values_was_filled"].to_list() == [1, 1, 1]

    def test_all_nan_values(self):
        """Test filling when all values are NaN."""
        df = pl.LazyFrame({"values": [float("nan"), float("nan"), float("nan")]})

        result = fill_na_columns(df, "values", fill_value=42.0, add_binary_flag=True).collect()

        assert result["values"].to_list() == [42.0, 42.0, 42.0]
        assert result["values_was_filled"].to_list() == [1, 1, 1]

    def test_preserves_other_columns(self):
        """Test that columns not being filled are preserved unchanged."""
        df = pl.LazyFrame(
            {
                "to_fill": [1, None, 3],
                "preserve_int": [100, 200, 300],
                "preserve_str": ["hello", "world", "test"],
                "preserve_bool": [True, False, True],
            }
        )

        result = fill_na_columns(df, "to_fill", fill_value=99).collect()

        # Check filled column
        assert result["to_fill"].to_list() == [1, 99, 3]
        # Check preserved columns
        assert result["preserve_int"].to_list() == [100, 200, 300]
        assert result["preserve_str"].to_list() == ["hello", "world", "test"]
        assert result["preserve_bool"].to_list() == [True, False, True]

    def test_column_order_preservation(self):
        """Test that column order is preserved."""
        df = pl.LazyFrame({"first": [1, None, 3], "second": [10, 20, 30], "third": [100, None, 300]})

        original_columns = df.collect_schema().names()
        result = fill_na_columns(df, ["first", "third"], fill_value=0)
        result_columns = result.collect_schema().names()

        assert result_columns == original_columns

    def test_column_order_with_flags(self):
        """Test that flag columns appear at the end."""
        df = pl.LazyFrame({"first": [1, None, 3], "second": [10, 20, 30], "third": [100, None, 300]})

        result = fill_na_columns(df, ["first", "third"], fill_value=0, add_binary_flag=True).collect()

        expected_columns = ["first", "second", "third", "first_was_filled", "third_was_filled"]
        assert list(result.columns) == expected_columns

    def test_column_types_preservation(self):
        """Test that original column data types are preserved."""
        df = pl.LazyFrame(
            {
                "int_col": pl.Series([1, None, 3], dtype=pl.Int32),
                "float_col": pl.Series([1.0, None, 3.0], dtype=pl.Float64),
            }
        )

        original_schema = df.collect_schema()
        result = fill_na_columns(df, ["int_col", "float_col"], fill_value=42)
        result_schema = result.collect_schema()

        # Original columns should keep their types
        assert result_schema["int_col"] == original_schema["int_col"]
        assert result_schema["float_col"] == original_schema["float_col"]

    def test_lazy_evaluation_maintained(self):
        """Test that LazyFrame remains lazy during operation."""
        # Create large dataset
        n_rows = 10000
        df = pl.LazyFrame({"values": [1 if i % 3 != 0 else None for i in range(n_rows)], "other": list(range(n_rows))})

        result = fill_na_columns(df, "values", fill_value=999)

        # Should return LazyFrame without collecting
        assert isinstance(result, pl.LazyFrame)

        # When collected, should have correct filled values
        collected = result.collect()
        assert collected.shape[0] == n_rows
        # Check that nulls were filled (indices where i % 3 == 0 should be 999)
        assert collected["values"][0] == 999  # i=0: 0 % 3 == 0, so None -> 999
        assert collected["values"][3] == 999  # i=3: 3 % 3 == 0, so None -> 999
        assert collected["values"][6] == 999  # i=6: 6 % 3 == 0, so None -> 999
        # Check that non-nulls were preserved
        assert collected["values"][1] == 1  # i=1: 1 % 3 != 0, so 1 -> 1
        assert collected["values"][2] == 1  # i=2: 2 % 3 != 0, so 1 -> 1

    def test_integer_fill_value_types(self):
        """Test different integer fill value types."""
        df = pl.LazyFrame({"values": [1, None, 3]})

        # Test with int
        result1 = fill_na_columns(df, "values", fill_value=42).collect()
        assert result1["values"][1] == 42

        # Test with float (should work for int columns)
        result2 = fill_na_columns(df, "values", fill_value=42.0).collect()
        assert result2["values"][1] == 42

    def test_float_fill_value_types(self):
        """Test different float fill value types."""
        df = pl.LazyFrame({"values": [1.0, None, 3.0]})

        # Test with float
        result1 = fill_na_columns(df, "values", fill_value=42.5).collect()
        assert result1["values"][1] == 42.5

        # Test with int (should work for float columns)
        result2 = fill_na_columns(df, "values", fill_value=42).collect()
        assert result2["values"][1] == 42.0

    def test_edge_case_zero_fill_value(self):
        """Test filling with zero as fill value."""
        df = pl.LazyFrame({"values": [1.0, None, float("nan"), 4.0]})

        result = fill_na_columns(df, "values", fill_value=0.0).collect()
        expected = pl.DataFrame({"values": [1.0, 0.0, 0.0, 4.0]})

        assert result.equals(expected)

    def test_edge_case_negative_fill_value(self):
        """Test filling with negative fill value."""
        df = pl.LazyFrame({"values": [1, None, 3, None, 5]})

        result = fill_na_columns(df, "values", fill_value=-999).collect()
        expected = pl.DataFrame({"values": [1, -999, 3, -999, 5]})

        assert result.equals(expected)

    def test_comprehensive_scenario(self):
        """Comprehensive test combining multiple features."""
        df = pl.LazyFrame(
            {
                "sensor_1": pl.Series([1.0, None, 3.0, float("nan"), 5.0], dtype=pl.Float64),
                "sensor_2": pl.Series([10, None, 30, 40, None], dtype=pl.Int32),
                "metadata": ["A", "B", "C", "D", "E"],
                "quality": [True, True, False, True, False],
            }
        )

        result = fill_na_columns(df, ["sensor_1", "sensor_2"], fill_value=-999, add_binary_flag=True).collect()

        # Check structure
        expected_columns = ["sensor_1", "sensor_2", "metadata", "quality", "sensor_1_was_filled", "sensor_2_was_filled"]
        assert list(result.columns) == expected_columns

        # Check filled values
        assert result["sensor_1"].to_list() == [1.0, -999.0, 3.0, -999.0, 5.0]
        assert result["sensor_2"].to_list() == [10, -999, 30, 40, -999]

        # Check flags
        assert result["sensor_1_was_filled"].to_list() == [0, 1, 0, 1, 0]
        assert result["sensor_2_was_filled"].to_list() == [0, 1, 0, 0, 1]

        # Check preserved columns
        assert result["metadata"].to_list() == ["A", "B", "C", "D", "E"]
        assert result["quality"].to_list() == [True, True, False, True, False]

        # Check data types
        assert result.schema["sensor_1"] == pl.Float64
        assert result.schema["sensor_2"] == pl.Int32
        assert result.schema["sensor_1_was_filled"] == pl.UInt8
        assert result.schema["sensor_2_was_filled"] == pl.UInt8
