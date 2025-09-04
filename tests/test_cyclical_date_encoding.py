import polars as pl
import pytest

from transfer_learning_publication.cleaners.cyclical_date_encoding import add_cyclical_date_encoding


class TestAddCyclicalDateEncoding:
    """Test suite for the add_cyclical_date_encoding function."""

    def test_basic_date_encoding(self):
        """Test basic cyclical encoding with date column."""
        df = pl.LazyFrame({"date": ["2024-01-01", "2024-07-01", "2024-12-31"]}).with_columns(
            pl.col("date").str.to_date()
        )

        result = add_cyclical_date_encoding(df).collect()

        # Check new columns exist
        assert "sin_day_of_year" in result.columns
        assert "cos_day_of_year" in result.columns
        assert result.shape[1] == 3  # original + 2 new columns

        # Check values are in expected range [-1, 1]
        sin_values = result["sin_day_of_year"].to_list()
        cos_values = result["cos_day_of_year"].to_list()

        for sin_val in sin_values:
            assert -1 <= sin_val <= 1
        for cos_val in cos_values:
            assert -1 <= cos_val <= 1

        # Check sin^2 + cos^2 ≈ 1 (unit circle property)
        for sin_val, cos_val in zip(sin_values, cos_values, strict=False):
            assert abs(sin_val**2 + cos_val**2 - 1) < 1e-10

    def test_datetime_column_encoding(self):
        """Test cyclical encoding with datetime column."""
        df = pl.LazyFrame(
            {"timestamp": ["2024-01-01T12:30:00", "2024-06-15T08:45:00", "2024-12-31T23:59:00"]}
        ).with_columns(pl.col("timestamp").str.to_datetime())

        result = add_cyclical_date_encoding(df, date_column="timestamp").collect()

        # Check structure
        assert "sin_day_of_year" in result.columns
        assert "cos_day_of_year" in result.columns
        assert result.shape[1] == 3

    def test_custom_column_name(self):
        """Test with custom date column name."""
        df = pl.LazyFrame({"measurement_date": ["2024-01-01", "2024-06-15"], "value": [1.0, 2.0]}).with_columns(
            pl.col("measurement_date").str.to_date()
        )

        result = add_cyclical_date_encoding(df, date_column="measurement_date").collect()

        assert "sin_day_of_year" in result.columns
        assert "cos_day_of_year" in result.columns
        assert "measurement_date" in result.columns
        assert "value" in result.columns

    def test_january_first_encoding(self):
        """Test that January 1st gives expected values (day 1 = 0 normalized)."""
        df = pl.LazyFrame({"date": ["2024-01-01"]}).with_columns(pl.col("date").str.to_date())

        result = add_cyclical_date_encoding(df).collect()

        # January 1st = day 1, normalized to 0, so sin(0) = 0, cos(0) = 1
        sin_val = result["sin_day_of_year"][0]
        cos_val = result["cos_day_of_year"][0]

        assert abs(sin_val) < 1e-10  # Should be ~0
        assert abs(cos_val - 1) < 1e-10  # Should be ~1

    def test_mid_year_encoding(self):
        """Test encoding around mid-year (should be opposite to Jan 1)."""
        # July 1st is approximately day 182 in non-leap year, day 183 in leap year
        df = pl.LazyFrame({"date": ["2023-07-02"]}).with_columns(pl.col("date").str.to_date())  # Non-leap year

        result = add_cyclical_date_encoding(df).collect()

        sin_val = result["sin_day_of_year"][0]
        cos_val = result["cos_day_of_year"][0]

        # Mid-year should have sin close to 0 and cos close to -1
        assert abs(sin_val) < 0.1  # Allow some tolerance since it's not exact halfway
        assert cos_val < -0.8  # Should be strongly negative

    def test_leap_year_handling(self):
        """Test that leap years are handled (though implementation uses 365.25 normalization)."""
        df = pl.LazyFrame({"date": ["2024-02-29", "2024-12-31"]}).with_columns(  # 2024 is leap year
            pl.col("date").str.to_date()
        )

        result = add_cyclical_date_encoding(df).collect()

        # Should still produce valid values even for leap year
        sin_values = result["sin_day_of_year"].to_list()
        cos_values = result["cos_day_of_year"].to_list()

        for sin_val in sin_values:
            assert -1 <= sin_val <= 1
        for cos_val in cos_values:
            assert -1 <= cos_val <= 1

    def test_multiple_years_same_day(self):
        """Test that same day-of-year across different years produces same encoding."""
        df = pl.LazyFrame({"date": ["2023-03-15", "2024-03-15", "2025-03-15"]}).with_columns(
            pl.col("date").str.to_date()
        )

        result = add_cyclical_date_encoding(df).collect()

        sin_values = result["sin_day_of_year"].to_list()

        # Same day of year should produce same encoding (within tolerance for leap years)
        assert abs(sin_values[0] - sin_values[2]) < 1e-10  # 2023, 2025 both non-leap
        # 2024 is leap year, so March 15 is one day offset, but should be close
        assert abs(sin_values[0] - sin_values[1]) < 0.02  # Small difference due to leap year

    def test_preserves_original_columns(self):
        """Test that original columns are preserved."""
        df = pl.LazyFrame(
            {"date": ["2024-01-01", "2024-06-15"], "value": [10.0, 20.0], "category": ["A", "B"]}
        ).with_columns(pl.col("date").str.to_date())

        result = add_cyclical_date_encoding(df).collect()

        # Original columns should be preserved
        assert result["value"].to_list() == [10.0, 20.0]
        assert result["category"].to_list() == ["A", "B"]
        from datetime import date

        result_dates = result["date"].to_list()
        expected_dates = [date(2024, 1, 1), date(2024, 6, 15)]
        assert result_dates == expected_dates

    def test_column_order_preservation(self):
        """Test that new columns are added at the end."""
        df = pl.LazyFrame({"first": [1, 2], "date": ["2024-01-01", "2024-06-15"], "last": [10, 20]}).with_columns(
            pl.col("date").str.to_date()
        )

        result = add_cyclical_date_encoding(df).collect()

        expected_columns = ["first", "date", "last", "sin_day_of_year", "cos_day_of_year"]
        assert list(result.columns) == expected_columns

    def test_lazy_evaluation_maintained(self):
        """Test that LazyFrame remains lazy during operation."""
        n_rows = 1000
        from datetime import date, timedelta

        start_date = date(2024, 1, 1)
        dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_rows)]
        df = pl.LazyFrame({"date": dates, "value": list(range(n_rows))}).with_columns(pl.col("date").str.to_date())

        result = add_cyclical_date_encoding(df)

        # Should return LazyFrame without collecting
        assert isinstance(result, pl.LazyFrame)

        # When collected, should have correct structure
        collected = result.collect()
        assert collected.shape == (n_rows, 4)  # original 2 + cyclical 2
        assert "sin_day_of_year" in collected.columns
        assert "cos_day_of_year" in collected.columns

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pl.LazyFrame({"date": []}, schema={"date": pl.Date})

        result = add_cyclical_date_encoding(df).collect()

        assert result.shape[0] == 0
        assert "sin_day_of_year" in result.columns
        assert "cos_day_of_year" in result.columns
        assert result.schema["sin_day_of_year"] == pl.Float64
        assert result.schema["cos_day_of_year"] == pl.Float64

    def test_data_types_preserved(self):
        """Test that original data types are preserved."""
        df = pl.LazyFrame(
            {
                "date": ["2024-01-01"],
                "int_col": pl.Series([42], dtype=pl.Int32),
                "float_col": pl.Series([3.14], dtype=pl.Float32),
            }
        ).with_columns(pl.col("date").str.to_date())

        original_schema = df.collect_schema()
        result = add_cyclical_date_encoding(df)
        result_schema = result.collect_schema()

        # Original columns should keep their types
        assert result_schema["date"] == original_schema["date"]
        assert result_schema["int_col"] == original_schema["int_col"]
        assert result_schema["float_col"] == original_schema["float_col"]

        # New columns should be Float64
        assert result_schema["sin_day_of_year"] == pl.Float64
        assert result_schema["cos_day_of_year"] == pl.Float64

    def test_column_not_found_error(self):
        """Test error when specified date column doesn't exist."""
        df = pl.LazyFrame({"other_col": [1, 2, 3]})

        with pytest.raises(ValueError, match="Column 'date' not found"):
            add_cyclical_date_encoding(df).collect()

    def test_custom_column_not_found_error(self):
        """Test error when custom date column doesn't exist."""
        df = pl.LazyFrame({"date": ["2024-01-01"]}).with_columns(pl.col("date").str.to_date())

        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            add_cyclical_date_encoding(df, date_column="nonexistent").collect()

    def test_invalid_column_type_error(self):
        """Test error when column is not Date or Datetime type."""
        df = pl.LazyFrame({"date": [1, 2, 3]})  # Integer column

        with pytest.raises(ValueError, match="Column 'date' must be Date or Datetime type"):
            add_cyclical_date_encoding(df).collect()

    def test_string_column_type_error(self):
        """Test error when column is string type."""
        df = pl.LazyFrame({"date": ["2024-01-01", "2024-06-15"]})

        with pytest.raises(ValueError, match="Column 'date' must be Date or Datetime type"):
            add_cyclical_date_encoding(df).collect()

    def test_float_column_type_error(self):
        """Test error when column is float type."""
        df = pl.LazyFrame({"date": [1.5, 2.5, 3.5]})

        with pytest.raises(ValueError, match="Column 'date' must be Date or Datetime type"):
            add_cyclical_date_encoding(df).collect()

    def test_mathematical_properties(self):
        """Test mathematical properties of cyclical encoding."""
        # Test full year cycle
        from datetime import date, timedelta

        start_date = date(2024, 1, 1)
        dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(0, 366, 30)]  # Every 30 days
        df = pl.LazyFrame({"date": dates}).with_columns(pl.col("date").str.to_date())

        result = add_cyclical_date_encoding(df).collect()

        sin_values = result["sin_day_of_year"].to_list()
        cos_values = result["cos_day_of_year"].to_list()

        # Check unit circle property for all values
        for sin_val, cos_val in zip(sin_values, cos_values, strict=False):
            assert abs(sin_val**2 + cos_val**2 - 1) < 1e-10

        # Check values are distributed around the circle
        assert min(sin_values) < -0.5  # Should have negative values
        assert max(sin_values) > 0.5  # Should have positive values
        assert min(cos_values) < -0.5  # Should have negative values
        assert max(cos_values) > 0.5  # Should have positive values

    def test_year_boundary_continuity(self):
        """Test that encoding is continuous across year boundaries."""
        df = pl.LazyFrame({"date": ["2023-12-31", "2024-01-01"]}).with_columns(pl.col("date").str.to_date())

        result = add_cyclical_date_encoding(df).collect()

        sin_vals = result["sin_day_of_year"].to_list()
        cos_vals = result["cos_day_of_year"].to_list()

        # Dec 31 (day 365) should be close to Jan 1 (day 1) in the encoding
        # Both should be close to (sin≈0, cos≈1)
        assert abs(sin_vals[0]) < 0.1  # Dec 31
        assert abs(sin_vals[1]) < 0.1  # Jan 1
        assert cos_vals[0] > 0.9  # Dec 31
        assert cos_vals[1] > 0.9  # Jan 1

    def test_comprehensive_scenario(self):
        """Comprehensive test with multiple date patterns."""
        df = pl.LazyFrame(
            {
                "date": [
                    "2024-01-01",  # New Year
                    "2024-04-01",  # Q2 start
                    "2024-07-01",  # Mid year
                    "2024-10-01",  # Q4 start
                    "2024-12-31",  # Year end
                ],
                "sensor_id": ["A", "B", "C", "D", "E"],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        ).with_columns(pl.col("date").str.to_date())

        result = add_cyclical_date_encoding(df).collect()

        # Check structure
        expected_columns = ["date", "sensor_id", "value", "sin_day_of_year", "cos_day_of_year"]
        assert list(result.columns) == expected_columns

        # Check preserved data
        assert result["sensor_id"].to_list() == ["A", "B", "C", "D", "E"]
        assert result["value"].to_list() == [10.0, 20.0, 30.0, 40.0, 50.0]

        # Check cyclical properties
        sin_vals = result["sin_day_of_year"].to_list()
        cos_vals = result["cos_day_of_year"].to_list()

        # Jan 1 and Dec 31 should be similar (close to cos=1, sin=0)
        assert abs(sin_vals[0] - sin_vals[4]) < 0.1
        assert abs(cos_vals[0] - cos_vals[4]) < 0.1

        # Mid-year (July) should be opposite to year boundaries
        assert cos_vals[2] < 0  # Should be negative
