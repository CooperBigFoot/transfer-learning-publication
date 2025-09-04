import warnings
from datetime import date, datetime

import polars as pl
import pytest

from transfer_learning_publication.cleaners.temporal_consistency import ensure_temporal_consistency


class TestEnsureTemporalConsistency:
    def test_basic_functionality_with_date(self):
        """Test basic sorting and null removal with Date column."""
        lf = pl.LazyFrame({"date": [date(2023, 3, 1), date(2023, 1, 1), date(2023, 2, 1)], "value": [30, 10, 20]})

        result = ensure_temporal_consistency(lf, "date")
        collected = result.collect()

        expected = pl.DataFrame({"date": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)], "value": [10, 20, 30]})

        assert collected.equals(expected)

    def test_basic_functionality_with_datetime(self):
        """Test basic sorting and null removal with Datetime column."""
        lf = pl.LazyFrame(
            {
                "timestamp": [datetime(2023, 3, 1, 12, 0), datetime(2023, 1, 1, 8, 0), datetime(2023, 2, 1, 16, 0)],
                "value": [30, 10, 20],
            }
        )

        result = ensure_temporal_consistency(lf, "timestamp")
        collected = result.collect()

        expected = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1, 8, 0), datetime(2023, 2, 1, 16, 0), datetime(2023, 3, 1, 12, 0)],
                "value": [10, 20, 30],
            }
        )

        assert collected.equals(expected)

    def test_removes_null_dates(self):
        """Test that null dates are properly removed."""
        lf = pl.LazyFrame({"date": [date(2023, 1, 1), None, date(2023, 2, 1)], "value": [10, 99, 20]})

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Dropped .* rows .* due to null dates")
            result = ensure_temporal_consistency(lf, "date")
            collected = result.collect()

        expected = pl.DataFrame({"date": [date(2023, 1, 1), date(2023, 2, 1)], "value": [10, 20]})

        assert collected.equals(expected)

    def test_warns_on_significant_data_loss(self):
        """Test warning when >5% of data is dropped due to null dates."""
        lf = pl.LazyFrame(
            {"date": [date(2023, 1, 1), None, None, None, None, None, None, None, None, None], "value": list(range(10))}
        )

        with pytest.warns(UserWarning, match="Dropped 9 rows \\(90.0%\\) due to null dates"):
            result = ensure_temporal_consistency(lf, "date")
            result.collect()

    def test_no_warning_on_minimal_data_loss(self):
        """Test no warning when <5% of data is dropped."""
        lf = pl.LazyFrame(
            {
                "date": [date(2023, 1, 1), date(2023, 1, 2)] * 9
                + [date(2023, 1, 3)]
                + [None],  # 5% null, multiple dates
                "value": list(range(20)),
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            result = ensure_temporal_consistency(lf, "date")
            result.collect()  # Should not raise any warning

    def test_warns_on_single_date_collapse(self):
        """Test warning when all data collapses to single date."""
        single_date = date(2023, 1, 1)
        lf = pl.LazyFrame({"date": [single_date, single_date, single_date], "value": [10, 20, 30]})

        with pytest.warns(UserWarning, match=f"All data collapsed to single date: {single_date}"):
            result = ensure_temporal_consistency(lf, "date")
            result.collect()

    def test_default_date_column_parameter(self):
        """Test that default date_column parameter works."""
        lf = pl.LazyFrame({"date": [date(2023, 2, 1), date(2023, 1, 1)], "value": [20, 10]})

        result = ensure_temporal_consistency(lf)  # Using default "date" column
        collected = result.collect()

        expected = pl.DataFrame({"date": [date(2023, 1, 1), date(2023, 2, 1)], "value": [10, 20]})

        assert collected.equals(expected)

    def test_custom_date_column_parameter(self):
        """Test with custom date column name."""
        lf = pl.LazyFrame({"custom_date": [date(2023, 2, 1), date(2023, 1, 1)], "value": [20, 10]})

        result = ensure_temporal_consistency(lf, "custom_date")
        collected = result.collect()

        expected = pl.DataFrame({"custom_date": [date(2023, 1, 1), date(2023, 2, 1)], "value": [10, 20]})

        assert collected.equals(expected)

    def test_preserves_column_order(self):
        """Test that column order is preserved."""
        lf = pl.LazyFrame(
            {
                "value": [20, 10, 30],
                "date": [date(2023, 2, 1), date(2023, 1, 1), date(2023, 3, 1)],
                "other": ["b", "a", "c"],
            }
        )

        result = ensure_temporal_consistency(lf, "date")
        collected = result.collect()

        # Check column order
        assert collected.columns == ["value", "date", "other"]

        # Check sorting worked
        assert collected["date"].to_list() == [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)]
        assert collected["value"].to_list() == [10, 20, 30]
        assert collected["other"].to_list() == ["a", "b", "c"]

    def test_returns_lazy_frame(self):
        """Test that function returns LazyFrame, not DataFrame."""
        lf = pl.LazyFrame({"date": [date(2023, 1, 1), date(2023, 1, 2)], "value": [10, 20]})

        result = ensure_temporal_consistency(lf, "date")
        assert isinstance(result, pl.LazyFrame)

    def test_error_nonexistent_column(self):
        """Test error when date column doesn't exist."""
        lf = pl.LazyFrame({"value": [10, 20, 30]})

        with pytest.raises(ValueError, match="Column 'nonexistent' not found in LazyFrame"):
            ensure_temporal_consistency(lf, "nonexistent")

    def test_error_invalid_column_type_string(self):
        """Test error when date column is string type."""
        lf = pl.LazyFrame({"date": ["2023-01-01", "2023-02-01"], "value": [10, 20]})

        with pytest.raises(ValueError, match="Column 'date' must be Date or Datetime type. Got: String"):
            ensure_temporal_consistency(lf, "date")

    def test_error_invalid_column_type_integer(self):
        """Test error when date column is integer type."""
        lf = pl.LazyFrame({"date": [20230101, 20230201], "value": [10, 20]})

        with pytest.raises(ValueError, match="Column 'date' must be Date or Datetime type. Got: Int64"):
            ensure_temporal_consistency(lf, "date")

    def test_error_all_dates_null(self):
        """Test error when all dates are null."""
        lf = pl.LazyFrame(
            {"date": [None, None, None], "value": [10, 20, 30]}, schema={"date": pl.Date, "value": pl.Int64}
        )

        with pytest.raises(
            ValueError,
            match="All values in column 'date' are null. Cannot establish temporal consistency without valid dates.",
        ):
            ensure_temporal_consistency(lf, "date")

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        lf = pl.LazyFrame(schema={"date": pl.Date, "value": pl.Int64})

        with pytest.raises(ValueError, match="All values in column 'date' are null"):
            ensure_temporal_consistency(lf, "date")

    def test_mixed_datetime_precision(self):
        """Test with mixed datetime precision."""
        lf = pl.LazyFrame(
            {
                "timestamp": [
                    datetime(2023, 1, 1, 12, 30, 45),
                    datetime(2023, 1, 1, 8, 15, 30),
                    datetime(2023, 1, 1, 16, 45, 0),
                ],
                "value": [30, 10, 40],
            }
        )

        result = ensure_temporal_consistency(lf, "timestamp")
        collected = result.collect()

        # Should be sorted by timestamp
        expected_times = [
            datetime(2023, 1, 1, 8, 15, 30),
            datetime(2023, 1, 1, 12, 30, 45),
            datetime(2023, 1, 1, 16, 45, 0),
        ]
        assert collected["timestamp"].to_list() == expected_times
        assert collected["value"].to_list() == [10, 30, 40]

    def test_duplicate_dates_preserved(self):
        """Test that duplicate dates are preserved and sorted correctly."""
        duplicate_date = date(2023, 1, 1)
        lf = pl.LazyFrame(
            {
                "date": [date(2023, 2, 1), duplicate_date, duplicate_date, date(2023, 1, 2)],
                "value": [200, 100, 101, 120],
            }
        )

        result = ensure_temporal_consistency(lf, "date")
        collected = result.collect()

        # Should be sorted, duplicates preserved
        expected_dates = [duplicate_date, duplicate_date, date(2023, 1, 2), date(2023, 2, 1)]
        assert collected["date"].to_list() == expected_dates
        assert collected["value"].to_list() == [100, 101, 120, 200]
