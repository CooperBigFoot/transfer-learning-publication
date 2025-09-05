import warnings

import polars as pl


def ensure_temporal_consistency(
    lf: pl.LazyFrame,
    date_column: str = "date",
    fill_missing_dates: bool = False,
) -> pl.LazyFrame:
    """
    Ensures temporal consistency by validating, filtering, sorting, and optionally filling missing dates.

    This function should be called first in the pipeline to establish a clean temporal foundation.
    It drops rows with null/NaN dates, sorts the data chronologically, and can fill in missing
    dates in the sequence to ensure continuous temporal coverage.

    Args:
        lf: The input Polars LazyFrame.
        date_column: Name of the date/datetime column to use. Defaults to "date".
        fill_missing_dates: Whether to fill missing dates in the sequence with NaN values
                           for all other columns. Defaults to False.

    Returns:
        A new LazyFrame with:
        - All rows with null dates removed
        - Data sorted chronologically by the date column
        - Missing dates filled with NaN values (if fill_missing_dates=True)

    Raises:
        ValueError: If date_column doesn't exist, isn't a Date/Datetime type,
                   or if all dates are null.
    """
    schema = lf.collect_schema()

    # Validate column exists
    if date_column not in schema.names():
        raise ValueError(f"Column '{date_column}' not found in LazyFrame")

    # Validate column type - accept Date or Datetime only
    if schema[date_column] not in (pl.Date, pl.Datetime):
        raise ValueError(f"Column '{date_column}' must be Date or Datetime type. Got: {schema[date_column]}")

    # Count total rows before filtering
    total_rows = lf.select(pl.len()).collect().item()

    # Filter out null dates
    filtered_lf = lf.filter(pl.col(date_column).is_not_null())

    # Count rows after filtering
    remaining_rows = filtered_lf.select(pl.len()).collect().item()

    # Check if all dates were null
    if remaining_rows == 0:
        raise ValueError(
            f"All values in column '{date_column}' are null. Cannot establish temporal consistency without valid dates."
        )

    # Warn about significant data loss
    dropped_rows = total_rows - remaining_rows
    if dropped_rows > 0:
        drop_pct = (dropped_rows / total_rows) * 100
        if drop_pct > 5:
            warnings.warn(f"Dropped {dropped_rows} rows ({drop_pct:.1f}%) due to null dates.", stacklevel=2)

    # Warn if collapsed to single date
    unique_dates = filtered_lf.select(pl.col(date_column).n_unique()).collect().item()
    if unique_dates == 1:
        single_date = filtered_lf.select(pl.col(date_column).first()).collect().item()
        warnings.warn(
            f"All data collapsed to single date: {single_date}. This may affect time-based splits and validations.",
            stacklevel=2,
        )

    # Sort chronologically
    sorted_lf = filtered_lf.sort(date_column)

    # Fill missing dates if requested
    if fill_missing_dates and remaining_rows > 1:
        # Get date range
        date_stats = sorted_lf.select(
            [pl.col(date_column).min().alias("min_date"), pl.col(date_column).max().alias("max_date")]
        ).collect()

        min_date = date_stats["min_date"][0]
        max_date = date_stats["max_date"][0]

        if min_date != max_date:  # Only if we have a date range
            # Create complete daily date range - cast to date first to ensure consistent behavior
            if schema[date_column] == pl.Datetime:
                # For datetime columns, convert to date for range generation, then back to datetime
                min_date_only = min_date.date() if hasattr(min_date, "date") else min_date
                max_date_only = max_date.date() if hasattr(max_date, "date") else max_date
                date_range = pl.date_range(min_date_only, max_date_only, interval="1d", eager=True)
                # Cast back to datetime (will be at midnight)
                date_range = date_range.cast(pl.Datetime)
            else:
                # For date columns, use as-is
                date_range = pl.date_range(min_date, max_date, interval="1d", eager=True)

            complete_dates = date_range.to_frame(date_column).lazy()

            # Count gaps before filling
            original_date_count = unique_dates
            expected_date_count = complete_dates.select(pl.len()).collect().item()
            missing_dates = expected_date_count - original_date_count

            # Warn about significant gaps
            if missing_dates > 0:
                gap_pct = (missing_dates / expected_date_count) * 100
                if gap_pct > 10:  # Warn if >10% of dates are missing
                    warnings.warn(
                        f"Filled {missing_dates} missing dates ({gap_pct:.1f}%) to ensure daily continuity.",
                        stacklevel=2,
                    )

            # Left join to fill missing dates with NaN
            sorted_lf = complete_dates.join(sorted_lf, on=date_column, how="left")

    return sorted_lf
