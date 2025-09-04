import warnings

import polars as pl


def ensure_temporal_consistency(
    lf: pl.LazyFrame,
    date_column: str = "date",
) -> pl.LazyFrame:
    """
    Ensures temporal consistency by validating, filtering, and sorting by date.

    This function should be called first in the pipeline to establish a clean temporal foundation.
    It drops rows with null/NaN dates and sorts the data chronologically.

    Args:
        lf: The input Polars LazyFrame.
        date_column: Name of the date/datetime column to use. Defaults to "date".

    Returns:
        A new LazyFrame with:
        - All rows with null dates removed
        - Data sorted chronologically by the date column

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

    return sorted_lf
