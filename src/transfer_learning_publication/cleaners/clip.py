import warnings

import polars as pl


def clip_columns(
    lf: pl.LazyFrame,
    columns: str | list[str],
    min_value: float | None = None,
    max_value: float | None = None,
) -> pl.LazyFrame:
    """
    Clips values in specified columns to a given range.

    Args:
        lf: The input Polars LazyFrame.
        columns: Single column name or list of column names to clip.
        min_value: Optional minimum value. Values below this will be clipped to this value.
        max_value: Optional maximum value. Values above this will be clipped to this value.

    Returns:
        A new LazyFrame with clipping applied to specified columns.
        Nulls and NaNs are preserved unchanged.
    """
    # Check if at least one bound is provided
    if min_value is None and max_value is None:
        warnings.warn("No min or max value provided. Returning LazyFrame unchanged.", stacklevel=2)
        return lf

    # Normalize columns to list
    if isinstance(columns, str):
        columns = [columns]

    # Validate columns exist and are numeric
    schema = lf.collect_schema()
    numeric_types = (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    )

    for col in columns:
        if col not in schema.names():
            raise ValueError(f"Column '{col}' not found in LazyFrame")

        if schema[col] not in numeric_types:
            raise ValueError(f"Column '{col}' is not numeric. Type: {schema[col]}")

    # Apply clipping to each column
    clip_exprs = [pl.col(col).clip(lower_bound=min_value, upper_bound=max_value).alias(col) for col in columns]

    return lf.with_columns(clip_exprs)
