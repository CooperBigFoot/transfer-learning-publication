import polars as pl


def trim_to_column(lf: pl.LazyFrame, priority_column: str) -> pl.LazyFrame:
    """
    Trims the entire LazyFrame based on the non-null/non-NaN range of a priority column.

    Args:
        lf: The input Polars LazyFrame.
        priority_column: The column name to use for determining trim boundaries.
                        All rows will be trimmed to match this column's non-null/non-NaN range.

    Returns:
        A new LazyFrame trimmed to the priority column's data range.
    """
    if priority_column not in lf.collect_schema().names():
        raise ValueError(f"Column '{priority_column}' not found in LazyFrame")

    lf_indexed = lf.with_row_index("_row_idx")

    # Create a mask that catches both nulls AND NaNs
    # is_null() catches nulls, is_nan() catches NaNs (only for numeric types)
    priority_col_expr = pl.col(priority_column)
    schema = lf.collect_schema()
    priority_dtype = schema[priority_column]

    # Only apply is_nan() to numeric types (Float32, Float64)
    if priority_dtype in (pl.Float32, pl.Float64):
        is_valid = ~(priority_col_expr.is_null() | priority_col_expr.is_nan())
    else:
        is_valid = ~priority_col_expr.is_null()

    # Get the first and last valid (non-null/non-NaN) indices for the priority column
    bounds = lf_indexed.filter(is_valid).select(
        [pl.col("_row_idx").min().alias("start_idx"), pl.col("_row_idx").max().alias("end_idx")]
    )

    # Join and filter to that range, then sort to maintain original order
    result = (
        lf_indexed.join(bounds, how="cross")
        .filter((pl.col("_row_idx") >= pl.col("start_idx")) & (pl.col("_row_idx") <= pl.col("end_idx")))
        .sort("_row_idx")
        .drop(["_row_idx", "start_idx", "end_idx"])
    )

    return result
