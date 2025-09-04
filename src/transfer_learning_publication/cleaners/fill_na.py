import warnings

import polars as pl


def fill_na_columns(
    lf: pl.LazyFrame,
    columns: str | list[str],
    fill_value: float | int,
    add_binary_flag: bool = False,
) -> pl.LazyFrame:
    """
    Fills null and NaN values in specified numeric columns with a given value.

    Args:
        lf: The input Polars LazyFrame.
        columns: Single column name or list of column names to fill.
        fill_value: The value to use for filling nulls and NaNs.
        add_binary_flag: If True, adds a binary flag column for each filled column
                        indicating which values were filled (1) or original (0).

    Returns:
        A new LazyFrame with nulls/NaNs filled in specified columns.
        If add_binary_flag is True, adds {column}_was_filled columns at the end.
    """
    if isinstance(columns, str):
        columns = [columns]

    if not columns:
        warnings.warn("No columns provided. Returning LazyFrame unchanged.", stacklevel=2)
        return lf

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

        # Allow Null type (inferred when all values are null) alongside numeric types
        if schema[col] not in numeric_types and schema[col] != pl.Null:
            raise ValueError(f"Column '{col}' is not numeric. Type: {schema[col]}")

    # Build expressions for filling and flag columns
    fill_exprs = []
    flag_exprs = []

    for col in columns:
        col_expr = pl.col(col)
        dtype = schema[col]

        # Create mask for nulls and NaNs (similar to trim_to_column logic)
        is_missing = col_expr.is_null() | col_expr.is_nan() if dtype in (pl.Float32, pl.Float64) else col_expr.is_null()

        # Fill expression
        filled = pl.when(is_missing).then(fill_value).otherwise(col_expr).alias(col)
        fill_exprs.append(filled)

        # Flag expression (if requested)
        if add_binary_flag:
            flag_name = f"{col}_was_filled"
            flag = is_missing.cast(pl.UInt8).alias(flag_name)
            flag_exprs.append(flag)

    # Apply transformations - calculate flags and fills together to use original data
    result = lf.with_columns(fill_exprs + flag_exprs) if add_binary_flag else lf.with_columns(fill_exprs)

    return result
