import polars as pl


def add_cyclical_date_encoding(lf: pl.LazyFrame, date_column: str = "date") -> pl.LazyFrame:
    """
    Adds cyclical encoding of day-of-year from a date column.

    Creates two new columns with sine and cosine transformations of the day-of-year.

    Args:
        lf: The input Polars LazyFrame.
        date_column: Name of the date/datetime column to encode. Defaults to "date".

    Returns:
        A new LazyFrame with two additional columns:
        - sin_day_of_year: Sine encoding of day-of-year
        - cos_day_of_year: Cosine encoding of day-of-year

    Raises:
        ValueError: If date_column doesn't exist or isn't a date/datetime type.
    """
    schema = lf.collect_schema()

    # Validate column exists
    if date_column not in schema.names():
        raise ValueError(f"Column '{date_column}' not found in LazyFrame")

    try:
        test_expr = pl.col(date_column).dt.ordinal_day()
        lf.select(test_expr).limit(0).collect()
    except Exception:
        raise ValueError(f"Column '{date_column}' must be Date or Datetime type. Got: {schema[date_column]}") from None

    day_of_year_normalized = (pl.col(date_column).dt.ordinal_day() - 1) / 365.25

    sin_expr = (2 * pl.lit(3.141592653589793) * day_of_year_normalized).sin().alias("sin_day_of_year")
    cos_expr = (2 * pl.lit(3.141592653589793) * day_of_year_normalized).cos().alias("cos_day_of_year")

    return lf.with_columns([sin_expr, cos_expr])
