import warnings

import polars as pl


def train_val_test_split(
    lf: pl.LazyFrame,
    train_prop: float,
    val_prop: float,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """
    Splits a LazyFrame sequentially into train, validation, and test sets.

    This function performs a sequential (non-random) split, preserving the order of rows.

    Args:
        lf: The input Polars LazyFrame.
        train_prop: Proportion of data for training set (0 to 1).
        val_prop: Proportion of data for validation set (0 to 1).
                  Test proportion is inferred as 1 - (train_prop + val_prop).

    Returns:
        A tuple of three LazyFrames: (train_lf, val_lf, test_lf)

    Raises:
        ValueError: If proportions are negative, or if their sum exceeds 1.0.
    """
    if train_prop < 0 or val_prop < 0:
        raise ValueError(f"Proportions must be non-negative. Got train_prop={train_prop}, val_prop={val_prop}")

    if train_prop > 1 or val_prop > 1:
        raise ValueError(f"Proportions must be <= 1. Got train_prop={train_prop}, val_prop={val_prop}")

    total_prop = train_prop + val_prop
    if total_prop > 1.0:
        raise ValueError(f"Sum of proportions ({total_prop}) exceeds 1.0")

    test_prop = 1.0 - total_prop

    # Warn about edge cases
    if train_prop == 1.0:
        warnings.warn("train_prop is 1.0. Creating empty validation and test sets.", stacklevel=2)
    elif total_prop == 1.0:
        warnings.warn("No test set created (train_prop + val_prop = 1.0).", stacklevel=2)
    elif val_prop == 0 and test_prop > 0:
        warnings.warn("No validation set created (val_prop = 0).", stacklevel=2)

    total_rows = lf.select(pl.len()).collect().item()

    if total_rows == 0:
        warnings.warn("Input LazyFrame is empty. Returning three empty LazyFrames.", stacklevel=2)
        return lf, lf, lf

    train_size = int(total_rows * train_prop)
    val_size = int(total_rows * val_prop)
    test_size = total_rows - train_size - val_size

    train_lf = lf.slice(0, train_size)
    val_lf = lf.slice(train_size, val_size)
    test_lf = lf.slice(train_size + val_size, test_size)

    return train_lf, val_lf, test_lf
