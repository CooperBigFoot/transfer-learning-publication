import polars as pl

from .clip import clip_columns
from .cyclical_date_encoding import add_cyclical_date_encoding
from .fill_na import fill_na_columns
from .temporal_consistency import ensure_temporal_consistency
from .trim_to_col import trim_to_column


class GaugeCleaner:
    """
    A builder-pattern class for orchestrating data cleaning operations on gauge time series.

    This class allows you to define a sequence of cleaning steps that can be applied
    to individual gauge LazyFrames. Each method adds a cleaning step to the pipeline
    and returns self for method chaining.

    Example:
        >>> cleaner = (GaugeCleaner()
        ...            .ensure_temporal_consistency()
        ...            .trim_to_column("streamflow")
        ...            .fill_na_columns(["precipitation", "temperature"], 0.0)
        ...            .clip_columns(["streamflow"], min_value=0.0)
        ...            .add_cyclical_date_encoding())
        >>> cleaned_lf = cleaner.apply(raw_lf)
    """

    def __init__(self):
        """Initialize with empty step list."""
        self.steps: list[tuple[callable, dict]] = []

    def ensure_temporal_consistency(self, date_column: str = "date") -> "GaugeCleaner":
        """
        Add temporal consistency check step.

        Ensures data is temporally valid, removes null dates, and sorts chronologically.

        Args:
            date_column: Name of the date/datetime column. Defaults to "date".

        Returns:
            Self for method chaining.
        """
        self.steps.append((ensure_temporal_consistency, {"date_column": date_column}))
        return self

    def trim_to_column(self, priority_column: str) -> "GaugeCleaner":
        """
        Add trimming step based on priority column.

        Trims the entire LazyFrame to the non-null/non-NaN range of the priority column.

        Args:
            priority_column: Column name to use for determining trim boundaries.

        Returns:
            Self for method chaining.
        """
        self.steps.append((trim_to_column, {"priority_column": priority_column}))
        return self

    def fill_na_columns(
        self,
        columns: list[str],
        fill_value: float,
        add_binary_flag: bool = False,
    ) -> "GaugeCleaner":
        """
        Add NA filling step.

        Fills null and NaN values in specified columns with the given value.

        Args:
            columns: List of column names to fill.
            fill_value: The value to use for filling nulls and NaNs.
            add_binary_flag: If True, adds binary flag columns indicating which values were filled.

        Returns:
            Self for method chaining.
        """
        self.steps.append(
            (
                fill_na_columns,
                {
                    "columns": columns,
                    "fill_value": fill_value,
                    "add_binary_flag": add_binary_flag,
                },
            )
        )
        return self

    def clip_columns(
        self,
        columns: list[str],
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> "GaugeCleaner":
        """
        Add clipping step.

        Clips values in specified columns to the given range.

        Args:
            columns: List of column names to clip.
            min_value: Optional minimum value for clipping.
            max_value: Optional maximum value for clipping.

        Returns:
            Self for method chaining.
        """
        self.steps.append(
            (
                clip_columns,
                {
                    "columns": columns,
                    "min_value": min_value,
                    "max_value": max_value,
                },
            )
        )
        return self

    def add_cyclical_date_encoding(self, date_column: str = "date") -> "GaugeCleaner":
        """
        Add cyclical encoding step.

        Adds sine and cosine features for day-of-year seasonality.

        Args:
            date_column: Name of the date/datetime column. Defaults to "date".

        Returns:
            Self for method chaining.
        """
        self.steps.append((add_cyclical_date_encoding, {"date_column": date_column}))
        return self

    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Execute all steps in order on the LazyFrame.

        Applies each cleaning step sequentially, with each step's output becoming
        the next step's input.

        Args:
            lf: The input Polars LazyFrame to clean.

        Returns:
            A new LazyFrame with all cleaning steps applied.

        Raises:
            Any exceptions raised by the individual cleaning functions will propagate up.
        """
        result = lf
        for func, kwargs in self.steps:
            result = func(result, **kwargs)
        return result
