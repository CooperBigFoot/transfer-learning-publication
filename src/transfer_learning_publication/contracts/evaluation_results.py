"""EvaluationResults contract for model testing and evaluation."""

import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .forecast_output import ForecastOutput

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation results from multiple models.

    Transforms ForecastOutput objects into a standardized long-format DataFrame
    for analysis and comparison across models.

    The DataFrame contains columns:
    - model_name: Name of the model
    - group_identifier: Basin/gauge ID
    - issue_date: When the forecast was issued (if available)
    - prediction_date: Date being predicted (if available)
    - lead_time: Forecast horizon (1 to output_len)
    - prediction: Model prediction value
    - observation: Ground truth value

    Attributes:
        results_dict: Dictionary mapping model names to ForecastOutput objects
        output_length: Length of the forecast horizon
        include_dates: Whether date columns are included
    """

    results_dict: dict[str, ForecastOutput]
    output_length: int
    include_dates: bool = False

    def __post_init__(self):
        """Validate results consistency."""
        if not self.results_dict:
            raise ValueError("results_dict cannot be empty")

        # Validate all ForecastOutputs have same output length
        output_lengths = set()
        for name, output in self.results_dict.items():
            if not isinstance(output, ForecastOutput):
                raise TypeError(f"Expected ForecastOutput for model '{name}', got {type(output)}")
            output_lengths.add(output.predictions.shape[1])

        if len(output_lengths) > 1:
            raise ValueError(f"Inconsistent output lengths across models: {output_lengths}")

        # Set output length if not provided
        if self.output_length is None:
            self.output_length = next(iter(output_lengths))

        # Check if dates are available
        first_output = next(iter(self.results_dict.values()))
        self.include_dates = first_output.input_end_dates is not None

    @property
    def raw_data(self) -> pl.DataFrame:
        """Get full DataFrame with all predictions.

        Returns:
            Long-format DataFrame with all model results
        """
        frames = []

        for model_name, forecast_output in self.results_dict.items():
            df = self._forecast_output_to_dataframe(model_name, forecast_output)
            frames.append(df)

        if not frames:
            return pl.DataFrame()

        return pl.concat(frames)

    def _forecast_output_to_dataframe(self, model_name: str, output: ForecastOutput) -> pl.DataFrame:
        """Convert a single ForecastOutput to long-format DataFrame.

        Args:
            model_name: Name of the model
            output: ForecastOutput object

        Returns:
            Long-format DataFrame
        """
        n_samples = output.predictions.shape[0]
        output_len = output.predictions.shape[1]

        # Prepare data lists
        data_dict: dict[str, list] = {
            "model_name": [],
            "group_identifier": [],
            "lead_time": [],
            "prediction": [],
            "observation": [],
        }

        if self.include_dates:
            data_dict["issue_date"] = []
            data_dict["prediction_date"] = []

        # Convert tensors to numpy for efficiency
        predictions_np = output.predictions.cpu().numpy()
        observations_np = output.observations.cpu().numpy()

        if self.include_dates and output.input_end_dates is not None:
            # Convert timestamps to datetime objects
            input_end_dates_np = output.input_end_dates.cpu().numpy()
            # Timestamps are in milliseconds since epoch - convert to microseconds for polars
            base_dates_us = (input_end_dates_np * 1000).astype("int64")
        else:
            base_dates_us = None

        # Build long-format data
        for i in range(n_samples):
            group_id = output.group_identifiers[i]

            for t in range(output_len):
                data_dict["model_name"].append(model_name)
                data_dict["group_identifier"].append(group_id)
                data_dict["lead_time"].append(t + 1)  # 1-indexed
                data_dict["prediction"].append(float(predictions_np[i, t]))
                data_dict["observation"].append(float(observations_np[i, t]))

                if self.include_dates and base_dates_us is not None:
                    # Store as microseconds for polars datetime
                    issue_date_us = base_dates_us[i]
                    # Add days converted to microseconds (1 day = 86400000000 microseconds)
                    prediction_date_us = issue_date_us + (t + 1) * 86400000000
                    data_dict["issue_date"].append(issue_date_us)
                    data_dict["prediction_date"].append(prediction_date_us)

        # Create DataFrame with proper datetime columns
        df = pl.DataFrame(data_dict)
        if self.include_dates:
            # Convert microsecond columns to datetime
            df = df.with_columns(
                [
                    pl.col("issue_date").cast(pl.Datetime(time_unit="us")),
                    pl.col("prediction_date").cast(pl.Datetime(time_unit="us")),
                ]
            )
        return df

    def filter(
        self,
        model_name: str | list[str] | None = None,
        basin_id: str | list[str] | None = None,
        lead_time: int | list[int] | None = None,
    ) -> pl.DataFrame:
        """Filter results by any combination of criteria.

        Args:
            model_name: Single model name or list of model names to filter
            basin_id: Single basin ID or list of basin IDs to filter
            lead_time: Single lead time or list of lead times to filter

        Returns:
            Filtered DataFrame with matching results

        Examples:
            # Single dimension (equivalent to by_model)
            df = results.filter(model_name="tide")

            # Multiple dimensions
            df = results.filter(model_name="tide", lead_time=1)
            df = results.filter(model_name=["tide", "ealstm"], basin_id="01234567")

            # All three dimensions
            df = results.filter(
                model_name="tide",
                basin_id=["01234567", "98765432"],
                lead_time=[1, 2, 3]
            )
        """
        df = self.raw_data

        # Filter by model_name
        if model_name is not None:
            if isinstance(model_name, str):
                df = df.filter(pl.col("model_name") == model_name)
            else:
                df = df.filter(pl.col("model_name").is_in(model_name))

        # Filter by basin_id
        if basin_id is not None:
            if isinstance(basin_id, str):
                df = df.filter(pl.col("group_identifier") == basin_id)
            else:
                df = df.filter(pl.col("group_identifier").is_in(basin_id))

        # Filter by lead_time
        if lead_time is not None:
            if isinstance(lead_time, int):
                # Validate single lead_time
                if lead_time < 1 or lead_time > self.output_length:
                    raise ValueError(f"lead_time must be between 1 and {self.output_length}, got {lead_time}")
                df = df.filter(pl.col("lead_time") == lead_time)
            else:
                # Validate list of lead_times
                for lt in lead_time:
                    if lt < 1 or lt > self.output_length:
                        raise ValueError(f"lead_time must be between 1 and {self.output_length}, got {lt}")
                df = df.filter(pl.col("lead_time").is_in(lead_time))

        return df

    def by_model(self, model_name: str) -> pl.DataFrame:
        """Filter results by model name.

        Args:
            model_name: Name of the model to filter for

        Returns:
            DataFrame with results for specified model

        Raises:
            KeyError: If model_name not found
        """
        if model_name not in self.results_dict:
            available = list(self.results_dict.keys())
            raise KeyError(f"Model '{model_name}' not found. Available models: {available}")

        return self.filter(model_name=model_name)

    def by_basin(self, basin_id: str) -> pl.DataFrame:
        """Filter results by basin/gauge ID.

        Args:
            basin_id: Basin/gauge identifier to filter for

        Returns:
            DataFrame with results for specified basin across all models
        """
        return self.filter(basin_id=basin_id)

    def by_lead_time(self, lead_time: int) -> pl.DataFrame:
        """Filter results by lead time.

        Args:
            lead_time: Lead time to filter for (1 to output_length)

        Returns:
            DataFrame with results for specified lead time across all models

        Raises:
            ValueError: If lead_time is out of range
        """
        return self.filter(lead_time=lead_time)

    def to_parquet(self, output_dir: str | Path, partition_cols: list[str] | None = None) -> None:
        """Export results to parquet with partitioning.

        Args:
            output_dir: Directory to write parquet files
            partition_cols: Columns to partition by (defaults to ["model_name"])
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if partition_cols is None:
            partition_cols = ["model_name"]

        df = self.raw_data

        # For partitioned writes, we need to manually partition with polars
        if partition_cols:
            # Group by partition columns and write each group
            for keys, group_df in df.group_by(partition_cols, maintain_order=True):
                # Build partition path
                if not isinstance(keys, tuple):
                    keys = (keys,)
                partition_path = output_dir
                for col_name, key_val in zip(partition_cols, keys, strict=True):
                    partition_path = partition_path / f"{col_name}={key_val}"
                partition_path.mkdir(parents=True, exist_ok=True)

                # Write the group without partition columns
                group_df.drop(partition_cols).write_parquet(partition_path / "data.parquet", compression="snappy")
        else:
            # Simple write without partitioning
            df.write_parquet(output_dir / "data.parquet", compression="snappy")

        logger.info(f"Wrote evaluation results to {output_dir} partitioned by {partition_cols}")

    def list_models(self) -> list[str]:
        """Get list of model names.

        Returns:
            List of model names in results
        """
        return list(self.results_dict.keys())

    def list_basins(self) -> list[str]:
        """Get unique list of basin/gauge IDs across all models.

        Returns:
            Sorted list of unique basin IDs
        """
        all_basins = set()
        for output in self.results_dict.values():
            all_basins.update(output.group_identifiers)
        return sorted(all_basins)

    def update(self, other: "EvaluationResults") -> None:
        """Update results with additional models.

        Args:
            other: EvaluationResults to merge in

        Raises:
            ValueError: If output_length doesn't match
        """
        if other.output_length != self.output_length:
            raise ValueError(
                f"Cannot merge results with different output_lengths: {self.output_length} vs {other.output_length}"
            )

        # Add new models
        for model_name, forecast_output in other.results_dict.items():
            if model_name in self.results_dict:
                logger.warning(f"Overwriting existing results for '{model_name}'")
            self.results_dict[model_name] = forecast_output

        # Update include_dates flag
        self.include_dates = self.include_dates or other.include_dates

    def summary(self) -> pl.DataFrame:
        """Get summary statistics by model.

        Returns:
            DataFrame with model names and basic statistics
        """
        summaries = []

        for model_name, output in self.results_dict.items():
            n_samples = output.predictions.shape[0]
            n_basins = len(set(output.group_identifiers))

            summaries.append(
                {
                    "model_name": model_name,
                    "n_samples": n_samples,
                    "n_basins": n_basins,
                    "output_length": self.output_length,
                    "has_dates": output.input_end_dates is not None,
                }
            )

        return pl.DataFrame(summaries)
