from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from .base import BaseTransform
from .pipeline import GlobalPipeline, PerBasinPipeline


@dataclass
class CompositePipelineStep:
    """A single step in a composite pipeline.

    Attributes:
        pipeline_type: Either "per_basin" or "global"
        transforms: List of transforms to apply in this step
        columns: List of column names to transform in this step
    """

    pipeline_type: Literal["per_basin", "global"]
    transforms: list[BaseTransform]
    columns: list[str]


class CompositePipeline:
    """Pipeline that applies sequential transformation steps to Polars DataFrames.

    Each step can target specific columns and use either per-basin or global
    transformation strategies. Original DataFrame columns are preserved - only
    specified columns are transformed at each step.
    """

    def __init__(self, steps: list[CompositePipelineStep], group_identifier: str):
        """
        Args:
            steps: List of pipeline steps to execute in sequence
            group_identifier: Column name containing group identifiers (e.g., "basin_id")
        """
        self.steps = steps
        self.group_identifier = group_identifier
        self._fitted_steps: list[PerBasinPipeline | GlobalPipeline] = []
        self._is_fitted = False
        self._group_mapping = {}  # Maps numeric indices back to original group values

    def __repr__(self) -> str:
        return f"CompositePipeline(steps={self.steps}, group_identifier='{self.group_identifier}')"

    def _validate_dataframe(self, df: pl.DataFrame) -> None:
        """Validate input DataFrame format."""
        if not isinstance(df, pl.DataFrame):
            raise TypeError("Input must be a Polars DataFrame")

        if self.group_identifier not in df.columns:
            raise ValueError(f"Group identifier '{self.group_identifier}' not found in DataFrame columns")

        if df.shape[0] == 0:
            raise ValueError("DataFrame cannot be empty")

    def _validate_step_columns(self, df: pl.DataFrame, step: CompositePipelineStep) -> None:
        """Validate that all step columns exist in the DataFrame."""
        missing_cols = [col for col in step.columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame")

    def _dataframe_to_numpy(self, df: pl.DataFrame, columns: list[str]) -> np.ndarray:
        """Convert Polars DataFrame to numpy array for pipeline processing.

        Args:
            df: Input DataFrame
            columns: Feature columns to include (group_identifier added automatically)

        Returns:
            2D numpy array with features as first columns, group_identifier as last column (encoded as numeric)
        """
        # Select feature columns and fill nulls with NaN for numeric handling
        features_df = df.select(columns).fill_null(float('nan'))
        features_array = features_df.to_numpy()
        
        # Get group identifiers
        group_values = df[self.group_identifier].to_numpy()
        
        # Encode string group identifiers as numeric values
        unique_groups = np.unique(group_values)
        
        # Create or update group mapping
        if not self._group_mapping:
            # First time: create mapping
            self._group_mapping = {i: group for i, group in enumerate(unique_groups)}
        else:
            # Subsequent calls: ensure all groups are in mapping
            existing_groups = set(self._group_mapping.values())
            new_groups = set(unique_groups) - existing_groups
            if new_groups:
                next_idx = max(self._group_mapping.keys()) + 1 if self._group_mapping else 0
                for group in new_groups:
                    self._group_mapping[next_idx] = group
                    next_idx += 1
        
        # Create reverse mapping for encoding
        group_to_idx = {v: k for k, v in self._group_mapping.items()}
        
        # Encode groups as numeric values
        numeric_groups = np.array([float(group_to_idx[g]) for g in group_values]).reshape(-1, 1)
        
        # Combine features with numeric group identifiers
        array_data = np.hstack([features_array, numeric_groups])
        
        return array_data

    def _numpy_to_dataframe_update(
        self, df: pl.DataFrame, transformed_array: np.ndarray, columns: list[str]
    ) -> pl.DataFrame:
        """Update DataFrame with transformed values from numpy array.

        Args:
            df: Original DataFrame to update
            transformed_array: Numpy array with transformed features (group_id as last column)
            columns: Column names that were transformed

        Returns:
            Updated DataFrame with transformed columns
        """
        # Extract transformed features (exclude group identifier column)
        transformed_features = transformed_array[:, :-1]

        # Create mapping of transformed values for each column
        updates = {}
        for i, col in enumerate(columns):
            updates[col] = transformed_features[:, i]

        return df.with_columns(**updates)

    def fit(self, df: pl.DataFrame) -> "CompositePipeline":
        """Fit the composite pipeline to training data.

        Args:
            df: Training DataFrame

        Returns:
            Self for method chaining
        """
        self._validate_dataframe(df)

        current_df = df
        self._fitted_steps = []

        for step in self.steps:
            self._validate_step_columns(current_df, step)

            # Convert to numpy array for pipeline processing
            step_array = self._dataframe_to_numpy(current_df, step.columns)

            # Create and fit appropriate pipeline
            if step.pipeline_type == "per_basin":
                pipeline = PerBasinPipeline(step.transforms)
            else:  # global
                pipeline = GlobalPipeline(step.transforms)

            pipeline.fit(step_array)
            self._fitted_steps.append(pipeline)

            # Transform current data for next step
            transformed_array = pipeline.transform(step_array)
            current_df = self._numpy_to_dataframe_update(current_df, transformed_array, step.columns)

        self._is_fitted = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform data using fitted pipeline.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame with same structure, updated column values
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform()")

        self._validate_dataframe(df)

        current_df = df

        for step, fitted_pipeline in zip(self.steps, self._fitted_steps, strict=True):
            self._validate_step_columns(current_df, step)

            # Convert to numpy for transformation
            step_array = self._dataframe_to_numpy(current_df, step.columns)

            # Apply transformation
            transformed_array = fitted_pipeline.transform(step_array)

            # Update DataFrame with transformed values
            current_df = self._numpy_to_dataframe_update(current_df, transformed_array, step.columns)

        return current_df

    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Inverse transform data using fitted pipeline.

        Args:
            df: Transformed DataFrame to inverse transform

        Returns:
            Inverse transformed DataFrame with original column values
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before inverse_transform()")

        self._validate_dataframe(df)

        current_df = df

        # Apply inverse transforms in reverse order
        for step, fitted_pipeline in zip(reversed(self.steps), reversed(self._fitted_steps), strict=True):
            self._validate_step_columns(current_df, step)

            # Convert to numpy for inverse transformation
            step_array = self._dataframe_to_numpy(current_df, step.columns)

            # Apply inverse transformation
            inverse_array = fitted_pipeline.inverse_transform(step_array)

            # Update DataFrame with inverse transformed values
            current_df = self._numpy_to_dataframe_update(current_df, inverse_array, step.columns)

        return current_df

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit pipeline and transform data in one step.

        Args:
            df: DataFrame to fit and transform

        Returns:
            Transformed DataFrame
        """
        return self.fit(df).transform(df)

    def get_step_summary(self) -> list[dict[str, Any]]:
        """Get summary information about each pipeline step.

        Returns:
            List of dictionaries with step information
        """
        summary = []
        for i, step in enumerate(self.steps):
            step_info = {
                "step": i,
                "pipeline_type": step.pipeline_type,
                "transforms": [type(t).__name__ for t in step.transforms],
                "columns": step.columns,
                "fitted": i < len(self._fitted_steps),
            }
            summary.append(step_info)
        return summary
