from typing import Literal

import polars as pl

from .base import BaseTransform
from .composite import CompositePipeline, CompositePipelineStep


class PipelineBuilder:
    """Fluent API builder for creating composite transformation pipelines.

    Provides intuitive methods for composing data preprocessing pipelines that
    operate on Polars DataFrames with grouped data (e.g., hydrological basins).

    Key features:
    - Column-specific transformations
    - Automatic optimization of consecutive operations
    - Preserves all DataFrame columns (only specified ones are transformed)
    - Support for both per-basin and global transformation strategies

    Example:
        ```python
        pipeline = (
            PipelineBuilder(group_identifier="basin_id")
            .add_per_basin(Log(), columns=["streamflow"])
            .add_global(ZScore(), columns=["streamflow", "precipitation"])
            .build()
        )

        df_transformed = pipeline.fit_transform(df)
        ```
    """

    def __init__(self, group_identifier: str):
        self.group_identifier = group_identifier
        self._steps: list[CompositePipelineStep] = []

    def add_per_basin(self, transforms: BaseTransform | list[BaseTransform], columns: list[str]) -> "PipelineBuilder":
        """Add per-basin transformation step.

        Transforms are applied separately to each group/basin. Each group gets
        its own fitted transform instances.

        Args:
            transforms: Single transform or list of transforms to apply
            columns: Column names to transform

        Returns:
            Self for method chaining
        """
        return self._add_step("per_basin", transforms, columns)

    def add_global(self, transforms: BaseTransform | list[BaseTransform], columns: list[str]) -> "PipelineBuilder":
        """Add global transformation step.

        Transforms are fitted on pooled data from all groups and applied
        uniformly across all groups.

        Args:
            transforms: Single transform or list of transforms to apply
            columns: Column names to transform

        Returns:
            Self for method chaining
        """
        return self._add_step("global", transforms, columns)

    def _add_step(
        self,
        pipeline_type: Literal["per_basin", "global"],
        transforms: BaseTransform | list[BaseTransform],
        columns: list[str],
    ) -> "PipelineBuilder":
        """Add a transformation step with automatic optimization.

        Args:
            pipeline_type: Type of pipeline ("per_basin" or "global")
            transforms: Transform(s) to add
            columns: Columns to transform

        Returns:
            Self for method chaining
        """
        transform_list = [transforms] if isinstance(transforms, BaseTransform) else list(transforms)

        if not transform_list:
            raise ValueError("Must provide at least one transform")
        if not columns:
            raise ValueError("Must specify at least one column")

        if self._can_optimize_with_previous_step(pipeline_type, columns):
            self._steps[-1].transforms.extend(transform_list)

        else:
            step = CompositePipelineStep(
                pipeline_type=pipeline_type,
                transforms=transform_list,
                columns=columns.copy(),
            )
            self._steps.append(step)

        return self

    def _can_optimize_with_previous_step(
        self, pipeline_type: Literal["per_basin", "global"], columns: list[str]
    ) -> bool:
        """Check if current operation can be merged with the previous step.

        Optimization is possible when:
        1. There is a previous step
        2. Pipeline types match (per_basin/global)
        3. Column sets match exactly

        Args:
            pipeline_type: Type of current operation
            columns: Columns for current operation

        Returns:
            True if optimization is possible
        """
        if not self._steps:
            return False

        previous_step = self._steps[-1]

        if previous_step.pipeline_type != pipeline_type:
            return False

        return set(previous_step.columns) == set(columns)

    def build(self) -> CompositePipeline:
        """Build the composite pipeline from configured steps.

        Returns:
            CompositePipeline ready for fitting and transformation

        Raises:
            ValueError: If no steps have been added
        """
        if not self._steps:
            raise ValueError("Cannot build pipeline: no steps have been added")

        return CompositePipeline(self._steps.copy(), self.group_identifier)

    def get_step_preview(self) -> list[dict]:
        """Get preview of current pipeline configuration.

        Returns:
            List of dictionaries describing each step
        """
        preview = []
        for i, step in enumerate(self._steps):
            step_info = {
                "step": i,
                "pipeline_type": step.pipeline_type,
                "transforms": [type(t).__name__ for t in step.transforms],
                "columns": step.columns,
                "num_transforms": len(step.transforms),
            }
            preview.append(step_info)
        return preview

    def reset(self) -> "PipelineBuilder":
        """Clear all configured steps and start fresh.

        Returns:
            Self for method chaining
        """
        self._steps.clear()
        return self

    def validate_against_dataframe(self, df: pl.DataFrame) -> list[str]:
        """Validate current pipeline configuration against a DataFrame.

        Checks for common issues like missing columns or empty steps.

        Args:
            df: DataFrame to validate against

        Returns:
            List of validation error messages (empty if no issues)
        """
        errors = []

        if not isinstance(df, pl.DataFrame):
            errors.append("Input must be a Polars DataFrame")
            return errors

        if self.group_identifier not in df.columns:
            errors.append(f"Group identifier '{self.group_identifier}' not found in DataFrame")

        for i, step in enumerate(self._steps):
            missing_cols = [col for col in step.columns if col not in df.columns]
            if missing_cols:
                errors.append(f"Step {i}: columns {missing_cols} not found in DataFrame")

        if df.shape[0] == 0:
            errors.append("DataFrame cannot be empty")

        return errors
