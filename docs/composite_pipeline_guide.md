# CompositePipeline Guide

## Overview

The `CompositePipeline` is a self-aware transformation system that applies sequential preprocessing steps to time series data while maintaining full metadata about its structure. This enables intelligent handling of partial DataFrames during inverse transformation, which is particularly important for model evaluation where predictions and observations need to be transformed back to their original scale.

## Key Concepts

### Self-Aware Pipelines

Traditional pipelines lose metadata about their structure after creation. Our `CompositePipeline` solves this by tracking:

- **Fitted columns**: All columns present when the pipeline was fitted
- **Transformed columns**: Specific columns that were actually transformed
- **Pipeline structure**: The sequence of transformations and their target columns

This metadata enables the pipeline to intelligently handle incomplete DataFrames during inverse transformation.

### The Problem It Solves

When evaluating models, we face an impedance mismatch:

1. **Training**: Pipeline transforms complete DataFrames with columns like `streamflow`, `precipitation`, `temperature`
2. **Evaluation**: Results contain only `prediction` and `observation` columns in long format
3. **Challenge**: How to inverse transform when most columns are missing and names don't match?

The `CompositePipeline` solves this with its `inverse_transform_partial()` method.

## Core Features

### 1. Metadata Tracking

The pipeline automatically tracks its structure during fitting:

```python
from transfer_learning_publication.transforms import (
    CompositePipeline,
    CompositePipelineStep,
    Log,
    ZScore
)

# Create pipeline with multiple steps
steps = [
    CompositePipelineStep(
        pipeline_type="per_basin",
        transforms=[Log()],
        columns=["streamflow"]
    ),
    CompositePipelineStep(
        pipeline_type="global",
        transforms=[ZScore()],
        columns=["streamflow", "precipitation", "temperature"]
    )
]

pipeline = CompositePipeline(steps, group_identifier="gauge_id")

# Fit the pipeline - it remembers the data structure
pipeline.fit(training_df)

# Query pipeline metadata
fitted_cols = pipeline.get_fitted_columns()
# Returns: ['gauge_id', 'streamflow', 'precipitation', 'temperature', 'area', ...]

transformed_cols = pipeline.get_transformed_columns()
# Returns: {'streamflow', 'precipitation', 'temperature'}

# Get detailed structure
structure = pipeline.describe()
print(structure)
# {
#     'fitted': True,
#     'fitted_columns': ['gauge_id', 'streamflow', ...],
#     'transformed_columns': ['streamflow', 'precipitation', 'temperature'],
#     'group_identifier': 'gauge_id',
#     'steps': [
#         {'type': 'per_basin', 'columns': ['streamflow'], 'transforms': ['Log']},
#         {'type': 'global', 'columns': ['streamflow', ...], 'transforms': ['ZScore']}
#     ]
# }
```

### 2. Partial Inverse Transform

The key innovation is handling incomplete DataFrames intelligently:

```python
# After model evaluation, you have predictions in transformed space
eval_df = pl.DataFrame({
    'gauge_id': ['01234567', '01234567', ...],
    'prediction': [2.34, 1.89, ...],  # These are Log+ZScore transformed
    'observation': [2.41, 1.92, ...]  # These are also transformed
})

# Problem: Pipeline expects 'streamflow' not 'prediction'
# Solution: Use inverse_transform_partial with column mapping

# Transform predictions back to original scale
pred_df = eval_df.select(['gauge_id', 'prediction'])
pred_inverse = pipeline.inverse_transform_partial(
    pred_df,
    column_mapping={'prediction': 'streamflow'}
)
# Returns DataFrame with only 'gauge_id' and 'prediction' (in original scale)

# Transform observations separately
obs_df = eval_df.select(['gauge_id', 'observation'])
obs_inverse = pipeline.inverse_transform_partial(
    obs_df,
    column_mapping={'observation': 'streamflow'}
)
```

### 3. Zero-Filling Strategy

Missing columns are automatically zero-filled during partial inverse transform:

```python
# Pipeline was fitted on full DataFrame with many columns
full_columns = ['gauge_id', 'streamflow', 'precipitation', 'temperature', 'area']

# But you only have streamflow values to inverse transform
partial_df = pl.DataFrame({
    'gauge_id': ['01234567'],
    'streamflow': [2.34]  # Only this column
})

# inverse_transform_partial automatically:
# 1. Zero-fills missing columns (precipitation, temperature, area)
# 2. Applies inverse transform to all columns
# 3. Returns only the originally present column (streamflow)
result = pipeline.inverse_transform_partial(partial_df)
# Result contains only: ['gauge_id', 'streamflow']
```

The zero-filling approach works because:
- Zero in normalized space (ZScore) maps to the mean in original space
- We only care about the columns we're inverting (others are dropped)
- It's simple, predictable, and already tested

### 4. Joblib Serialization Support

Pipeline metadata survives serialization:

```python
import joblib

# Save fitted pipeline
pipeline.fit(training_df)
joblib.dump(pipeline, 'pipeline.pkl')

# Load elsewhere
loaded_pipeline = joblib.load('pipeline.pkl')

# Metadata is preserved
assert loaded_pipeline.get_fitted_columns() == pipeline.get_fitted_columns()
assert loaded_pipeline.get_transformed_columns() == pipeline.get_transformed_columns()

# Can immediately use for inverse transform
result = loaded_pipeline.inverse_transform_partial(eval_df)
```

## API Reference

### Core Methods

#### `fit(df: pl.DataFrame) -> CompositePipeline`

Fit the pipeline to training data and store metadata.

```python
pipeline = CompositePipeline(steps, group_identifier="gauge_id")
pipeline.fit(training_df)
```

#### `transform(df: pl.DataFrame) -> pl.DataFrame`

Apply fitted transformations to new data.

```python
transformed_df = pipeline.transform(test_df)
```

#### `inverse_transform(df: pl.DataFrame) -> pl.DataFrame`

Apply inverse transformations to complete DataFrame.

```python
original_df = pipeline.inverse_transform(transformed_df)
```

### Metadata Methods

#### `get_fitted_columns() -> list[str]`

Returns all columns the pipeline was fitted on.

```python
columns = pipeline.get_fitted_columns()
# ['gauge_id', 'streamflow', 'precipitation', ...]
```

**Raises**: `RuntimeError` if pipeline is not fitted

#### `get_transformed_columns() -> set[str]`

Returns only columns that were actually transformed.

```python
transformed = pipeline.get_transformed_columns()
# {'streamflow', 'precipitation', 'temperature'}
```

**Raises**: `RuntimeError` if pipeline is not fitted

#### `describe() -> dict[str, Any]`

Returns complete pipeline structure for introspection.

```python
info = pipeline.describe()
print(info['steps'])  # List of transformation steps
print(info['fitted_columns'])  # All columns from training
```

### Partial Inverse Transform

#### `inverse_transform_partial(df: pl.DataFrame, column_mapping: dict[str, str] | None = None) -> pl.DataFrame`

Inverse transform a DataFrame with potentially missing columns.

**Parameters:**
- `df`: DataFrame with subset of original columns
- `column_mapping`: Map DataFrame columns to expected columns (e.g., `{"prediction": "streamflow"}`)

**Returns:**
- DataFrame with only the originally present columns, inverse transformed

**Example:**

```python
# Evaluation results with different column names
eval_df = pl.DataFrame({
    'gauge_id': ['basin_1', 'basin_2'],
    'prediction': [1.5, 2.0],  # Transformed streamflow values
})

# Map 'prediction' to 'streamflow' for inverse transform
result = pipeline.inverse_transform_partial(
    eval_df,
    column_mapping={'prediction': 'streamflow'}
)
# result has columns: ['gauge_id', 'prediction'] with original scale values
```

## Integration with ModelEvaluator

The `ModelEvaluator` uses `CompositePipeline` automatically when available:

```python
from transfer_learning_publication.models.model_evaluator import ModelEvaluator

# ModelEvaluator checks for pipeline in datamodule
evaluator = ModelEvaluator(
    models_and_datamodules={
        "lstm": (lstm_model, datamodule),  # datamodule may have pipeline
    }
)

# Test with automatic inverse transform (default)
results = evaluator.test_models(
    cache_dir="cache/exp001",
    apply_inverse_transform=True  # Default behavior
)
# Results are automatically in original scale if pipeline available

# Or disable inverse transform
results_transformed = evaluator.test_models(
    cache_dir="cache/exp001", 
    apply_inverse_transform=False
)
# Results remain in transformed space
```

### How ModelEvaluator Applies Inverse Transforms

1. Checks if datamodule has a fitted pipeline via `datamodule.get_pipeline()`
2. Gets target name from `datamodule.get_target_name()` (e.g., "streamflow")
3. Creates separate DataFrames for predictions and observations
4. Applies `inverse_transform_partial()` with appropriate column mapping
5. Updates the `ForecastOutput` with inverse transformed values

## Practical Examples

### Example 1: Training Pipeline for Streamflow Prediction

```python
import polars as pl
from transfer_learning_publication.transforms import (
    CompositePipeline,
    CompositePipelineStep,
    Log,
    ZScore
)

# Load training data
train_df = pl.read_parquet("data/train.parquet")
# Columns: gauge_id, streamflow, precipitation, temperature, area, elevation

# Create multi-step pipeline
steps = [
    # Step 1: Log transform streamflow per basin
    CompositePipelineStep(
        pipeline_type="per_basin",
        transforms=[Log()],
        columns=["streamflow"]
    ),
    # Step 2: Standardize all meteorological variables globally
    CompositePipelineStep(
        pipeline_type="global",
        transforms=[ZScore()],
        columns=["streamflow", "precipitation", "temperature"]
    )
]

# Create and fit pipeline
pipeline = CompositePipeline(steps, group_identifier="gauge_id")
pipeline.fit(train_df)

# Transform training data
train_transformed = pipeline.transform(train_df)

# Save for later use
import joblib
joblib.dump(pipeline, "models/preprocessing_pipeline.pkl")
```

### Example 2: Inverse Transform Evaluation Results

```python
# Load fitted pipeline
pipeline = joblib.load("models/preprocessing_pipeline.pkl")

# After model evaluation, you have results in transformed space
evaluation_results = pl.DataFrame({
    'model_name': ['lstm', 'lstm', 'lstm', 'lstm'],
    'gauge_id': ['01234567', '01234567', '87654321', '87654321'],
    'lead_time': [1, 2, 1, 2],
    'prediction': [1.23, 1.45, 0.89, 0.92],  # Transformed values
    'observation': [1.31, 1.42, 0.91, 0.88]  # Transformed values
})

# Process each basin's predictions
for basin_id in evaluation_results['gauge_id'].unique():
    basin_data = evaluation_results.filter(pl.col('gauge_id') == basin_id)
    
    # Inverse transform predictions
    pred_df = basin_data.select(['gauge_id', 'prediction'])
    pred_original = pipeline.inverse_transform_partial(
        pred_df,
        column_mapping={'prediction': 'streamflow'}
    )
    
    # Inverse transform observations
    obs_df = basin_data.select(['gauge_id', 'observation'])
    obs_original = pipeline.inverse_transform_partial(
        obs_df,
        column_mapping={'observation': 'streamflow'}
    )
    
    print(f"Basin {basin_id}:")
    print(f"  Predictions (original scale): {pred_original['prediction'].to_list()}")
    print(f"  Observations (original scale): {obs_original['observation'].to_list()}")
```

### Example 3: Handling Multiple Target Scenarios

```python
# If you have multiple models predicting different targets
pipelines = {
    'streamflow': streamflow_pipeline,
    'soil_moisture': soil_moisture_pipeline,
}

def inverse_transform_results(results_df, target_name):
    """Helper to inverse transform based on target type."""
    pipeline = pipelines[target_name]
    
    # Transform predictions
    pred_inverse = pipeline.inverse_transform_partial(
        results_df.select(['gauge_id', 'prediction']),
        column_mapping={'prediction': target_name}
    )
    
    # Transform observations  
    obs_inverse = pipeline.inverse_transform_partial(
        results_df.select(['gauge_id', 'observation']),
        column_mapping={'observation': target_name}
    )
    
    # Combine results
    return results_df.with_columns([
        pred_inverse['prediction'].alias('prediction_original'),
        obs_inverse['observation'].alias('observation_original')
    ])
```

## Design Rationale

### Why Self-Aware Pipelines?

1. **Information Locality**: The pipeline knows its own structure best
2. **Single Source of Truth**: Metadata lives with the transformation logic
3. **Serialization Safety**: Metadata persists through save/load cycles
4. **Clean API**: Users don't need to track metadata separately

### Why Zero-Filling?

1. **Simplicity**: One strategy that works for all transforms
2. **Mathematical Soundness**: Zero in normalized space = mean in original space
3. **Practical**: We drop the filled columns anyway
4. **Tested**: The approach is already validated in our test suite

### Why Separate Predictions and Observations?

1. **Column Independence**: They're in different columns but represent the same variable
2. **Flexibility**: Can transform only what you need
3. **Memory Efficiency**: No need to duplicate data

## Common Patterns

### Pattern 1: Pipeline with Data Module

```python
class LSHDataModule:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self._pipeline = None
        self._pipeline_path = self.config.get('pipeline_path')
    
    def get_pipeline(self):
        """Lazy load pipeline when needed."""
        if self._pipeline is None and self._pipeline_path:
            self._pipeline = joblib.load(self._pipeline_path)
        return self._pipeline
    
    def get_target_name(self):
        """Return target variable name."""
        return self.config['features']['target']
```

### Pattern 2: Conditional Inverse Transform

```python
# Only inverse transform if pipeline is available and fitted
pipeline = datamodule.get_pipeline()

if pipeline and hasattr(pipeline, '_is_fitted') and pipeline._is_fitted:
    results = pipeline.inverse_transform_partial(
        results_df,
        column_mapping={'prediction': target_name}
    )
else:
    # Keep original transformed values
    results = results_df
```

## Testing

The `CompositePipeline` has comprehensive test coverage in `tests/test_composite.py`:

- Metadata tracking and persistence
- Partial inverse transform with missing columns
- Column mapping both directions
- Joblib serialization/deserialization
- Zero-filling behavior
- Integration with various transform types

Run tests with:

```bash
uv run pytest tests/test_composite.py -v
```

## Troubleshooting

### Pipeline Not Fitted Error

```python
RuntimeError: Pipeline must be fitted before inverse_transform()
```

**Solution**: Ensure `pipeline.fit()` was called before attempting transforms.

### Column Not Found in Fitted Columns

```python
ValueError: Column 'invalid_col' was not present during pipeline fitting
```

**Solution**: Check column mapping matches columns from original training data.

### Missing Group Identifier

```python
ValueError: Group identifier 'gauge_id' not found in DataFrame columns
```

**Solution**: Ensure DataFrame includes the group identifier column specified during pipeline creation.

## Best Practices

1. **Always Fit on Complete Data**: Fit the pipeline on DataFrames with all expected columns
2. **Use Consistent Group Identifiers**: Match the identifier across pipeline and evaluation
3. **Save Fitted Pipelines**: Serialize after fitting for reuse
4. **Document Column Mappings**: Make mappings explicit in your code
5. **Test Inverse Transforms**: Verify values return to expected scale

## Conclusion

The `CompositePipeline` provides a robust solution for handling complex transformation workflows in time series forecasting. Its self-aware design and intelligent partial inverse transform capabilities make it particularly well-suited for model evaluation scenarios where data structure changes between training and inference.