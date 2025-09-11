# ModelEvaluator Guide

## Overview

The `ModelEvaluator` is a powerful orchestration system for testing and evaluating multiple trained time series forecasting models. It abstracts the repetitive testing workflow, provides intelligent caching to avoid redundant computations, and produces standardized outputs for downstream analysis.

## Key Features

- **Batch Testing**: Test multiple models in a single call
- **Intelligent Caching**: Automatic caching with Hive-style partitioning to avoid recomputation
- **Cache Validation**: Detects configuration changes and invalidates stale cache
- **Incremental Updates**: Add and test new models without recomputing existing results
- **Standardized Output**: Transforms model outputs into consistent DataFrame format
- **Flexible Filtering**: Access results by model, basin, or lead time

## Architecture

The ModelEvaluator system consists of three main components:

### 1. ModelEvaluator Class

The main orchestrator that manages model testing, caching, and result collection.

### 2. EvaluationResults Contract

A standardized container that transforms `ForecastOutput` objects into analysis-ready DataFrames.

### 3. Cache System

Hive-partitioned cache structure that stores results by model name for efficient incremental updates.

## Basic Usage

### Setting Up the Evaluator

```python
from transfer_learning_publication.models.model_evaluator import ModelEvaluator
from transfer_learning_publication.data import LSHDataModule

# Load your trained models and datamodule
datamodule = LSHDataModule(config_path="configs/data_config.yaml")
# ... load your trained models ...

# Create evaluator with multiple models
evaluator = ModelEvaluator(
    models_and_datamodules={
        "tide": (tide_model, datamodule),
        "ealstm": (ealstm_model, datamodule),
        "naive": (naive_model, datamodule),
    },
    trainer_kwargs={"accelerator": "cpu", "devices": 1}
)
```

### Testing Models

```python
# Test all models and cache results
results = evaluator.test_models(cache_dir="cache/experiment_001")

# On subsequent runs, results are loaded from cache (fast!)
results = evaluator.test_models(cache_dir="cache/experiment_001")

# Test specific models only
results = evaluator.test_models(
    cache_dir="cache/experiment_001",
    only=["tide", "ealstm"]
)

# Force recomputation even if cache exists
results = evaluator.test_models(
    cache_dir="cache/experiment_001",
    force_recompute=True
)

# Disable inverse transform to keep results in transformed space
results_transformed = evaluator.test_models(
    cache_dir="cache/experiment_001",
    apply_inverse_transform=False  # Default is True
)
```

### Accessing Results

The `EvaluationResults` object provides multiple ways to access and filter predictions:

```python
# Get raw DataFrame with all predictions
df = results.raw_data
# Columns: model_name, group_identifier, lead_time, prediction, observation

# Filter by model
tide_results = results.by_model("tide")

# Filter by basin/gauge
basin_results = results.by_basin("01234567")

# Filter by lead time (forecast horizon)
day1_results = results.by_lead_time(1)

# Get summary statistics
summary = results.summary()
# Returns DataFrame with: model_name, n_samples, n_basins, output_length, has_dates

# List available models and basins
models = results.list_models()
basins = results.list_basins()
```

## Inverse Transform Support

### Automatic Pipeline Integration

The ModelEvaluator now automatically applies inverse transforms when a preprocessing pipeline is available. This ensures evaluation results are in the original scale, making them interpretable and suitable for domain-specific metrics.

```python
# If your datamodule has a fitted pipeline, inverse transform is automatic
evaluator = ModelEvaluator(
    models_and_datamodules={
        "lstm": (lstm_model, datamodule),  # datamodule.get_pipeline() returns fitted pipeline
    }
)

# Results are automatically in original scale
results = evaluator.test_models()  # apply_inverse_transform=True by default

# Check if streamflow values are in original scale (e.g., m³/s)
print(results.raw_data["prediction"].describe())
# Original scale: might show min=0.5, max=1000.0 (realistic flow values)
# Transformed scale: might show min=-2.3, max=3.1 (normalized values)
```

### How It Works

1. **Pipeline Detection**: ModelEvaluator checks if the datamodule has a pipeline via `get_pipeline()`
2. **Target Identification**: Gets the target variable name from `get_target_name()` (e.g., "streamflow")
3. **Separate Processing**: Predictions and observations are inverse transformed independently
4. **Column Mapping**: Maps "prediction" → target name → "prediction" for seamless integration
5. **Smart Handling**: Only transforms if pipeline is fitted and available

### Controlling Inverse Transform

```python
# Keep results in transformed space for custom processing
results_transformed = evaluator.test_models(
    apply_inverse_transform=False
)

# Apply your own transformations
custom_results = my_custom_inverse_transform(results_transformed.raw_data)

# Or selectively inverse transform later
if need_original_scale:
    pipeline = datamodule.get_pipeline()
    if pipeline:
        # Use the pipeline's partial inverse transform
        predictions_original = pipeline.inverse_transform_partial(
            results_transformed.raw_data[["group_identifier", "prediction"]],
            column_mapping={"prediction": "streamflow"}
        )
```

### Requirements for Inverse Transform

For automatic inverse transform to work:

1. **DataModule Methods**: Must implement:
   - `get_pipeline()`: Returns CompositePipeline or None
   - `get_target_name()`: Returns target column name
   - `get_group_identifier_name()`: Returns group ID column name

2. **Pipeline State**: Pipeline must be fitted (`pipeline._is_fitted == True`)

3. **Compatible Structure**: Group identifiers must match between pipeline and evaluation data

See the [CompositePipeline Guide](composite_pipeline_guide.md) for detailed pipeline documentation.

## Advanced Features

### Incremental Model Addition

Add and test new models without recomputing existing cached results:

```python
# Initial setup and testing
evaluator = ModelEvaluator(
    models_and_datamodules={"lstm": (lstm_model, datamodule)}
)
results = evaluator.test_models(cache_dir="cache/exp001")

# Later: Add new models
evaluator.add_models({
    "gru": (gru_model, datamodule),
    "transformer": (transformer_model, datamodule)
})

# Test only the new models
new_results = evaluator.test_models(
    cache_dir="cache/exp001",
    only=["gru", "transformer"]
)

# Merge results
results.update(new_results)
```

### Cache Management

```python
# Get detailed cache information
cache_info = evaluator.get_cache_info("cache/exp001")
print(cache_info)
# {
#     "exists": True,
#     "path": "cache/exp001",
#     "models": [
#         {
#             "model_name": "tide",
#             "has_output": True,
#             "has_config": True,
#             "is_valid": True,
#             "output_size_bytes": 12345
#         },
#         ...
#     ],
#     "metadata": {...}
# }

# Validate specific model cache
is_valid = evaluator.validate_cache("cache/exp001", "tide")

# Clear cache for specific models
evaluator.clear_cache("cache/exp001", model_names=["tide"])

# Clear entire cache
evaluator.clear_cache("cache/exp001")
```

### Exporting Results

Export results to Parquet format with partitioning:

```python
# Default: Partition by model_name
results.to_parquet("results/predictions/")
# Creates: results/predictions/model_name=tide/*.parquet
#          results/predictions/model_name=ealstm/*.parquet

# Custom partitioning
results.to_parquet(
    "results/by_lead_time/",
    partition_cols=["lead_time"]
)
# Creates: results/by_lead_time/lead_time=1/*.parquet
#          results/by_lead_time/lead_time=2/*.parquet
```

## Cache Structure

The cache uses Hive-style partitioning for efficient storage and retrieval:

```text
cache/experiment_001/
├── metadata.json                    # Cache metadata
├── model_name=tide/
│   ├── forecast_output.joblib      # Serialized ForecastOutput
│   └── config.json                  # DataModule config for validation
├── model_name=ealstm/
│   ├── forecast_output.joblib
│   └── config.json
└── model_name=naive/
    ├── forecast_output.joblib
    └── config.json
```

### Cache Validation

The cache is automatically validated before use. A cache entry is considered invalid if:

1. The DataModule configuration has changed (different features, sequence lengths, etc.)
2. Required files are missing (forecast_output.joblib or config.json)
3. The model is no longer in the evaluator's registry

When cache is invalid, the model is automatically retested and the cache is updated.

## Output DataFrame Format

The `EvaluationResults.raw_data` DataFrame has the following structure:

| Column | Type | Description |
|--------|------|-------------|
| model_name | str | Name of the model |
| group_identifier | str | Basin or gauge ID |
| lead_time | int | Forecast horizon (1 to output_length) |
| prediction | float | Model prediction value† |
| observation | float | Ground truth value† |
| issue_date* | datetime | When forecast was issued |
| prediction_date* | datetime | Date being predicted |

*Date columns are only included if the model provides timestamp information.

†Values are in original scale if `apply_inverse_transform=True` (default) and a pipeline is available, otherwise in transformed scale.

## Integration with Training Pipeline

The ModelEvaluator assumes models are already trained. Here's a typical workflow:

```python
import lightning as pl
from transfer_learning_publication.models import ModelFactory
from transfer_learning_publication.data import LSHDataModule
from transfer_learning_publication.models.model_evaluator import ModelEvaluator

# 1. Setup data
datamodule = LSHDataModule(config_path="configs/data_config.yaml")

# 2. Train models
model_factory = ModelFactory()
models = {}

for model_name in ["tide", "ealstm", "tsmixer"]:
    # Create model
    model = model_factory.create_from_config(f"configs/{model_name}_config.yaml")
    
    # Train
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model=model, datamodule=datamodule)
    
    # Store trained model
    models[model_name] = model

# 3. Evaluate all models
evaluator = ModelEvaluator(
    models_and_datamodules={
        name: (model, datamodule) for name, model in models.items()
    }
)

results = evaluator.test_models(cache_dir="cache/experiment_001")

# 4. Analyze results
summary = results.summary()
print(summary)
```

## Configuration Requirements

### DataModule Requirements

The `LSHDataModule` must implement several methods for full functionality:

#### Configuration Method

`get_config_dict()` returns configuration for cache validation:

```python
{
    "input_length": 365,
    "output_length": 7,
    "forcing_features": ["streamflow", "precipitation", "temperature"],
    "static_features": ["area", "elevation"],
    "future_features": ["temperature"],
    "target_name": "streamflow",
    "is_autoregressive": True,
    "group_identifier_name": "gauge_id",
    "include_dates": False
}
```

#### Pipeline Access Methods (for inverse transform)

```python
def get_pipeline(self) -> CompositePipeline | None:
    """Return preprocessing pipeline if available."""
    if self._pipeline is None and self._pipeline_path:
        self._pipeline = joblib.load(self._pipeline_path)
    return self._pipeline

def get_target_name(self) -> str:
    """Return target column name."""
    return self.config["features"]["target"]

def get_group_identifier_name(self) -> str:
    """Return group identifier column name."""
    return "gauge_id"  # Or from config
```

### Model Requirements

Models must:

1. Extend `BaseLitModel`
2. Store test results in the `forecast_output` property after testing
3. Return predictions with shape `[batch_size, output_len, 1]`

## Performance Considerations

### Caching Benefits

- **First run**: Full computation for all models
- **Subsequent runs**: Near-instantaneous loading from cache
- **Incremental updates**: Only new/changed models are computed

### Memory Management

- Results are loaded on-demand from cache
- Large datasets can be processed in batches
- Parquet export enables efficient storage and retrieval

### Parallel Testing

When testing multiple models without cache, consider:

```python
# Models share the same datamodule (memory efficient)
evaluator = ModelEvaluator(
    models_and_datamodules={
        "model1": (model1, shared_datamodule),
        "model2": (model2, shared_datamodule),
    }
)
```

## Error Handling

The ModelEvaluator provides clear error messages and logging:

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

# The evaluator will log:
# - Cache hits/misses
# - Models being tested
# - Cache validation results
# - Warnings for overwriting results
```

Common errors and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: Testing failed` | Model testing error | Check model and data compatibility |
| `ValueError: No models to test` | Empty model list | Add models before testing |
| `KeyError: Model not found` | Requesting unknown model | Check model name in registry |
| `ValueError: Different output_lengths` | Merging incompatible results | Ensure all models have same forecast horizon |

## Best Practices

1. **Use consistent DataModule**: Share the same DataModule across models for fair comparison
2. **Set cache directory**: Always specify a cache directory to avoid recomputation
3. **Version your cache**: Use different cache directories for different experiments
4. **Monitor cache size**: Periodically clear old cache directories
5. **Validate before production**: Use `force_recompute=True` periodically to ensure cache validity

## Example: Complete Experiment

```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Setup evaluator with multiple models
evaluator = ModelEvaluator(
    models_and_datamodules={
        "tide": (tide_model, datamodule),
        "ealstm": (ealstm_model, datamodule),
        "naive": (naive_baseline, datamodule),
    }
)

# Run evaluation with caching and automatic inverse transform
cache_dir = Path("cache") / "experiment_001"
results = evaluator.test_models(
    cache_dir=cache_dir,
    apply_inverse_transform=True  # Default - results in original scale
)

# Get summary statistics
summary = results.summary()
print("\nModel Summary:")
print(summary)

# Check if results are in original scale
if results.raw_data["prediction"].min() > 0:
    print("Results are in original scale (e.g., m³/s for streamflow)")
else:
    print("Results appear to be in transformed scale (normalized)")

# Export to parquet for further analysis
results.to_parquet("results/experiment_001/")

# Analyze specific basin
basin_id = "01234567"
basin_df = results.by_basin(basin_id)

# Calculate metrics per model
for model_name in results.list_models():
    model_df = basin_df[basin_df["model_name"] == model_name]
    
    # Calculate RMSE
    rmse = ((model_df["prediction"] - model_df["observation"]) ** 2).mean() ** 0.5
    print(f"{model_name} RMSE for basin {basin_id}: {rmse:.4f}")

# Visualize lead time performance
lead1_df = results.by_lead_time(1)
pivot_df = lead1_df.pivot_table(
    values="prediction",
    index="group_identifier",
    columns="model_name",
    aggfunc="mean"
)

# Plot model comparison
pivot_df.plot(kind="box", figsize=(10, 6))
plt.title("Model Performance Comparison (Lead Time 1)")
plt.ylabel("Prediction Value")
plt.show()
```

## API Reference

### ModelEvaluator

```python
class ModelEvaluator:
    def __init__(
        self,
        models_and_datamodules: dict[str, tuple[BaseLitModel, LSHDataModule]] | None = None,
        trainer_kwargs: dict[str, Any] | None = None,
    )
    
    def add_models(
        self,
        models_and_datamodules: dict[str, tuple[BaseLitModel, LSHDataModule]]
    ) -> None
    
    def test_models(
        self,
        cache_dir: str | Path | None = None,
        only: list[str] | None = None,
        force_recompute: bool = False,
        apply_inverse_transform: bool = True,
    ) -> EvaluationResults
    
    def validate_cache(self, cache_dir: str | Path, model_name: str) -> bool
    
    def get_cache_info(self, cache_dir: str | Path) -> dict[str, Any]
    
    def clear_cache(
        self,
        cache_dir: str | Path,
        model_names: list[str] | None = None
    ) -> None
    
    def get_model(self, model_name: str) -> BaseLitModel
    
    def get_datamodule(self, model_name: str) -> LSHDataModule
    
    def list_models(self) -> list[str]
```

### EvaluationResults

```python
class EvaluationResults:
    @property
    def raw_data(self) -> pd.DataFrame
    
    def by_model(self, model_name: str) -> pd.DataFrame
    
    def by_basin(self, basin_id: str) -> pd.DataFrame
    
    def by_lead_time(self, lead_time: int) -> pd.DataFrame
    
    def update(self, other: EvaluationResults) -> None
    
    def to_parquet(
        self,
        output_dir: str | Path,
        partition_cols: list[str] | None = None
    ) -> None
    
    def list_models(self) -> list[str]
    
    def list_basins(self) -> list[str]
    
    def summary(self) -> pd.DataFrame
```

## Troubleshooting

### Q: Why is my cache not being used?

Check that:

1. The DataModule configuration hasn't changed
2. The cache directory exists and contains valid files
3. You're not using `force_recompute=True`

### Q: How do I compare models with different output lengths?

Models must have the same output length to be compared. Train separate evaluators for different forecast horizons.

### Q: Can I use different DataModules for different models?

Yes, but this may affect fair comparison. The cache validation is per-model, so each model's cache is validated against its own DataModule configuration.

### Q: How large can the cache grow?

Cache size depends on:

- Number of models
- Number of test samples
- Output length

Each model's cache is typically a few MB to hundreds of MB for large datasets.

## Conclusion

The ModelEvaluator provides a robust, efficient system for evaluating multiple time series forecasting models. Its intelligent caching, standardized output format, and flexible API make it ideal for both experimentation and production evaluation pipelines.
