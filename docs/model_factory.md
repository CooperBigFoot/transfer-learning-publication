# Model Factory System

The model factory provides a centralized registry and creation system for all time series forecasting models in the framework. It simplifies model instantiation from YAML configurations and enables easy discovery of available models.

## Quick Start

### Creating a Model

```python
from pathlib import Path
from transfer_learning_publication.models.model_factory import ModelFactory
from transfer_learning_publication.data import LSHDataModule

# New way: Create model from experiment configuration
model = ModelFactory.create_from_config("experiments/tide_365_10.yaml")
datamodule = LSHDataModule("experiments/tide_365_10.yaml")

# Or the traditional way: Create model with explicit type and config
model = ModelFactory.create("tide", Path("configs/experiment.yaml"))

# The model is ready for training with PyTorch Lightning
trainer.fit(model, datamodule)

# Note: If datamodule has a preprocessing pipeline, ModelEvaluator will
# automatically apply inverse transforms during evaluation
```

### Discovering Available Models

```python
# List all registered models
available_models = ModelFactory.list_available()
print(available_models)
# Output: ['ealstm', 'naive_last_value', 'tft', 'tide', 'tsmixer']

# Get the configuration class for a specific model
config_class = ModelFactory.get_config_class("tide")
```

## YAML Configuration Structure

### Experiment Configuration (New)

The new `create_from_config()` method expects experiment configurations with this structure:

```yaml
# Sequence configuration
sequence:
  input_length: 365    # Historical sequence length
  output_length: 10     # Forecast horizon

# Feature configuration
features:
  # Time-varying input features
  forcing:
    - "streamflow"
    - "total_precipitation_sum"
    - "temperature_2m_mean"
  
  # Time-invariant catchment attributes
  static:
    - "area"
    - "elevation"
  
  # Known future features (optional, subset of forcing)
  future:
    - "total_precipitation_sum"
    - "temperature_2m_mean"
  
  # Target variable to predict
  target: "streamflow"

# Data preparation settings
data_preparation:
  is_autoregressive: true    # Whether to include target in inputs
  include_dates: false       # Whether to include timestamps

# Model configuration
model:
  type: "tide"                                    # Required: model architecture
  config_file: "configs/models/tide_best.yaml"    # Optional: external hyperparameters
  overrides:                                       # Optional: experiment-specific overrides
    learning_rate: 0.001
    dropout: 0.2

# Training parameters (optional)
training:
  learning_rate: 0.001  # Note: will be overridden by model.overrides if present
```

### External Hyperparameter Files

Hyperparameter files (referenced by `model.config_file`) contain model-specific parameters:

```yaml
# Example: configs/models/tide_best.yaml
hidden_size: 128
dropout: 0.1
num_encoder_layers: 2
decoder_output_size: 16
temporal_decoder_hidden_size: 32
learning_rate: 0.0005
```

### Parameter Priority

When using `create_from_config()`, parameters are merged with this priority:

1. **Highest**: `model.overrides` - Experiment-specific tweaks
2. **Middle**: `model.config_file` - Tuned hyperparameters from external file
3. **Lowest**: Model defaults - Built-in default values

Data-derived parameters (input_len, output_len, etc.) are always extracted from the experiment config.

### Legacy Configuration (for `create()` method)

The traditional `create()` method still works with the original format:

```yaml
# Direct model parameters embedded in config
sequence:
  input_length: 365
  output_length: 10

features:
  forcing: ["streamflow", "precipitation"]
  static: ["area"]
  target: "streamflow"

model:
  hidden_size: 128
  dropout: 0.1
  # ... other model-specific parameters
```

## Available Models

| Model Name | Description | Config Class |
|------------|-------------|--------------|
| `tide` | Time-series Dense Encoder | `TiDEConfig` |
| `ealstm` | Entity-Aware LSTM | `EALSTMConfig` |
| `tsmixer` | Time Series Mixer | `TSMixerConfig` |
| `tft` | Temporal Fusion Transformer | `TFTConfig` |
| `naive_last_value` | Baseline that repeats last value | `NaiveLastValueConfig` |

## Adding a New Model

To register a new model with the factory:

### 1. Create Your Model Components

Follow the conventions in `docs/model_implementation_guidelines.md`:

- Configuration class extending `BaseConfig`
- Core model (nn.Module)
- Lightning wrapper extending `BaseLitModel`

### 2. Register with the Factory

Add the `@register_model` decorator to your Lightning module:

```python
from ..model_factory import register_model
from .config import YourModelConfig

@register_model("your_model_name", config_class=YourModelConfig)
class LitYourModel(BaseLitModel):
    def __init__(self, config):
        # Handle dict or config object
        if isinstance(config, dict):
            config = YourModelConfig.from_dict(config)
        super().__init__(config)
        self.model = YourModel(config)
    
    def forward(self, x, static=None, future=None):
        return self.model(x, static, future)
```

### 3. Import in Factory Module

Add an import statement at the bottom of `model_factory.py`:

```python
# In model_factory.py, at the bottom with other imports
from .your_model.lightning import LitYourModel
```

That's it! Your model is now registered and can be created via the factory.

## Configuration Mapping

The factory automatically maps YAML structure to model configuration parameters:

| YAML Path | Config Parameter | Description |
|-----------|-----------------|-------------|
| `sequence.input_length` | `input_len` | Historical sequence length |
| `sequence.output_length` | `output_len` | Forecast horizon |
| `len(features.forcing)` | `input_size` | Number of input features |
| `len(features.static)` | `static_size` | Number of static features |
| `len(features.future)` | `future_input_size` | Number of future features |
| `features.target` | `target` | Target variable name |
| `training.learning_rate` | `learning_rate` | Learning rate |
| `model.*` | (various) | Model-specific parameters |

## Error Handling

The factory provides clear error messages:

```python
# Unknown model
ModelFactory.create("unknown_model", "config.yaml")
# ValueError: Model 'unknown_model' not found in registry. 
# Available models: ['ealstm', 'naive_last_value', 'tft', 'tide', 'tsmixer']

# Missing configuration file
ModelFactory.create("tide", "nonexistent.yaml")
# FileNotFoundError: Configuration file not found: nonexistent.yaml

# Invalid YAML
ModelFactory.create("tide", "invalid.yaml")
# yaml.YAMLError: ...
```

## Example: Complete Workflow

### Using the New Experiment-Based Approach

```python
from pathlib import Path
from transfer_learning_publication.models.model_factory import ModelFactory
from transfer_learning_publication.data import LSHDataModule
import lightning as L

# 1. List available models
print("Available models:", ModelFactory.list_available())

# 2. Create model and datamodule from same experiment config
config_path = "experiments/tide_365_10.yaml"
model = ModelFactory.create_from_config(config_path)
datamodule = LSHDataModule(config_path)

# 3. Train with PyTorch Lightning
trainer = L.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
)
trainer.fit(model, datamodule)

# 4. Test the model
trainer.test(model, datamodule)
```

### Using External Hyperparameters with Overrides

```python
# Your experiment config (experiments/tsmixer_tuning.yaml) might have:
# model:
#   type: "tsmixer"
#   config_file: "configs/models/tsmixer_best.yaml"
#   overrides:
#     learning_rate: 0.0001  # Override for this specific experiment

model = ModelFactory.create_from_config("experiments/tsmixer_tuning.yaml")
# This will:
# 1. Load hyperparameters from configs/models/tsmixer_best.yaml
# 2. Override learning_rate with 0.0001
# 3. Extract data parameters from the experiment config
# 4. Create the TSMixer model with merged configuration
```

### Legacy Approach (Still Supported)

```python
# Traditional way with explicit model type
model = ModelFactory.create("tide", "configs/experiment.yaml")
datamodule = LSHDataModule("configs/experiment.yaml")
trainer.fit(model, datamodule)
```

## Testing the Factory

Run the factory tests to ensure everything is working:

```bash
uv run pytest tests/models/test_model_factory.py -v
```

**Note on Transformed Data**: Models work with preprocessed data during training. If your data uses a `CompositePipeline` for transformations (Log, ZScore, etc.), the models train on transformed values. The `ModelEvaluator` automatically handles inverse transforms during evaluation to compute metrics on the original scale. See the [CompositePipeline Guide](composite_pipeline_guide.md) for details.

## Design Rationale

The factory design prioritizes:

1. **Simplicity**: Single file implementation, straightforward API
2. **Self-registration**: Models register themselves via decorator
3. **Compatibility**: Works with existing YAML structure
4. **Extensibility**: Easy to add new models
5. **Type safety**: Proper type hints throughout
6. **Clear errors**: Helpful error messages for common issues

## Migration Guide

### Migrating from Old to New Configuration Structure

If you have existing configurations using the old structure, here's how to migrate:

**Old Structure:**
```yaml
model:
  is_autoregressive: true
  include_dates: false
  hidden_size: 128
  dropout: 0.1
```

**New Structure:**
```yaml
data_preparation:
  is_autoregressive: true
  include_dates: false

model:
  type: "tide"
  overrides:  # Or use config_file for external params
    hidden_size: 128
    dropout: 0.1
```

### Best Practices

1. **Organize Hyperparameters**: Keep tuned hyperparameters in `configs/models/` directory
2. **Name Experiments Clearly**: Use descriptive names like `tide_365_10_camels.yaml`
3. **Document Overrides**: Add comments explaining why specific overrides are used
4. **Version Control**: Track both experiment configs and hyperparameter files
5. **Reuse Hyperparameters**: Share successful hyperparameters across experiments via `config_file`

## Troubleshooting

### Model not appearing in registry

- Ensure the `@register_model` decorator is applied to the Lightning class
- Check that the model is imported in `model_factory.py`
- Verify the model name is unique

### Configuration errors

- Check YAML structure matches expected format
- Ensure required parameters are present
- Model-specific parameters go in the `model:` section

### Import errors

- The factory imports all models at module load time
- If a model has missing dependencies, it will fail at import
- Check that all model dependencies are installed

## API Reference

### `ModelFactory.create_from_config(config_path: Path | str) -> BaseLitModel`

Create a model instance from an experiment configuration file.

**Parameters:**

- `config_path`: Path to experiment configuration file

**Returns:**

- Instantiated Lightning module ready for training

**Raises:**

- `ValueError`: If `model.type` not found in config or model not in registry
- `FileNotFoundError`: If config file or external hyperparameter file doesn't exist
- `yaml.YAMLError`: If YAML file is invalid

**Example:**

```python
# Experiment config with external hyperparameters
model = ModelFactory.create_from_config("experiments/tide_365_10.yaml")
```

### `ModelFactory.create(name: str, yaml_path: Path | str) -> BaseLitModel`

Create a model instance from YAML configuration (legacy method).

**Parameters:**

- `name`: Registered model name (e.g., "tide", "ealstm")
- `yaml_path`: Path to YAML configuration file

**Returns:**

- Instantiated Lightning module ready for training

**Raises:**

- `ValueError`: If model name not found in registry
- `FileNotFoundError`: If YAML file doesn't exist
- `yaml.YAMLError`: If YAML file is invalid

### `ModelFactory.list_available() -> list[str]`

Get sorted list of all registered model names.

### `ModelFactory.get_config_class(name: str) -> Type[BaseConfig]`

Get the configuration class for a specific model.

**Parameters:**

- `name`: Registered model name

**Returns:**

- Configuration class for the model

**Raises:**

- `ValueError`: If model name not found in registry

### `@register_model(name: str, config_class: Type[BaseConfig] = None)`

Decorator to register a model with the factory.

**Parameters:**

- `name`: Unique identifier for the model
- `config_class`: Configuration class (defaults to BaseConfig if not provided)

**Example:**

```python
@register_model("my_model", config_class=MyModelConfig)
class LitMyModel(BaseLitModel):
    ...
```
