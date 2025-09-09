# Model Factory System

The model factory provides a centralized registry and creation system for all time series forecasting models in the framework. It simplifies model instantiation from YAML configurations and enables easy discovery of available models.

## Quick Start

### Creating a Model

```python
from pathlib import Path
from transfer_learning_publication.models.model_factory import ModelFactory

# Create a model from YAML configuration
model = ModelFactory.create("tide", Path("configs/experiment.yaml"))

# The model is ready for training with PyTorch Lightning
trainer.fit(model, datamodule)
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

The factory expects YAML files with the following structure:

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

# Model-specific parameters (optional)
model:
  hidden_size: 128
  dropout: 0.1
  num_layers: 2
  # ... any other model-specific parameters

# Training parameters (optional)
training:
  learning_rate: 0.001
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

```python
from pathlib import Path
from transfer_learning_publication.models.model_factory import ModelFactory
from transfer_learning_publication.data import LSHDataModule
import lightning as L

# 1. List available models
print("Available models:", ModelFactory.list_available())

# 2. Create model from YAML
config_path = Path("configs/experiment.yaml")
model = ModelFactory.create("tide", config_path)

# 3. Create data module (uses same YAML)
datamodule = LSHDataModule.from_yaml(config_path)

# 4. Train with PyTorch Lightning
trainer = L.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
)
trainer.fit(model, datamodule)

# 5. Test the model
trainer.test(model, datamodule)
```

## Testing the Factory

Run the factory tests to ensure everything is working:

```bash
uv run pytest tests/models/test_model_factory.py -v
```

## Design Rationale

The factory design prioritizes:

1. **Simplicity**: Single file implementation, straightforward API
2. **Self-registration**: Models register themselves via decorator
3. **Compatibility**: Works with existing YAML structure
4. **Extensibility**: Easy to add new models
5. **Type safety**: Proper type hints throughout
6. **Clear errors**: Helpful error messages for common issues

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

### `ModelFactory.create(name: str, yaml_path: Path | str) -> BaseLitModel`

Create a model instance from YAML configuration.

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