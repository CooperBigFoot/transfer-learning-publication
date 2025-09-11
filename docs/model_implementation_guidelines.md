# Model Implementation Conventions

This document defines the conventions for implementing time series forecasting models in our hydrological analysis framework. These guidelines ensure consistency and maintainability across different model architectures.

## Core Architecture

Every model implementation consists of three components that work together:

### 1. Configuration Class

Each model has a configuration class that extends `BaseConfig`. This serves as the single source of truth for all hyperparameters. No values should be hardcoded in the model implementation itself.

The configuration handles both standard parameters (inherited from `BaseConfig`) and model-specific parameters. Standard parameters include sequence lengths, feature dimensions, learning rate, and other common settings. Model-specific parameters are whatever that particular architecture needs.

### 2. Core Model (nn.Module)

The actual model implementation using PyTorch's `nn.Module`. This contains all the computational logic, layers, and the forward pass. The model should be modular - complex architectures should be broken down into clear sub-components.

### 3. Lightning Module Wrapper

A thin wrapper extending `BaseLitModel` that connects the core model to the training pipeline. Since `BaseLitModel` handles all the standard training logic, this wrapper typically only needs to instantiate the core model and delegate the forward pass.

## Data Interface

### Data Preprocessing

Data may be preprocessed by a `CompositePipeline` before reaching the model. Common transformations include:
- Log transformation for positive-only variables (e.g., streamflow)
- Z-score normalization for standardization
- Per-basin or global transformations

The ModelEvaluator automatically handles inverse transforms during evaluation, so models always work with transformed data during training and inference, while evaluation metrics are computed on the original scale.

### Batch Contract

Models interact with data through the `Batch` contract:

```python
Batch:
    X: torch.Tensor            # [batch_size, input_len, input_size]
    y: torch.Tensor            # [batch_size, output_len]
    static: torch.Tensor       # [batch_size, static_size]
    future: torch.Tensor       # [batch_size, output_len, future_size]
    group_identifiers: List[str]
    input_end_dates: Optional[torch.Tensor]
```

The forward method signature is standardized across all models:

```python
def forward(
    self, 
    x: torch.Tensor,
    static: Optional[torch.Tensor] = None,
    future: Optional[torch.Tensor] = None,
) -> torch.Tensor:  # Returns [batch_size, output_len, 1]
```

Important conventions:

- The target variable is always at position 0 in the input features
- Models should handle missing optional inputs gracefully
- Output shape must be `[batch_size, output_len, 1]` for compatibility

## Configuration Guidelines

Configuration classes should follow this pattern:

```python
class ModelNameConfig(BaseConfig):
    """Configuration for ModelName."""
    
    # List model-specific parameters for serialization
    MODEL_PARAMS = ["param1", "param2", ...]
    
    def __init__(self, 
                 # Standard parameters
                 input_len: int,
                 output_len: int,
                 input_size: int,
                 # Model-specific parameters
                 param1: int = default_value,
                 ...):
        
        # Initialize base config
        super().__init__(...)
        
        # Set model-specific parameters
        self.param1 = param1
        
        # Validate if needed
        self._validate_params()
```

Standard parameters inherited from `BaseConfig`:

- `input_len`: Historical sequence length
- `output_len`: Forecast horizon length  
- `input_size`: Number of input features
- `static_size`: Number of static features (default: 0)
- `future_input_size`: Number of future features (default: max(1, input_size - 1))
- `learning_rate`: Learning rate (default: 1e-5)
- `group_identifier`: Column name for group IDs (default: "gauge_id")
- `use_rev_in`: Whether to use RevIN normalization (default: True)

Model-specific parameters should be defined in the MODEL_PARAMS list and handled in the derived config class.

## Lightning Module Pattern

The Lightning wrapper should be minimal:

```python
class LitModelName(BaseLitModel):
    """Lightning wrapper for ModelName."""
    
    def __init__(self, config: Union[ModelNameConfig, Dict[str, Any]]):
        # Handle dict configs for compatibility
        if isinstance(config, dict):
            config = ModelNameConfig.from_dict(config)
            
        super().__init__(config)
        
        # Create the core model
        self.model = ModelName(config)
    
    def forward(self, x, static=None, future=None):
        # Delegate to core model
        return self.model(x, static, future)
```

The `BaseLitModel` handles:

- Training, validation, and test steps
- Loss computation (MSE)
- Optimizer configuration (Adam with optional ReduceLROnPlateau scheduler)
- Optional RevIN normalization/denormalization
- Test output collection into `ForecastOutput` contract
- Logging of standard metrics

**Note on Inverse Transforms**: The `BaseLitModel` works with transformed data. The `ModelEvaluator` handles applying inverse transforms to the `ForecastOutput` when a pipeline is available, ensuring evaluation metrics are computed on the original scale. Models themselves don't need to handle inverse transformation.

## Feature Terminology

We use consistent terminology throughout:

- **Target**: The variable being predicted (e.g., streamflow)
- **Dynamic features**: Time-varying inputs including the target
- **Static features**: Time-invariant attributes (e.g., catchment properties)
- **Future features**: Known future inputs for the forecast period

## Naming Conventions

For clarity and consistency:

- Use `_len` suffix for sequence lengths (e.g., `input_len`, `output_len`)
- Use `_size` suffix for dimensions (e.g., `hidden_size`, `static_size`)
- Use descriptive prefixes for component-specific sizes (e.g., `encoder_hidden_size`)

## Logging and Metrics

Required metrics (logged automatically by `BaseLitModel`):

- `train_loss`: Training loss
- `val_loss`: Validation loss  
- `test_loss`: Test loss

The validation loss is used for learning rate scheduling and early stopping. Comprehensive evaluation metrics (NSE, KGE, etc.) are computed externally by the evaluation framework.

## Implementation Checklist

When implementing a new model:

1. Create the configuration class extending `BaseConfig`
   - Define MODEL_PARAMS list for model-specific parameters
   - Keep truly universal parameters in BaseConfig
   - Add validation in __init__ if needed
2. Implement the core model (nn.Module) with standard forward signature
   - Accept config object in __init__
   - Forward method: (x, static=None, future=None) -> [batch, output_len, 1]
3. Create the Lightning wrapper extending `BaseLitModel`
   - Handle dict config conversion if needed
   - Instantiate core model
   - Delegate forward to core model
4. Write comprehensive tests
   - Test config initialization and validation
   - Test forward pass with various input combinations
   - Test training/validation/test steps
   - Integration test with realistic data
5. Document tensor shapes in all docstrings
6. Add type hints for all parameters
7. Follow existing naming conventions (e.g., NaiveLastValue, not RepeatLastValues)

## Error Handling

Models should validate critical assumptions but trust the upstream data pipeline for data integrity. Add clear error messages that help identify issues:

```python
if x.size(-1) != self.config.input_size:
    raise ValueError(
        f"Expected input_size={self.config.input_size}, "
        f"got {x.size(-1)}"
    )
```

## Documentation Standards

Use Google-style docstrings with clear descriptions and tensor shapes:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through the model.
    
    Args:
        x: Input sequences. Shape: [batch_size, input_len, input_size]
        
    Returns:
        Predictions. Shape: [batch_size, output_len, 1]
    """
```
