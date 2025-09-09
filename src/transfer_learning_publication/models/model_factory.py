from pathlib import Path
from typing import Any

import yaml

from .base.base_config import BaseConfig
from .base.base_lit_model import BaseLitModel

_MODELS: dict[str, tuple[type[BaseLitModel], type[BaseConfig]]] = {}


def register_model(name: str, config_class: type[BaseConfig] | None = None):
    """Decorator to register a model with the factory.

    Models self-register by applying this decorator to their Lightning module class.
    The decorator stores both the model class and its associated configuration class
    in the global registry.

    Args:
        name: Unique identifier for the model (e.g., "tide", "ealstm")
        config_class: Configuration class for this model. If not provided,
                     defaults to BaseConfig.

    Returns:
        The decorated class unchanged.

    Example:
        @register_model("tide", config_class=TiDEConfig)
        class LitTiDE(BaseLitModel):
            ...
    """

    def decorator(model_class: type[BaseLitModel]) -> type[BaseLitModel]:
        """Inner decorator that registers the model."""
        if name in _MODELS:
            raise ValueError(
                f"Model '{name}' is already registered. Choose a different name or remove the duplicate registration."
            )

        # Use BaseConfig if no specific config class provided
        cfg_class = config_class or BaseConfig
        _MODELS[name] = (model_class, cfg_class)

        return model_class

    return decorator


def _extract_config(yaml_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract and standardize configuration from YAML structure.

    Maps the hierarchical YAML structure to a flat configuration dictionary
    expected by model config classes. Handles extraction of standard parameters
    from their respective sections and merges with model-specific parameters.

    Args:
        yaml_dict: Loaded YAML configuration as a nested dictionary

    Returns:
        Flat dictionary with standardized parameter names
    """
    config = {}

    # Extract sequence parameters
    if "sequence" in yaml_dict:
        seq = yaml_dict["sequence"]
        config["input_len"] = seq.get("input_length", seq.get("input_len"))
        config["output_len"] = seq.get("output_length", seq.get("output_len"))

    # Extract feature dimensions
    if "features" in yaml_dict:
        features = yaml_dict["features"]

        # Calculate input size from forcing features
        forcing = features.get("forcing", [])
        config["input_size"] = len(forcing) if forcing else 1

        # Calculate static size
        static = features.get("static", [])
        config["static_size"] = len(static) if static else 0

        # Calculate future input size
        future = features.get("future", [])
        config["future_input_size"] = len(future) if future else None

        # Note: target is not included in config as it's not a model parameter
        # It's only used for data loading, not model configuration

    # Extract model-specific parameters
    if "model" in yaml_dict:
        model_params = yaml_dict["model"]
        # Merge model-specific params, avoiding overwrites of standard params
        for key, value in model_params.items():
            if key not in config:  # Don't overwrite extracted standard params
                config[key] = value

    # Extract training parameters if present
    if "training" in yaml_dict:
        training = yaml_dict["training"]
        if "learning_rate" in training:
            config["learning_rate"] = training["learning_rate"]

    return config


class ModelFactory:
    """Factory class for creating models from configurations.

    This class provides static methods for model creation and discovery.
    It uses the module-level registry populated by the @register_model decorator.
    """

    @staticmethod
    def create(name: str, yaml_path: Path | str) -> BaseLitModel:
        """Create a model instance from a YAML configuration file.

        This method loads a YAML configuration, extracts and standardizes the
        parameters, and creates an instance of the requested model.

        Args:
            name: Registered model name (e.g., "tide", "ealstm")
            yaml_path: Path to YAML configuration file

        Returns:
            Instantiated Lightning module ready for training

        Raises:
            ValueError: If model name not found in registry
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML file is invalid

        Example:
            model = ModelFactory.create("tide", Path("configs/experiment.yaml"))
        """
        # Validate model exists
        if name not in _MODELS:
            available = list(_MODELS.keys())
            raise ValueError(f"Model '{name}' not found in registry. Available models: {sorted(available)}")

        # Convert to Path if string
        yaml_path = Path(yaml_path)

        # Check file exists
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        # Load YAML configuration
        with open(yaml_path) as f:
            yaml_dict = yaml.safe_load(f)

        # Extract and standardize configuration
        config_dict = _extract_config(yaml_dict)

        # Get model and config classes
        model_class, config_class = _MODELS[name]

        # Filter config_dict to only include parameters the config class accepts
        # This prevents errors when models don't accept certain parameters
        accepted_params = set(config_class.STANDARD_PARAMS + config_class.MODEL_PARAMS)
        filtered_config = {key: value for key, value in config_dict.items() if key in accepted_params}

        # Create model instance
        # The model's __init__ will handle conversion from dict to config object
        model = model_class(filtered_config)

        return model

    @staticmethod
    def list_available() -> list[str]:
        """Get list of all registered model names.

        Returns:
            Sorted list of model names available in the registry

        Example:
            available_models = ModelFactory.list_available()
            # Returns: ["dummy", "ealstm", "tft", "tide", "tsmixer", ...]
        """
        return sorted(_MODELS.keys())

    @staticmethod
    def get_config_class(name: str) -> type[BaseConfig]:
        """Get the configuration class for a specific model.

        Args:
            name: Registered model name

        Returns:
            Configuration class for the model

        Raises:
            ValueError: If model name not found in registry
        """
        if name not in _MODELS:
            available = list(_MODELS.keys())
            raise ValueError(f"Model '{name}' not found in registry. Available models: {sorted(available)}")

        _, config_class = _MODELS[name]
        return config_class


# Import all models to trigger registration
# This section must be at the bottom after the decorator is defined
# When adding a new model, add its import here

# Import individual models - they will self-register via decorator
# noqa: E402, F401 - Late imports needed for registration
from .dummy.lightning import LitNaiveLastValue  # noqa: E402, F401
from .ealstm.lightning import LitEALSTM  # noqa: E402, F401
from .tft.lightning import LitTFT  # noqa: E402, F401
from .tide.lightning import LitTiDE  # noqa: E402, F401
from .tsmixer.lightning import LitTSMixer  # noqa: E402, F401

# Note: When adding new models, remember to:
# 1. Add the @register_model decorator to the Lightning module
# 2. Import the model here
# 3. The model will automatically appear in the factory

__all__ = [
    "ModelFactory",
    "register_model",
]
