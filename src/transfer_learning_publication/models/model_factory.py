from pathlib import Path
from typing import Any

import yaml

from .base.base_config import BaseConfig
from .base.base_lit_model import BaseLitModel

_MODELS: dict[str, tuple[type[BaseLitModel], type[BaseConfig]]] = {}
_MODELS_REGISTERED = False


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
    def create_from_dict(name: str, config_dict: dict[str, Any]) -> BaseLitModel:
        """Create a model instance from a configuration dictionary.

        This is the core creation method that other methods delegate to.

        Args:
            name: Registered model name (e.g., "tide", "ealstm")
            config_dict: Dictionary of configuration parameters

        Returns:
            Instantiated Lightning module ready for training

        Raises:
            ValueError: If model name not found in registry

        Example:
            config = {"input_len": 365, "output_len": 10, ...}
            model = ModelFactory.create_from_dict("tide", config)
        """
        # Ensure models are registered
        _ensure_models_registered()

        # Validate model exists
        if name not in _MODELS:
            available = list(_MODELS.keys())
            raise ValueError(f"Model '{name}' not found in registry. Available models: {sorted(available)}")

        # Get model and config classes
        model_class, config_class = _MODELS[name]

        # Filter config_dict to only include parameters the config class accepts
        accepted_params = set(config_class.STANDARD_PARAMS + config_class.MODEL_PARAMS)
        filtered_config = {key: value for key, value in config_dict.items() if key in accepted_params}

        # Create model instance
        # The model's __init__ will handle conversion from dict to config object
        model = model_class(filtered_config)

        return model

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

        # Delegate to create_from_dict
        return ModelFactory.create_from_dict(name, config_dict)

    @staticmethod
    def create_from_config(config_path: Path | str) -> BaseLitModel:
        """Create a model instance from an experiment configuration file.

        This method loads a complete experiment configuration, extracts the model type,
        optionally loads external hyperparameters, applies overrides, and creates the model.

        Args:
            config_path: Path to experiment configuration file

        Returns:
            Instantiated Lightning module ready for training

        Raises:
            ValueError: If model.type not found in config or model not in registry
            FileNotFoundError: If config file or external hyperparameter file doesn't exist
            yaml.YAMLError: If YAML file is invalid

        Example:
            # Config with external hyperparameters
            model = ModelFactory.create_from_config("experiments/tide_365_10.yaml")

            # The config file should have:
            # model:
            #   type: "tide"
            #   config_file: "configs/models/tide_best.yaml"  # optional
            #   overrides:  # optional
            #     learning_rate: 0.0001
        """
        # Convert to Path if string
        config_path = Path(config_path)

        # Check file exists
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load experiment configuration
        with open(config_path) as f:
            experiment_config = yaml.safe_load(f)

        # Validate model section exists
        if "model" not in experiment_config:
            raise ValueError("Missing 'model' section in configuration file")

        model_section = experiment_config["model"]

        # Validate model type exists
        if "type" not in model_section:
            raise ValueError("Missing 'model.type' in configuration file")

        model_type = model_section["type"]

        # Start with empty config dict
        config_dict = {}

        # Load external hyperparameters if specified
        if "config_file" in model_section:
            hyperparameter_path = Path(model_section["config_file"])

            # Handle relative paths (relative to the experiment config file)
            if not hyperparameter_path.is_absolute():
                hyperparameter_path = config_path.parent / hyperparameter_path

            if not hyperparameter_path.exists():
                raise FileNotFoundError(f"Hyperparameter file not found: {hyperparameter_path}")

            with open(hyperparameter_path) as f:
                external_config = yaml.safe_load(f)

            # Add external hyperparameters to config
            if external_config:
                config_dict.update(external_config)

        # Apply overrides if specified
        if "overrides" in model_section:
            overrides = model_section["overrides"]
            if overrides:
                config_dict.update(overrides)

        # Extract data-derived parameters from experiment config
        data_derived_params = _extract_config(experiment_config)

        # Merge data-derived parameters (these should take precedence)
        # But don't overwrite model-specific parameters that were explicitly set
        for key, value in data_derived_params.items():
            if key not in config_dict:  # Only add if not already set by model config/overrides
                config_dict[key] = value

        # Create model using create_from_dict
        return ModelFactory.create_from_dict(model_type, config_dict)

    @staticmethod
    def create_from_checkpoint(model_name: str, checkpoint_path: Path | str) -> BaseLitModel:
        """Load a model from a PyTorch Lightning checkpoint.

        Uses the model registry to find the correct model class and leverages
        Lightning's built-in checkpoint loading functionality.

        Args:
            model_name: Registered model name (e.g., "tide", "ealstm")
            checkpoint_path: Path to the checkpoint file

        Returns:
            Model loaded with weights and configuration from checkpoint

        Raises:
            ValueError: If model name not found in registry
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint is incompatible with model class

        Example:
            model = ModelFactory.create_from_checkpoint(
                "tide",
                "checkpoints/tide_best.ckpt"
            )
        """
        # Ensure models are registered
        _ensure_models_registered()

        # Validate model exists
        if model_name not in _MODELS:
            available = list(_MODELS.keys())
            raise ValueError(f"Model '{model_name}' not found in registry. Available models: {sorted(available)}")

        # Convert to Path if string
        checkpoint_path = Path(checkpoint_path)

        # Check checkpoint exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Get model class from registry
        model_class, _ = _MODELS[model_name]

        # Use Lightning's built-in checkpoint loading
        # This handles everything: hyperparameters, weights, optimizer state, etc.
        try:
            model = model_class.load_from_checkpoint(checkpoint_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint for model '{model_name}'. "
                f"The checkpoint may be incompatible with the model class. "
                f"Error: {e}"
            )

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
        # Ensure models are registered
        _ensure_models_registered()
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
        # Ensure models are registered
        _ensure_models_registered()

        if name not in _MODELS:
            available = list(_MODELS.keys())
            raise ValueError(f"Model '{name}' not found in registry. Available models: {sorted(available)}")

        _, config_class = _MODELS[name]
        return config_class

    @staticmethod
    def get_model_class(name: str) -> type[BaseLitModel]:
        """Get the Lightning module class for a specific model.

        Args:
            name: Registered model name

        Returns:
            Lightning module class for the model

        Raises:
            ValueError: If model name not found in registry
        """
        # Ensure models are registered
        _ensure_models_registered()

        if name not in _MODELS:
            available = list(_MODELS.keys())
            raise ValueError(f"Model '{name}' not found in registry. Available models: {sorted(available)}")

        model_class, _ = _MODELS[name]
        return model_class


def _ensure_models_registered():
    """Ensure all models are registered by importing them.

    This function is called lazily to avoid circular imports.
    Models self-register via the @register_model decorator when imported.
    """
    global _MODELS_REGISTERED
    if _MODELS_REGISTERED:
        return

    # Import individual models - they will self-register via decorator
    from .dummy.lightning import LitNaiveLastValue  # noqa: F401
    from .ealstm.lightning import LitEALSTM  # noqa: F401
    from .tft.lightning import LitTFT  # noqa: F401
    from .tide.lightning import LitTiDE  # noqa: F401
    from .tsmixer.lightning import LitTSMixer  # noqa: F401

    _MODELS_REGISTERED = True


# Note: When adding new models, remember to:
# 1. Add the @register_model decorator to the Lightning module
# 2. Add the import in _ensure_models_registered() function above
# 3. The model will automatically appear in the factory

__all__ = [
    "ModelFactory",
    "register_model",
]
