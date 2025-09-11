"""Tests for the model factory system."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml

from transfer_learning_publication.models.base.base_config import BaseConfig
from transfer_learning_publication.models.base.base_lit_model import BaseLitModel
from transfer_learning_publication.models.model_factory import (
    _MODELS,
    ModelFactory,
    _extract_config,
    register_model,
)


class TestRegisterModel:
    """Test suite for the register_model decorator."""

    def test_register_model_basic(self):
        """Test basic model registration."""

        # Create a dummy config class
        class DummyConfig(BaseConfig):
            MODEL_PARAMS = ["dummy_param"]

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.dummy_param = kwargs.get("dummy_param", 42)

        # Create and register a dummy model
        @register_model("test_model", config_class=DummyConfig)
        class TestModel(BaseLitModel):
            def __init__(self, config):
                super().__init__(config)

            def forward(self, x, static=None, future=None):
                return x

        # Check registration
        assert "test_model" in _MODELS
        model_class, config_class = _MODELS["test_model"]
        assert model_class == TestModel
        assert config_class == DummyConfig

        # Clean up
        del _MODELS["test_model"]

    def test_register_model_without_config_class(self):
        """Test registration without specifying config_class (should use BaseConfig)."""

        @register_model("test_model_no_config")
        class TestModelNoConfig(BaseLitModel):
            def __init__(self, config):
                super().__init__(config)

            def forward(self, x, static=None, future=None):
                return x

        # Check registration uses BaseConfig
        assert "test_model_no_config" in _MODELS
        model_class, config_class = _MODELS["test_model_no_config"]
        assert model_class == TestModelNoConfig
        assert config_class == BaseConfig

        # Clean up
        del _MODELS["test_model_no_config"]

    def test_duplicate_registration_raises_error(self):
        """Test that duplicate model names raise an error."""

        # First registration should succeed
        @register_model("duplicate_test")
        class TestModel1(BaseLitModel):
            pass

        # Second registration with same name should fail
        with pytest.raises(ValueError, match="Model 'duplicate_test' is already registered"):

            @register_model("duplicate_test")
            class TestModel2(BaseLitModel):
                pass

        # Clean up
        del _MODELS["duplicate_test"]


class TestExtractConfig:
    """Test suite for the _extract_config helper function."""

    def test_extract_standard_config(self):
        """Test extraction of standard configuration parameters."""
        yaml_dict = {
            "sequence": {
                "input_length": 365,
                "output_length": 10,
            },
            "features": {
                "forcing": ["streamflow", "precipitation", "temperature"],
                "static": ["area", "elevation"],
                "future": ["precipitation"],
                "target": "streamflow",
            },
        }

        config = _extract_config(yaml_dict)

        assert config["input_len"] == 365
        assert config["output_len"] == 10
        assert config["input_size"] == 3
        assert config["static_size"] == 2
        assert config["future_input_size"] == 1
        # Note: target is not included in config as it's not a model parameter

    def test_extract_model_specific_params(self):
        """Test extraction of model-specific parameters."""
        yaml_dict = {
            "sequence": {
                "input_length": 100,
                "output_length": 5,
            },
            "features": {
                "forcing": ["streamflow"],
            },
            "model": {
                "hidden_size": 128,
                "dropout": 0.2,
                "num_layers": 3,
            },
        }

        config = _extract_config(yaml_dict)

        assert config["hidden_size"] == 128
        assert config["dropout"] == 0.2
        assert config["num_layers"] == 3

    def test_extract_training_params(self):
        """Test extraction of training parameters."""
        yaml_dict = {
            "sequence": {
                "input_length": 100,
                "output_length": 5,
            },
            "features": {
                "forcing": ["streamflow"],
            },
            "training": {
                "learning_rate": 0.001,
            },
        }

        config = _extract_config(yaml_dict)

        assert config["learning_rate"] == 0.001

    def test_extract_with_missing_sections(self):
        """Test extraction handles missing sections gracefully."""
        yaml_dict = {
            "sequence": {
                "input_length": 100,
                "output_length": 5,
            },
            # Missing features section
        }

        config = _extract_config(yaml_dict)

        # Should have sequence params but not feature params
        assert config["input_len"] == 100
        assert config["output_len"] == 5
        assert "input_size" not in config
        assert "static_size" not in config

    def test_model_params_dont_overwrite_standard(self):
        """Test that model params don't overwrite extracted standard params."""
        yaml_dict = {
            "sequence": {
                "input_length": 365,
                "output_length": 10,
            },
            "model": {
                "input_len": 999,  # Should not overwrite
                "hidden_size": 128,
            },
        }

        config = _extract_config(yaml_dict)

        # Standard param should not be overwritten
        assert config["input_len"] == 365
        assert config["hidden_size"] == 128


class TestCreateFromConfig:
    """Test suite for ModelFactory.create_from_config method."""

    def test_create_from_config_basic(self):
        """Test creating a model from experiment config."""
        yaml_content = {
            "sequence": {
                "input_length": 30,
                "output_length": 7,
            },
            "features": {
                "forcing": ["streamflow", "precipitation", "temperature"],
                "static": ["area"],
                "target": "streamflow",
            },
            "data_preparation": {
                "is_autoregressive": True,
                "include_dates": False,
            },
            "model": {
                "type": "tide",
                "overrides": {
                    "hidden_size": 64,
                    "dropout": 0.1,
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            # Create model
            model = ModelFactory.create_from_config(yaml_path)

            # Check model is created
            assert model is not None
            assert hasattr(model, "model")
            assert hasattr(model, "config")

            # Check config values
            assert model.config.input_len == 30
            assert model.config.output_len == 7
            assert model.config.input_size == 3
            assert model.config.static_size == 1
            assert model.config.hidden_size == 64
            assert model.config.dropout == 0.1

            # Test forward pass
            x = torch.randn(2, 30, 3)
            static = torch.randn(2, 1)
            output = model(x, static)
            assert output.shape == (2, 7, 1)

        finally:
            yaml_path.unlink()

    def test_create_from_config_with_external_file(self):
        """Test loading external hyperparameter file."""
        # Create external hyperparameter file
        hyperparams = {
            "hidden_size": 128,
            "dropout": 0.2,
            "num_encoder_layers": 3,
            "learning_rate": 0.001,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix="_hyperparams.yaml", delete=False) as f:
            yaml.dump(hyperparams, f)
            hyperparam_path = Path(f.name)

        # Create experiment config referencing external file
        yaml_content = {
            "sequence": {
                "input_length": 20,
                "output_length": 5,
            },
            "features": {
                "forcing": ["streamflow", "precipitation"],
                "target": "streamflow",
            },
            "model": {
                "type": "tide",
                "config_file": str(hyperparam_path),  # Reference external file
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            model = ModelFactory.create_from_config(yaml_path)

            # Check that hyperparameters from external file were loaded
            assert model.config.hidden_size == 128
            assert model.config.dropout == 0.2
            assert model.config.num_encoder_layers == 3
            assert model.config.learning_rate == 0.001

            # Check data-derived params still work
            assert model.config.input_len == 20
            assert model.config.output_len == 5

        finally:
            yaml_path.unlink()
            hyperparam_path.unlink()

    def test_create_from_config_overrides_priority(self):
        """Test that overrides take precedence over external config."""
        # Create external hyperparameter file
        hyperparams = {
            "hidden_size": 128,
            "dropout": 0.2,
            "learning_rate": 0.001,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix="_hyperparams.yaml", delete=False) as f:
            yaml.dump(hyperparams, f)
            hyperparam_path = Path(f.name)

        # Create experiment config with overrides
        yaml_content = {
            "sequence": {
                "input_length": 15,
                "output_length": 3,
            },
            "features": {
                "forcing": ["streamflow"],
                "target": "streamflow",
            },
            "model": {
                "type": "tide",
                "config_file": str(hyperparam_path),
                "overrides": {
                    "dropout": 0.3,  # Override external value
                    "new_param": 42,  # Add new parameter
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            model = ModelFactory.create_from_config(yaml_path)

            # Check override took precedence
            assert model.config.dropout == 0.3

            # Check non-overridden param from external file
            assert model.config.hidden_size == 128
            assert model.config.learning_rate == 0.001

            # New param might not be in config if not in MODEL_PARAMS
            # But it should have been passed through

        finally:
            yaml_path.unlink()
            hyperparam_path.unlink()

    def test_create_from_config_missing_model_section(self):
        """Test error when model section is missing."""
        yaml_content = {
            "sequence": {
                "input_length": 10,
                "output_length": 1,
            },
            "features": {
                "forcing": ["streamflow"],
                "target": "streamflow",
            },
            # Missing model section
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Missing 'model' section"):
                ModelFactory.create_from_config(yaml_path)
        finally:
            yaml_path.unlink()

    def test_create_from_config_missing_model_type(self):
        """Test error when model.type is missing."""
        yaml_content = {
            "sequence": {
                "input_length": 10,
                "output_length": 1,
            },
            "features": {
                "forcing": ["streamflow"],
                "target": "streamflow",
            },
            "model": {
                # Missing type field
                "overrides": {
                    "hidden_size": 64,
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Missing 'model.type'"):
                ModelFactory.create_from_config(yaml_path)
        finally:
            yaml_path.unlink()

    def test_create_from_config_invalid_model_type(self):
        """Test error with invalid model type."""
        yaml_content = {
            "sequence": {
                "input_length": 10,
                "output_length": 1,
            },
            "features": {
                "forcing": ["streamflow"],
                "target": "streamflow",
            },
            "model": {
                "type": "nonexistent_model",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Model 'nonexistent_model' not found"):
                ModelFactory.create_from_config(yaml_path)
        finally:
            yaml_path.unlink()

    def test_create_from_config_missing_external_file(self):
        """Test error when external config file doesn't exist."""
        yaml_content = {
            "sequence": {
                "input_length": 10,
                "output_length": 1,
            },
            "features": {
                "forcing": ["streamflow"],
                "target": "streamflow",
            },
            "model": {
                "type": "tide",
                "config_file": "/nonexistent/file.yaml",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            with pytest.raises(FileNotFoundError, match="Hyperparameter file not found"):
                ModelFactory.create_from_config(yaml_path)
        finally:
            yaml_path.unlink()

    def test_create_from_config_relative_path(self):
        """Test that relative paths work for config_file."""
        # Create a temporary directory structure

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create subdirectories
            configs_dir = tmpdir / "configs"
            configs_dir.mkdir()
            models_dir = configs_dir / "models"
            models_dir.mkdir()

            # Create hyperparameter file
            hyperparams = {
                "hidden_size": 256,
                "dropout": 0.15,
            }
            hyperparam_path = models_dir / "tide_params.yaml"
            with open(hyperparam_path, "w") as f:
                yaml.dump(hyperparams, f)

            # Create experiment config with relative path
            yaml_content = {
                "sequence": {
                    "input_length": 10,
                    "output_length": 1,
                },
                "features": {
                    "forcing": ["streamflow"],
                    "target": "streamflow",
                },
                "model": {
                    "type": "tide",
                    "config_file": "models/tide_params.yaml",  # Relative path
                },
            }

            config_path = configs_dir / "experiment.yaml"
            with open(config_path, "w") as f:
                yaml.dump(yaml_content, f)

            # Create model
            model = ModelFactory.create_from_config(config_path)

            # Check hyperparameters were loaded
            assert model.config.hidden_size == 256
            assert model.config.dropout == 0.15

    def test_create_from_config_delegates_to_create_from_dict(self):
        """Test that create_from_config delegates to create_from_dict."""
        yaml_content = {
            "sequence": {
                "input_length": 10,
                "output_length": 2,
            },
            "features": {
                "forcing": ["streamflow"],
                "target": "streamflow",
            },
            "model": {
                "type": "tide",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            with patch.object(ModelFactory, "create_from_dict") as mock_create_from_dict:
                mock_model = MagicMock(spec=BaseLitModel)
                mock_create_from_dict.return_value = mock_model

                # Call create_from_config
                model = ModelFactory.create_from_config(yaml_path)

                # Check that create_from_dict was called
                mock_create_from_dict.assert_called_once()
                call_args = mock_create_from_dict.call_args
                assert call_args[0][0] == "tide"  # First positional arg is model name
                assert isinstance(call_args[0][1], dict)  # Second is config dict

                # Check model was returned
                assert model == mock_model

        finally:
            yaml_path.unlink()


class TestCreateFromDict:
    """Test suite for the create_from_dict method."""

    def test_create_from_dict_basic(self):
        """Test creating a model from a configuration dictionary."""
        config_dict = {
            "input_len": 30,
            "output_len": 7,
            "input_size": 3,
            "static_size": 1,
            "hidden_size": 64,
            "dropout": 0.1,
        }

        # Create model
        model = ModelFactory.create_from_dict("tide", config_dict)

        # Check model is created
        assert model is not None
        assert hasattr(model, "model")
        assert hasattr(model, "config")

        # Check config values
        assert model.config.input_len == 30
        assert model.config.output_len == 7
        assert model.config.input_size == 3
        assert model.config.static_size == 1
        assert model.config.hidden_size == 64
        assert model.config.dropout == 0.1

        # Test forward pass
        x = torch.randn(2, 30, 3)
        static = torch.randn(2, 1)
        output = model(x, static)
        assert output.shape == (2, 7, 1)

    def test_create_from_dict_filters_params(self):
        """Test that create_from_dict filters out unaccepted parameters."""
        config_dict = {
            "input_len": 20,
            "output_len": 5,
            "input_size": 2,
            "hidden_size": 32,
            "invalid_param": "should_be_filtered",
            "another_invalid": 123,
        }

        # Create model - should not raise error despite invalid params
        model = ModelFactory.create_from_dict("tide", config_dict)

        # Check valid params are set
        assert model.config.input_len == 20
        assert model.config.output_len == 5
        assert model.config.input_size == 2
        assert model.config.hidden_size == 32

        # Invalid params should not be in config
        assert not hasattr(model.config, "invalid_param")
        assert not hasattr(model.config, "another_invalid")

    def test_create_from_dict_invalid_model(self):
        """Test error when model name is not in registry."""
        config_dict = {"input_len": 10, "output_len": 1}

        with pytest.raises(ValueError, match="Model 'unknown_model' not found"):
            ModelFactory.create_from_dict("unknown_model", config_dict)

    def test_create_from_dict_multiple_models(self):
        """Test creating different models with same config."""
        config_dict = {
            "input_len": 15,
            "output_len": 3,
            "input_size": 2,
            "static_size": 0,
        }

        # Create different models
        tide_model = ModelFactory.create_from_dict("tide", config_dict)
        ealstm_model = ModelFactory.create_from_dict("ealstm", config_dict)

        # Check both are created
        assert tide_model is not None
        assert ealstm_model is not None

        # Check they are different types
        assert type(tide_model).__name__ == "LitTiDE"
        assert type(ealstm_model).__name__ == "LitEALSTM"

        # Check both have same config values
        for model in [tide_model, ealstm_model]:
            assert model.config.input_len == 15
            assert model.config.output_len == 3
            assert model.config.input_size == 2


class TestCreateFromCheckpoint:
    """Test suite for the create_from_checkpoint method."""

    def test_create_from_checkpoint_basic(self):
        """Test loading a model from checkpoint."""
        # Create a temporary checkpoint file
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            checkpoint_path = Path(f.name)

        try:
            # Mock the model class directly in the registry
            from transfer_learning_publication.models.tide.lightning import LitTiDE
            
            with patch.object(LitTiDE, "load_from_checkpoint") as mock_load:
                mock_model_instance = MagicMock(spec=BaseLitModel)
                mock_load.return_value = mock_model_instance

                # Load model from checkpoint
                model = ModelFactory.create_from_checkpoint("tide", checkpoint_path)

                # Check that load_from_checkpoint was called with correct path
                mock_load.assert_called_once_with(checkpoint_path)

                # Check model was returned
                assert model == mock_model_instance

        finally:
            checkpoint_path.unlink()

    def test_create_from_checkpoint_missing_file(self):
        """Test error when checkpoint file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            ModelFactory.create_from_checkpoint("tide", Path("/nonexistent/checkpoint.ckpt"))

    def test_create_from_checkpoint_invalid_model(self):
        """Test error when model name is not in registry."""
        # Create a temporary checkpoint file
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            checkpoint_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Model 'unknown_model' not found"):
                ModelFactory.create_from_checkpoint("unknown_model", checkpoint_path)
        finally:
            checkpoint_path.unlink()

    def test_create_from_checkpoint_incompatible(self):
        """Test error when checkpoint is incompatible with model class."""
        # Create a temporary checkpoint file
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            checkpoint_path = Path(f.name)
            # Write some dummy data to make it a non-empty file
            torch.save({"state_dict": {"dummy": torch.tensor([1.0])}}, f.name)

        try:
            # This should fail because the checkpoint doesn't have the right structure
            with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
                ModelFactory.create_from_checkpoint("tide", checkpoint_path)
        finally:
            checkpoint_path.unlink()

    def test_create_from_checkpoint_string_path(self):
        """Test that string paths work as well as Path objects."""
        # Create a temporary checkpoint file
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            checkpoint_path = f.name  # String path

        try:
            # Mock the model class directly in the registry
            from transfer_learning_publication.models.tide.lightning import LitTiDE
            
            with patch.object(LitTiDE, "load_from_checkpoint") as mock_load:
                mock_model_instance = MagicMock(spec=BaseLitModel)
                mock_load.return_value = mock_model_instance

                # Load model from checkpoint using string path
                model = ModelFactory.create_from_checkpoint("tide", checkpoint_path)

                # Check model was returned
                assert model == mock_model_instance

        finally:
            Path(checkpoint_path).unlink()


class TestGetModelClass:
    """Test suite for the get_model_class method."""

    def test_get_model_class_valid(self):
        """Test getting model class for valid model names."""
        from transfer_learning_publication.models.tide.lightning import LitTiDE
        from transfer_learning_publication.models.ealstm.lightning import LitEALSTM

        # Test getting TiDE model class
        tide_class = ModelFactory.get_model_class("tide")
        assert tide_class == LitTiDE

        # Test getting EALSTM model class
        ealstm_class = ModelFactory.get_model_class("ealstm")
        assert ealstm_class == LitEALSTM

    def test_get_model_class_invalid(self):
        """Test error when model name is not in registry."""
        with pytest.raises(ValueError, match="Model 'unknown_model' not found"):
            ModelFactory.get_model_class("unknown_model")

    def test_get_model_class_all_registered(self):
        """Test that all registered models can have their class retrieved."""
        available_models = ModelFactory.list_available()

        for model_name in available_models:
            model_class = ModelFactory.get_model_class(model_name)
            assert model_class is not None
            assert issubclass(model_class, BaseLitModel)


class TestModelFactory:
    """Test suite for ModelFactory class methods."""

    def test_list_available_models(self):
        """Test listing available models."""
        available = ModelFactory.list_available()

        # Check that registered models are listed
        assert "tide" in available
        assert "ealstm" in available
        assert "tsmixer" in available
        assert "tft" in available
        assert "naive_last_value" in available

        # Check list is sorted
        assert available == sorted(available)

    def test_get_config_class(self):
        """Test getting config class for a model."""
        from transfer_learning_publication.models.tide.config import TiDEConfig

        config_class = ModelFactory.get_config_class("tide")
        assert config_class == TiDEConfig

    def test_get_config_class_unknown_model(self):
        """Test getting config class for unknown model raises error."""
        with pytest.raises(ValueError, match="Model 'unknown' not found"):
            ModelFactory.get_config_class("unknown")

    def test_create_model_from_yaml(self):
        """Test creating a model from YAML configuration."""
        # Create temporary YAML file
        yaml_content = {
            "sequence": {
                "input_length": 30,
                "output_length": 7,
            },
            "features": {
                "forcing": ["streamflow", "precipitation", "temperature"],
                "static": ["area"],
                "target": "streamflow",
            },
            "model": {
                "hidden_size": 64,
                "dropout": 0.1,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            # Create model
            model = ModelFactory.create("tide", yaml_path)

            # Check model is created
            assert model is not None
            assert hasattr(model, "model")
            assert hasattr(model, "config")

            # Check config values
            assert model.config.input_len == 30
            assert model.config.output_len == 7
            assert model.config.input_size == 3
            assert model.config.static_size == 1
            assert model.config.hidden_size == 64

            # Test forward pass
            x = torch.randn(2, 30, 3)
            static = torch.randn(2, 1)
            output = model(x, static)
            assert output.shape == (2, 7, 1)

        finally:
            # Clean up
            yaml_path.unlink()

    def test_create_delegates_to_create_from_dict(self):
        """Test that create method delegates to create_from_dict."""
        yaml_content = {
            "sequence": {
                "input_length": 10,
                "output_length": 2,
            },
            "features": {
                "forcing": ["streamflow"],
                "target": "streamflow",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            with patch.object(ModelFactory, "create_from_dict") as mock_create_from_dict:
                mock_model = MagicMock(spec=BaseLitModel)
                mock_create_from_dict.return_value = mock_model

                # Call create
                model = ModelFactory.create("tide", yaml_path)

                # Check that create_from_dict was called
                mock_create_from_dict.assert_called_once()
                call_args = mock_create_from_dict.call_args
                assert call_args[0][0] == "tide"  # First positional arg is model name
                assert isinstance(call_args[0][1], dict)  # Second is config dict

                # Check model was returned
                assert model == mock_model

        finally:
            yaml_path.unlink()

    def test_create_model_unknown_name(self):
        """Test creating model with unknown name raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.dump({"test": "data"}, f)
            yaml_path = Path(f.name)

            with pytest.raises(ValueError, match="Model 'unknown_model' not found"):
                ModelFactory.create("unknown_model", yaml_path)

    def test_create_model_missing_yaml(self):
        """Test creating model with missing YAML file raises error."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            ModelFactory.create("tide", Path("/nonexistent/file.yaml"))

    def test_create_model_invalid_yaml(self):
        """Test creating model with invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            yaml_path = Path(f.name)

        try:
            with pytest.raises(yaml.YAMLError):
                ModelFactory.create("tide", yaml_path)
        finally:
            yaml_path.unlink()

    def test_create_multiple_models(self):
        """Test creating multiple different models."""
        yaml_content = {
            "sequence": {
                "input_length": 20,
                "output_length": 5,
            },
            "features": {
                "forcing": ["streamflow", "temperature"],
                "static": [],
                "target": "streamflow",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            # Create different models with same config
            tide_model = ModelFactory.create("tide", yaml_path)
            ealstm_model = ModelFactory.create("ealstm", yaml_path)
            tsmixer_model = ModelFactory.create("tsmixer", yaml_path)

            # Check all models are created
            assert tide_model is not None
            assert ealstm_model is not None
            assert tsmixer_model is not None

            # Check they are different types
            assert type(tide_model).__name__ == "LitTiDE"
            assert type(ealstm_model).__name__ == "LitEALSTM"
            assert type(tsmixer_model).__name__ == "LitTSMixer"

            # Check all have same config values
            for model in [tide_model, ealstm_model, tsmixer_model]:
                assert model.config.input_len == 20
                assert model.config.output_len == 5
                assert model.config.input_size == 2

        finally:
            yaml_path.unlink()

    def test_create_model_with_string_path(self):
        """Test that string paths work as well as Path objects."""
        yaml_content = {
            "sequence": {
                "input_length": 10,
                "output_length": 1,
            },
            "features": {
                "forcing": ["streamflow"],
                "target": "streamflow",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = f.name  # String path

        try:
            # Create model with string path
            model = ModelFactory.create("naive_last_value", yaml_path)

            assert model is not None
            assert model.config.input_len == 10
            assert model.config.output_len == 1

        finally:
            Path(yaml_path).unlink()


class TestIntegrationWithRealModels:
    """Integration tests with actual registered models."""

    def test_all_registered_models_can_be_created(self):
        """Test that all registered models can be instantiated."""
        yaml_content = {
            "sequence": {
                "input_length": 15,
                "output_length": 3,
            },
            "features": {
                "forcing": ["streamflow", "precipitation"],
                "static": ["area"],
                "future": ["precipitation"],
                "target": "streamflow",
            },
            "model": {
                "hidden_size": 32,
                "dropout": 0.1,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            available_models = ModelFactory.list_available()

            for model_name in available_models:
                # Create model
                model = ModelFactory.create(model_name, yaml_path)

                # Basic checks
                assert model is not None
                assert hasattr(model, "config")
                assert hasattr(model, "model")

                # Test forward pass
                x = torch.randn(2, 15, 2)
                static = torch.randn(2, 1)
                future = torch.randn(2, 3, 1)

                output = model(x, static, future)
                assert output.shape == (2, 3, 1)

        finally:
            yaml_path.unlink()

    def test_models_work_with_minimal_config(self):
        """Test models work with minimal configuration."""
        yaml_content = {
            "sequence": {
                "input_length": 10,
                "output_length": 2,
            },
            "features": {
                "forcing": ["streamflow"],
                "target": "streamflow",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            # Test a few models with minimal config
            for model_name in ["tide", "naive_last_value"]:
                model = ModelFactory.create(model_name, yaml_path)

                # Test forward pass
                x = torch.randn(2, 10, 1)
                output = model(x)
                assert output.shape == (2, 2, 1)

        finally:
            yaml_path.unlink()
