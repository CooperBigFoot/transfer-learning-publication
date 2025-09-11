"""Tests for ModelEvaluator class."""

import json
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch

from transfer_learning_publication.contracts import EvaluationResults, ForecastOutput
from transfer_learning_publication.data import LSHDataModule
from transfer_learning_publication.models.base import BaseLitModel
from transfer_learning_publication.models.model_evaluator import ModelEvaluator


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    def test_init_empty(self):
        """Test initialization with no models."""
        evaluator = ModelEvaluator()

        assert evaluator.models_and_datamodules == {}
        assert evaluator.trainer_kwargs == {"accelerator": "auto", "devices": 1}

    def test_init_with_models(self):
        """Test initialization with models."""
        model1 = MagicMock(spec=BaseLitModel)
        model2 = MagicMock(spec=BaseLitModel)
        dm1 = MagicMock(spec=LSHDataModule)
        dm2 = MagicMock(spec=LSHDataModule)

        evaluator = ModelEvaluator(
            models_and_datamodules={
                "model1": (model1, dm1),
                "model2": (model2, dm2),
            },
            trainer_kwargs={"accelerator": "cpu", "devices": 1},
        )

        assert len(evaluator.models_and_datamodules) == 2
        assert evaluator.trainer_kwargs["accelerator"] == "cpu"

    def test_add_models(self):
        """Test adding models to evaluator."""
        evaluator = ModelEvaluator()

        model1 = MagicMock(spec=BaseLitModel)
        dm1 = MagicMock(spec=LSHDataModule)

        evaluator.add_models({"model1": (model1, dm1)})

        assert "model1" in evaluator.models_and_datamodules
        assert evaluator.models_and_datamodules["model1"] == (model1, dm1)

    def test_add_models_invalid_type(self):
        """Test adding models with invalid types."""
        evaluator = ModelEvaluator()

        # Invalid model type
        with pytest.raises(TypeError, match="must be a BaseLitModel"):
            evaluator.add_models({"model1": ("not_a_model", MagicMock(spec=LSHDataModule))})

        # Invalid datamodule type
        model = MagicMock(spec=BaseLitModel)
        with pytest.raises(TypeError, match="must be LSHDataModule"):
            evaluator.add_models({"model1": (model, "not_a_datamodule")})

    @patch("transfer_learning_publication.models.model_evaluator.L.Trainer")
    def test_test_models_single(self, mock_trainer_class):
        """Test testing a single model."""
        # Setup mock model and datamodule
        model = MagicMock(spec=BaseLitModel)
        datamodule = MagicMock(spec=LSHDataModule)

        # Mock ForecastOutput
        predictions = torch.tensor([[1.0, 2.0]])
        observations = torch.tensor([[1.1, 2.1]])
        forecast_output = ForecastOutput(
            predictions=predictions,
            observations=observations,
            group_identifiers=["basin1"],
        )
        model.forecast_output = forecast_output

        # Setup trainer mock
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        # Create evaluator and test
        evaluator = ModelEvaluator()
        evaluator.add_models({"model1": (model, datamodule)})

        results = evaluator.test_models()

        # Verify trainer was called
        mock_trainer_class.assert_called_once()
        mock_trainer.test.assert_called_once_with(model=model, datamodule=datamodule)

        # Verify results
        assert isinstance(results, EvaluationResults)
        assert "model1" in results.results_dict
        assert results.results_dict["model1"] == forecast_output

    @patch("transfer_learning_publication.models.model_evaluator.L.Trainer")
    def test_test_models_multiple(self, mock_trainer_class):
        """Test testing multiple models."""
        # Setup mock models and datamodules
        model1 = MagicMock(spec=BaseLitModel)
        model2 = MagicMock(spec=BaseLitModel)
        dm1 = MagicMock(spec=LSHDataModule)
        dm2 = MagicMock(spec=LSHDataModule)

        # Mock ForecastOutputs
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0]]),
            observations=torch.tensor([[1.1, 2.1]]),
            group_identifiers=["basin1"],
        )
        fo2 = ForecastOutput(
            predictions=torch.tensor([[3.0, 4.0]]),
            observations=torch.tensor([[3.1, 4.1]]),
            group_identifiers=["basin2"],
        )

        model1.forecast_output = fo1
        model2.forecast_output = fo2

        # Setup trainer mock
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        # Create evaluator and test
        evaluator = ModelEvaluator()
        evaluator.add_models(
            {
                "model1": (model1, dm1),
                "model2": (model2, dm2),
            }
        )

        results = evaluator.test_models()

        # Verify both models were tested
        assert mock_trainer.test.call_count == 2
        assert len(results.results_dict) == 2

    @patch("transfer_learning_publication.models.model_evaluator.L.Trainer")
    def test_test_models_only_subset(self, mock_trainer_class):
        """Test testing only a subset of models."""
        # Setup mock models
        model1 = MagicMock(spec=BaseLitModel)
        model2 = MagicMock(spec=BaseLitModel)
        model3 = MagicMock(spec=BaseLitModel)
        dm = MagicMock(spec=LSHDataModule)

        # Mock ForecastOutputs
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0]]),
            observations=torch.tensor([[1.1]]),
            group_identifiers=["basin1"],
        )
        fo3 = ForecastOutput(
            predictions=torch.tensor([[3.0]]),
            observations=torch.tensor([[3.1]]),
            group_identifiers=["basin3"],
        )

        model1.forecast_output = fo1
        model3.forecast_output = fo3

        # Setup trainer mock
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        # Create evaluator with 3 models
        evaluator = ModelEvaluator()
        evaluator.add_models(
            {
                "model1": (model1, dm),
                "model2": (model2, dm),
                "model3": (model3, dm),
            }
        )

        # Test only model1 and model3
        results = evaluator.test_models(only=["model1", "model3"])

        # Verify only 2 models were tested
        assert mock_trainer.test.call_count == 2
        assert len(results.results_dict) == 2
        assert "model1" in results.results_dict
        assert "model3" in results.results_dict
        assert "model2" not in results.results_dict

    def test_test_models_invalid_model_name(self):
        """Test requesting non-existent model."""
        evaluator = ModelEvaluator()
        model = MagicMock(spec=BaseLitModel)
        dm = MagicMock(spec=LSHDataModule)
        evaluator.add_models({"model1": (model, dm)})

        with pytest.raises(ValueError, match="Model 'model2' not found"):
            evaluator.test_models(only=["model2"])

    def test_test_models_no_models(self):
        """Test with no models to test."""
        evaluator = ModelEvaluator()

        with pytest.raises(ValueError, match="No models to test"):
            evaluator.test_models()

    @patch("transfer_learning_publication.models.model_evaluator.joblib.dump")
    @patch("transfer_learning_publication.models.model_evaluator.L.Trainer")
    def test_caching_save(self, mock_trainer_class, mock_joblib_dump, tmp_path):
        """Test saving results to cache."""
        # Setup mock model
        model = MagicMock(spec=BaseLitModel)
        datamodule = MagicMock(spec=LSHDataModule)
        datamodule.get_config_dict.return_value = {"input_length": 10}

        fo = ForecastOutput(
            predictions=torch.tensor([[1.0]]),
            observations=torch.tensor([[1.1]]),
            group_identifiers=["basin1"],
        )
        model.forecast_output = fo

        # Setup trainer mock
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        # Test with cache
        evaluator = ModelEvaluator()
        evaluator.add_models({"model1": (model, datamodule)})

        cache_dir = tmp_path / "cache"
        results = evaluator.test_models(cache_dir=cache_dir)

        # Verify cache structure created
        assert (cache_dir / "model_name=model1").exists()
        assert (cache_dir / "model_name=model1" / "config.json").exists()
        assert (cache_dir / "metadata.json").exists()

        # Verify joblib.dump was called
        mock_joblib_dump.assert_called_once()

    @patch("transfer_learning_publication.models.model_evaluator.joblib.load")
    @patch("transfer_learning_publication.models.model_evaluator.L.Trainer")
    def test_caching_load(self, mock_trainer_class, mock_joblib_load, tmp_path):
        """Test loading results from cache."""
        # Setup cache directory
        cache_dir = tmp_path / "cache"
        model_cache_dir = cache_dir / "model_name=model1"
        model_cache_dir.mkdir(parents=True)

        # Create config file
        config = {"input_length": 10}
        with open(model_cache_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Create empty forecast output file (just for existence check)
        (model_cache_dir / "forecast_output.joblib").touch()

        # Mock joblib.load to return ForecastOutput
        fo = ForecastOutput(
            predictions=torch.tensor([[1.0]]),
            observations=torch.tensor([[1.1]]),
            group_identifiers=["basin1"],
        )
        mock_joblib_load.return_value = fo

        # Setup model and datamodule
        model = MagicMock(spec=BaseLitModel)
        datamodule = MagicMock(spec=LSHDataModule)
        datamodule.get_config_dict.return_value = config  # Same config

        # Test loading from cache
        evaluator = ModelEvaluator()
        evaluator.add_models({"model1": (model, datamodule)})

        results = evaluator.test_models(cache_dir=cache_dir)

        # Verify trainer was NOT called (loaded from cache)
        mock_trainer_class.assert_not_called()

        # Verify joblib.load was called
        mock_joblib_load.assert_called_once()

        # Verify results
        assert results.results_dict["model1"] == fo

    def test_validate_cache_valid(self, tmp_path):
        """Test cache validation with valid cache."""
        # Setup cache directory
        cache_dir = tmp_path / "cache"
        model_cache_dir = cache_dir / "model_name=model1"
        model_cache_dir.mkdir(parents=True)

        # Create config file
        config = {"input_length": 10, "output_length": 1}
        with open(model_cache_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Create forecast output file
        (model_cache_dir / "forecast_output.joblib").touch()

        # Setup datamodule with matching config
        model = MagicMock(spec=BaseLitModel)
        datamodule = MagicMock(spec=LSHDataModule)
        datamodule.get_config_dict.return_value = config

        evaluator = ModelEvaluator()
        evaluator.add_models({"model1": (model, datamodule)})

        # Validate cache
        is_valid = evaluator.validate_cache(cache_dir, "model1")
        assert is_valid

    def test_validate_cache_config_mismatch(self, tmp_path):
        """Test cache validation with config mismatch."""
        # Setup cache directory
        cache_dir = tmp_path / "cache"
        model_cache_dir = cache_dir / "model_name=model1"
        model_cache_dir.mkdir(parents=True)

        # Create config file with old config
        old_config = {"input_length": 10}
        with open(model_cache_dir / "config.json", "w") as f:
            json.dump(old_config, f)

        # Create forecast output file
        (model_cache_dir / "forecast_output.joblib").touch()

        # Setup datamodule with different config
        model = MagicMock(spec=BaseLitModel)
        datamodule = MagicMock(spec=LSHDataModule)
        datamodule.get_config_dict.return_value = {"input_length": 20}  # Different

        evaluator = ModelEvaluator()
        evaluator.add_models({"model1": (model, datamodule)})

        # Validate cache
        is_valid = evaluator.validate_cache(cache_dir, "model1")
        assert not is_valid

    def test_validate_cache_missing_files(self, tmp_path):
        """Test cache validation with missing files."""
        cache_dir = tmp_path / "cache"

        evaluator = ModelEvaluator()

        # Test with non-existent cache dir
        is_valid = evaluator.validate_cache(cache_dir, "model1")
        assert not is_valid

        # Test with missing config file
        model_cache_dir = cache_dir / "model_name=model1"
        model_cache_dir.mkdir(parents=True)
        (model_cache_dir / "forecast_output.joblib").touch()

        is_valid = evaluator.validate_cache(cache_dir, "model1")
        assert not is_valid

        # Test with missing forecast output file
        model_cache_dir2 = cache_dir / "model_name=model2"
        model_cache_dir2.mkdir(parents=True)
        with open(model_cache_dir2 / "config.json", "w") as f:
            json.dump({}, f)

        is_valid = evaluator.validate_cache(cache_dir, "model2")
        assert not is_valid

    @patch("transfer_learning_publication.models.model_evaluator.L.Trainer")
    def test_force_recompute(self, mock_trainer_class, tmp_path):
        """Test force_recompute flag ignores cache."""
        # Setup valid cache
        cache_dir = tmp_path / "cache"
        model_cache_dir = cache_dir / "model_name=model1"
        model_cache_dir.mkdir(parents=True)

        config = {"input_length": 10}
        with open(model_cache_dir / "config.json", "w") as f:
            json.dump(config, f)
        (model_cache_dir / "forecast_output.joblib").touch()

        # Setup model
        model = MagicMock(spec=BaseLitModel)
        datamodule = MagicMock(spec=LSHDataModule)
        datamodule.get_config_dict.return_value = config

        fo = ForecastOutput(
            predictions=torch.tensor([[1.0]]),
            observations=torch.tensor([[1.1]]),
            group_identifiers=["basin1"],
        )
        model.forecast_output = fo

        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        evaluator = ModelEvaluator()
        evaluator.add_models({"model1": (model, datamodule)})

        # Test with force_recompute
        results = evaluator.test_models(cache_dir=cache_dir, force_recompute=True)

        # Verify trainer WAS called despite cache
        mock_trainer.test.assert_called_once()

    def test_get_cache_info(self, tmp_path):
        """Test getting cache information."""
        # Setup cache directory with models
        cache_dir = tmp_path / "cache"

        # Model 1 - valid cache
        model1_dir = cache_dir / "model_name=model1"
        model1_dir.mkdir(parents=True)
        with open(model1_dir / "config.json", "w") as f:
            json.dump({"input_length": 10}, f)
        with open(model1_dir / "forecast_output.joblib", "wb") as f:
            f.write(b"dummy_data")

        # Model 2 - missing config
        model2_dir = cache_dir / "model_name=model2"
        model2_dir.mkdir(parents=True)
        with open(model2_dir / "forecast_output.joblib", "wb") as f:
            f.write(b"data")

        # Add metadata
        with open(cache_dir / "metadata.json", "w") as f:
            json.dump({"created_at": "2024-01-01"}, f)

        evaluator = ModelEvaluator()

        # Get cache info
        info = evaluator.get_cache_info(cache_dir)

        assert info["exists"] is True
        assert len(info["models"]) == 2

        # Check model1 info
        model1_info = next(m for m in info["models"] if m["model_name"] == "model1")
        assert model1_info["has_output"] is True
        assert model1_info["has_config"] is True
        assert model1_info["output_size_bytes"] > 0

        # Check model2 info
        model2_info = next(m for m in info["models"] if m["model_name"] == "model2")
        assert model2_info["has_output"] is True
        assert model2_info["has_config"] is False

    def test_get_cache_info_no_cache(self, tmp_path):
        """Test cache info for non-existent cache."""
        evaluator = ModelEvaluator()
        info = evaluator.get_cache_info(tmp_path / "nonexistent")

        assert info["exists"] is False
        assert info["models"] == []

    def test_get_model(self):
        """Test getting model by name."""
        model1 = MagicMock(spec=BaseLitModel)
        model2 = MagicMock(spec=BaseLitModel)
        dm = MagicMock(spec=LSHDataModule)

        evaluator = ModelEvaluator()
        evaluator.add_models(
            {
                "model1": (model1, dm),
                "model2": (model2, dm),
            }
        )

        # Test valid model
        retrieved = evaluator.get_model("model1")
        assert retrieved is model1

        # Test invalid model
        with pytest.raises(KeyError, match="Model 'model3' not found"):
            evaluator.get_model("model3")

    def test_get_datamodule(self):
        """Test getting datamodule by model name."""
        model = MagicMock(spec=BaseLitModel)
        dm1 = MagicMock(spec=LSHDataModule)
        dm2 = MagicMock(spec=LSHDataModule)

        evaluator = ModelEvaluator()
        evaluator.add_models(
            {
                "model1": (model, dm1),
                "model2": (model, dm2),
            }
        )

        # Test valid model
        retrieved = evaluator.get_datamodule("model2")
        assert retrieved is dm2

        # Test invalid model
        with pytest.raises(KeyError, match="Model 'model3' not found"):
            evaluator.get_datamodule("model3")

    def test_list_models(self):
        """Test listing model names."""
        model = MagicMock(spec=BaseLitModel)
        dm = MagicMock(spec=LSHDataModule)

        evaluator = ModelEvaluator()

        # Empty list initially
        assert evaluator.list_models() == []

        # Add models
        evaluator.add_models(
            {
                "model_a": (model, dm),
                "model_b": (model, dm),
            }
        )

        models = evaluator.list_models()
        assert sorted(models) == ["model_a", "model_b"]

    def test_clear_cache_all(self, tmp_path):
        """Test clearing all cached models."""
        # Setup cache with multiple models
        cache_dir = tmp_path / "cache"

        for model_name in ["model1", "model2", "model3"]:
            model_dir = cache_dir / f"model_name={model_name}"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").touch()
            (model_dir / "forecast_output.joblib").touch()

        evaluator = ModelEvaluator()

        # Clear all
        evaluator.clear_cache(cache_dir)

        # Verify all model dirs removed
        assert not (cache_dir / "model_name=model1").exists()
        assert not (cache_dir / "model_name=model2").exists()
        assert not (cache_dir / "model_name=model3").exists()
        assert cache_dir.exists()  # Cache dir itself still exists

    def test_clear_cache_specific(self, tmp_path):
        """Test clearing specific cached models."""
        # Setup cache with multiple models
        cache_dir = tmp_path / "cache"

        for model_name in ["model1", "model2", "model3"]:
            model_dir = cache_dir / f"model_name={model_name}"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").touch()

        evaluator = ModelEvaluator()

        # Clear specific models
        evaluator.clear_cache(cache_dir, model_names=["model1", "model3"])

        # Verify only specified models removed
        assert not (cache_dir / "model_name=model1").exists()
        assert (cache_dir / "model_name=model2").exists()
        assert not (cache_dir / "model_name=model3").exists()

    def test_clear_cache_nonexistent(self, tmp_path):
        """Test clearing non-existent cache."""
        evaluator = ModelEvaluator()

        # Should not raise error
        evaluator.clear_cache(tmp_path / "nonexistent")

    @patch("transfer_learning_publication.models.model_evaluator.L.Trainer")
    def test_test_model_error_handling(self, mock_trainer_class):
        """Test error handling during model testing."""
        model = MagicMock(spec=BaseLitModel)
        datamodule = MagicMock(spec=LSHDataModule)

        # Test trainer.test() failure
        mock_trainer = MagicMock()
        mock_trainer.test.side_effect = Exception("Training failed")
        mock_trainer_class.return_value = mock_trainer

        evaluator = ModelEvaluator()
        evaluator.add_models({"model1": (model, datamodule)})

        with pytest.raises(RuntimeError, match="Testing failed for model 'model1'"):
            evaluator.test_models()

        # Test forecast_output retrieval failure
        mock_trainer.test.side_effect = None  # Reset

        # Create a new model mock that raises RuntimeError when forecast_output is accessed
        model2 = MagicMock(spec=BaseLitModel)
        type(model2).forecast_output = PropertyMock(side_effect=RuntimeError("No forecast output available"))

        evaluator2 = ModelEvaluator()
        evaluator2.add_models({"model2": (model2, datamodule)})

        with pytest.raises(RuntimeError, match="Could not retrieve forecast output"):
            evaluator2.test_models()
