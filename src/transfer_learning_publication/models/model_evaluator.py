"""ModelEvaluator for orchestrating model testing and evaluation."""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import lightning as pl

from ..contracts import EvaluationResults, ForecastOutput
from ..data import LSHDataModule
from .base import BaseLitModel

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Orchestrates testing and evaluation of multiple trained models.

    This class abstracts the testing workflow for comparing model performance,
    providing intelligent caching to avoid redundant computations and
    standardized outputs for downstream analysis.

    The evaluator assumes models are already trained and focuses on:
    - Running test phase with PyTorch Lightning
    - Collecting ForecastOutput from models
    - Caching results with Hive-style partitioning
    - Validating cache against configuration changes
    - Returning standardized EvaluationResults

    Example:
        >>> evaluator = ModelEvaluator(
        ...     models_and_datamodules={
        ...         "tide": (tide_model, datamodule),
        ...         "ealstm": (ealstm_model, datamodule),
        ...     },
        ...     trainer_kwargs={"accelerator": "cpu"}
        ... )
        >>> results = evaluator.test_models(cache_dir="cache/exp001")
        >>> df = results.by_model("tide")

    Attributes:
        models_and_datamodules: Registry of model/datamodule pairs
        trainer_kwargs: Arguments for PyTorch Lightning Trainer
    """

    def __init__(
        self,
        models_and_datamodules: dict[str, tuple[BaseLitModel, LSHDataModule]] | None = None,
        trainer_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize ModelEvaluator.

        Args:
            models_and_datamodules: Dictionary mapping model names to
                (model, datamodule) tuples
            trainer_kwargs: Keyword arguments for PyTorch Lightning Trainer
        """
        self.models_and_datamodules: dict[str, tuple[BaseLitModel, LSHDataModule]] = {}
        self.trainer_kwargs = trainer_kwargs or {"accelerator": "auto", "devices": 1}

        if models_and_datamodules:
            self.add_models(models_and_datamodules)

    def add_models(self, models_and_datamodules: dict[str, tuple[BaseLitModel, LSHDataModule]]) -> None:
        """Add models to the evaluator registry.

        Args:
            models_and_datamodules: Dictionary mapping model names to
                (model, datamodule) tuples

        Raises:
            TypeError: If model is not a BaseLitModel or datamodule is not LSHDataModule
        """
        for name, (model, datamodule) in models_and_datamodules.items():
            if not isinstance(model, BaseLitModel):
                raise TypeError(f"Model '{name}' must be a BaseLitModel, got {type(model)}")
            if not isinstance(datamodule, LSHDataModule):
                raise TypeError(f"DataModule for '{name}' must be LSHDataModule, got {type(datamodule)}")

            self.models_and_datamodules[name] = (model, datamodule)
            logger.info(f"Added model '{name}' to evaluator")

    def test_models(
        self,
        cache_dir: str | Path | None = None,
        only: list[str] | None = None,
        force_recompute: bool = False,
    ) -> EvaluationResults:
        """Test models and return evaluation results.

        Tests specified models, using cache when available and valid.
        Results are returned as an EvaluationResults object for analysis.

        Args:
            cache_dir: Directory for caching results (optional)
            only: List of model names to test (defaults to all)
            force_recompute: Force recomputation even if cache exists

        Returns:
            EvaluationResults containing all model outputs

        Raises:
            ValueError: If requested model not found in registry
            RuntimeError: If model testing fails
        """
        # Determine which models to test
        if only is None:
            models_to_test = list(self.models_and_datamodules.keys())
        else:
            # Validate requested models exist
            for name in only:
                if name not in self.models_and_datamodules:
                    available = list(self.models_and_datamodules.keys())
                    raise ValueError(f"Model '{name}' not found. Available: {available}")
            models_to_test = only

        if not models_to_test:
            raise ValueError("No models to test")

        logger.info(f"Testing {len(models_to_test)} models: {models_to_test}")

        # Prepare cache directory if provided
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._write_cache_metadata(cache_dir, models_to_test)

        # Collect results
        results_dict: dict[str, ForecastOutput] = {}

        for model_name in models_to_test:
            logger.info(f"Processing model '{model_name}'...")

            # Check cache if not forcing recompute
            if cache_dir and not force_recompute:
                cached_output = self._load_from_cache(cache_dir, model_name)
                if cached_output is not None:
                    logger.info(f"Loaded '{model_name}' from cache")
                    results_dict[model_name] = cached_output
                    continue

            # Compute if not cached or forcing recompute
            logger.info(f"Testing model '{model_name}'...")
            forecast_output = self._test_model(model_name)
            results_dict[model_name] = forecast_output

            # Save to cache if directory provided
            if cache_dir:
                self._save_to_cache(cache_dir, model_name, forecast_output)
                logger.info(f"Cached results for '{model_name}'")

        # Get output length from first model
        first_output = next(iter(results_dict.values()))
        output_length = first_output.predictions.shape[1]

        return EvaluationResults(
            results_dict=results_dict,
            output_length=output_length,
        )

    def _test_model(self, model_name: str) -> ForecastOutput:
        """Run test phase for a single model.

        Args:
            model_name: Name of model to test

        Returns:
            ForecastOutput from model testing

        Raises:
            RuntimeError: If testing fails
        """
        model, datamodule = self.models_and_datamodules[model_name]

        trainer = pl.Trainer(
            **self.trainer_kwargs,
            logger=False,
            enable_progress_bar=True,
        )

        try:
            trainer.test(model=model, datamodule=datamodule)
        except Exception as e:
            raise RuntimeError(f"Testing failed for model '{model_name}': {e}")

        # Get forecast output
        try:
            forecast_output = model.forecast_output
        except RuntimeError as e:
            raise RuntimeError(f"Could not retrieve forecast output for '{model_name}': {e}")

        return forecast_output

    def _load_from_cache(self, cache_dir: Path, model_name: str) -> ForecastOutput | None:
        """Load cached results if valid.

        Args:
            cache_dir: Cache directory
            model_name: Name of model

        Returns:
            ForecastOutput if cache is valid, None otherwise
        """
        model_cache_dir = cache_dir / f"model_name={model_name}"

        if not model_cache_dir.exists():
            logger.debug(f"No cache found for '{model_name}'")
            return None

        # Check if cache is valid
        if not self.validate_cache(cache_dir, model_name):
            logger.warning(f"Cache invalid for '{model_name}', will recompute")
            return None

        # Load forecast output
        output_path = model_cache_dir / "forecast_output.joblib"

        try:
            forecast_output = joblib.load(output_path)
            return forecast_output
        except Exception as e:
            logger.warning(f"Failed to load cache for '{model_name}': {e}")
            return None

    def _save_to_cache(self, cache_dir: Path, model_name: str, forecast_output: ForecastOutput) -> None:
        """Save results to cache.

        Args:
            cache_dir: Cache directory
            model_name: Name of model
            forecast_output: Results to cache
        """
        model_cache_dir = cache_dir / f"model_name={model_name}"
        model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Save forecast output
        output_path = model_cache_dir / "forecast_output.joblib"
        joblib.dump(forecast_output, output_path)

        # Save datamodule config for validation
        _, datamodule = self.models_and_datamodules[model_name]
        config_path = model_cache_dir / "config.json"

        with open(config_path, "w") as f:
            json.dump(datamodule.get_config_dict(), f, indent=2)

    def validate_cache(self, cache_dir: str | Path, model_name: str) -> bool:
        """Validate if cached results are still valid.

        Compares stored configuration with current datamodule config.

        Args:
            cache_dir: Cache directory
            model_name: Name of model to validate

        Returns:
            True if cache is valid, False otherwise
        """
        cache_dir = Path(cache_dir)
        model_cache_dir = cache_dir / f"model_name={model_name}"

        if not model_cache_dir.exists():
            return False

        config_path = model_cache_dir / "config.json"
        if not config_path.exists():
            logger.debug(f"No config found for cached '{model_name}'")
            return False

        # Load stored config
        try:
            with open(config_path) as f:
                stored_config = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached config for '{model_name}': {e}")
            return False

        # Get current config
        if model_name not in self.models_and_datamodules:
            logger.debug(f"Model '{model_name}' not in current registry")
            return False

        _, datamodule = self.models_and_datamodules[model_name]
        current_config = datamodule.get_config_dict()

        # Compare configs
        if stored_config != current_config:
            logger.debug(f"Config mismatch for '{model_name}'")
            logger.debug(f"Stored: {stored_config}")
            logger.debug(f"Current: {current_config}")
            return False

        # Check that forecast output file exists
        output_path = model_cache_dir / "forecast_output.joblib"
        if not output_path.exists():
            logger.debug(f"Forecast output missing for '{model_name}'")
            return False

        return True

    def get_cache_info(self, cache_dir: str | Path) -> dict[str, Any]:
        """Get information about cached results.

        Args:
            cache_dir: Cache directory

        Returns:
            Dictionary with cache information
        """
        cache_dir = Path(cache_dir)

        if not cache_dir.exists():
            return {"exists": False, "models": []}

        # Find all cached models
        cached_models = []
        for model_dir in cache_dir.glob("model_name=*"):
            if model_dir.is_dir():
                model_name = model_dir.name.replace("model_name=", "")

                # Check if files exist
                has_output = (model_dir / "forecast_output.joblib").exists()
                has_config = (model_dir / "config.json").exists()
                is_valid = self.validate_cache(cache_dir, model_name)

                # Get file sizes
                output_size = 0
                if has_output:
                    output_size = (model_dir / "forecast_output.joblib").stat().st_size

                cached_models.append(
                    {
                        "model_name": model_name,
                        "has_output": has_output,
                        "has_config": has_config,
                        "is_valid": is_valid,
                        "output_size_bytes": output_size,
                    }
                )

        # Load metadata if exists
        metadata = {}
        metadata_path = cache_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            except Exception:
                pass

        return {
            "exists": True,
            "path": str(cache_dir),
            "models": cached_models,
            "metadata": metadata,
        }

    def _write_cache_metadata(self, cache_dir: Path, models_to_test: list[str]) -> None:
        """Write metadata file for cache directory.

        Args:
            cache_dir: Cache directory
            models_to_test: List of models being tested
        """
        import datetime

        metadata = {
            "created_at": datetime.datetime.now().isoformat(),
            "models_requested": models_to_test,
            "evaluator_version": "1.0.0",
        }

        metadata_path = cache_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def get_model(self, model_name: str) -> BaseLitModel:
        """Get model by name.

        Args:
            model_name: Name of model

        Returns:
            Model instance

        Raises:
            KeyError: If model not found
        """
        if model_name not in self.models_and_datamodules:
            available = list(self.models_and_datamodules.keys())
            raise KeyError(f"Model '{model_name}' not found. Available: {available}")

        model, _ = self.models_and_datamodules[model_name]
        return model

    def get_datamodule(self, model_name: str) -> LSHDataModule:
        """Get datamodule for a model.

        Args:
            model_name: Name of model

        Returns:
            DataModule instance

        Raises:
            KeyError: If model not found
        """
        if model_name not in self.models_and_datamodules:
            available = list(self.models_and_datamodules.keys())
            raise KeyError(f"Model '{model_name}' not found. Available: {available}")

        _, datamodule = self.models_and_datamodules[model_name]
        return datamodule

    def list_models(self) -> list[str]:
        """Get list of registered model names.

        Returns:
            List of model names
        """
        return list(self.models_and_datamodules.keys())

    def clear_cache(self, cache_dir: str | Path, model_names: list[str] | None = None) -> None:
        """Clear cached results.

        Args:
            cache_dir: Cache directory
            model_names: Specific models to clear (None for all)
        """
        cache_dir = Path(cache_dir)

        if not cache_dir.exists():
            logger.info("Cache directory does not exist")
            return

        if model_names is None:
            # Clear all model caches
            for model_dir in cache_dir.glob("model_name=*"):
                if model_dir.is_dir():
                    import shutil

                    shutil.rmtree(model_dir)
                    logger.info(f"Cleared cache for '{model_dir.name}'")
        else:
            # Clear specific models
            for model_name in model_names:
                model_dir = cache_dir / f"model_name={model_name}"
                if model_dir.exists():
                    import shutil

                    shutil.rmtree(model_dir)
                    logger.info(f"Cleared cache for '{model_name}'")
