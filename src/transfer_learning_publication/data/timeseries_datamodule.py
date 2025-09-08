import logging
from pathlib import Path

import torch
import yaml
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from ..containers import (
    DatasetConfig,
    LSHDataContainer,
    SequenceIndex,
)
from ..contracts import collate_fn
from .caravanify_parquet import CaravanDataSource
from .lsh_dataset import LSHDataset

logger = logging.getLogger(__name__)

class LSHDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for time series forecasting.

    Loads pre-processed, pre-split data from disk and provides DataLoaders
    for training, validation, and testing.

    Design Principles:
    - Configuration-driven from YAML
    - Fail-fast validation at construction
    - Single responsibility: data loading only
    - Trust upstream data preparation
    - Build containers once, reuse throughout training

    Args:
        config_path: Path to YAML configuration file
    """

    def __init__(self, config_path: str | Path):
        """
        Initialize DataModule from configuration file.

        Args:
            config_path: Path to YAML configuration file
        """
        super().__init__()
        self.config = self._load_config(config_path)
        self.config_path = Path(config_path)

        # Will be populated in setup()
        self.train_dataset: LSHDataset | None = None
        self.val_dataset: LSHDataset | None = None
        self.test_dataset: LSHDataset | None = None

        # Cache for containers (avoid rebuilding)
        self._train_container: LSHDataContainer | None = None
        self._val_container: LSHDataContainer | None = None
        self._test_container: LSHDataContainer | None = None

        # Pipeline for inverse transform (lazy loaded)
        self._pipeline = None
        self._pipeline_path = self.config["data"].get("pipeline_path")

        logger.info(f"LSHDataModule initialized with config: {config_path}")

    def _load_config(self, config_path: str | Path) -> dict[str, any]:
        """Load and validate configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Basic validation
        required_keys = ["data", "features", "sequence", "model", "dataloader"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config section: {key}")

        if "base_path" not in config["data"]:
            raise ValueError("Missing required config: data.base_path")
        if "region" not in config["data"]:
            raise ValueError("Missing required config: data.region")

        if "forcing" not in config["features"]:
            raise ValueError("Missing required config: features.forcing")
        if "static" not in config["features"]:
            raise ValueError("Missing required config: features.static")
        if "target" not in config["features"]:
            raise ValueError("Missing required config: features.target")

        if "input_length" not in config["sequence"]:
            raise ValueError("Missing required config: sequence.input_length")
        if "output_length" not in config["sequence"]:
            raise ValueError("Missing required config: sequence.output_length")

        if "is_autoregressive" not in config["model"]:
            raise ValueError("Missing required config: model.is_autoregressive")

        if "batch_size" not in config["dataloader"]:
            raise ValueError("Missing required config: dataloader.batch_size")

        return config

    def setup(self, stage: str | None = None) -> None:
        """
        Create datasets for the requested stage.

        Called by PyTorch Lightning before training/validation/testing.

        Args:
            stage: One of 'fit', 'validate', 'test', or None for all
        """
        if stage == "fit" or stage is None:
            if self.train_dataset is None:
                logger.info("Building training dataset...")
                self._train_container = self._build_container("train")
                self.train_dataset = LSHDataset(self._train_container)
                logger.info(f"Training dataset ready: {len(self.train_dataset)} sequences")

            if self.val_dataset is None:
                logger.info("Building validation dataset...")
                self._val_container = self._build_container("val")
                self.val_dataset = LSHDataset(self._val_container)
                logger.info(f"Validation dataset ready: {len(self.val_dataset)} sequences")

        if (stage == "test" or stage is None) and self.test_dataset is None:
            logger.info("Building test dataset...")
            self._test_container = self._build_container("test")
            self.test_dataset = LSHDataset(self._test_container)
            logger.info(f"Test dataset ready: {len(self.test_dataset)} sequences")

    def _build_container(self, split: str) -> LSHDataContainer:
        """
        Build LSHDataContainer for a given split.

        Args:
            split: One of 'train', 'val', 'test'

        Returns:
            LSHDataContainer ready for dataset creation
        """
        # Initialize data source
        base_path = Path(self.config["data"]["base_path"]) / split
        if not base_path.exists():
            raise FileNotFoundError(f"Data path not found for split '{split}': {base_path}")

        caravan = CaravanDataSource(base_path=base_path, region=self.config["data"]["region"])

        basins = caravan.list_gauge_ids()
        if not basins:
            raise ValueError(f"No basins found for split '{split}' in {base_path}")

        logger.info(f"Loading data for {len(basins)} basins from {split} split...")

        # Load time series data
        forcing_columns = self.config["features"]["forcing"]
        ts_lf = caravan.get_timeseries(gauge_ids=basins, columns=forcing_columns)

        # Load static attributes
        static_columns = self.config["features"]["static"]
        static_lf = caravan.get_static_attributes(gauge_ids=basins, columns=static_columns)

        # Convert to collections
        time_series = caravan.to_time_series_collection(ts_lf)
        static_attributes = caravan.to_static_attribute_collection(static_lf)

        # Validate we got the expected features (order-independent)
        if set(time_series.feature_names) != set(forcing_columns):
            raise ValueError(f"Feature mismatch: expected {forcing_columns}, got {time_series.feature_names}")

        # Build dataset configuration
        dataset_config = self._build_dataset_config(time_series.feature_names)

        # Create sequence index
        logger.info(f"Finding valid sequences for {split} split...")
        sequences = SequenceIndex.find_valid_sequences(
            time_series=time_series,
            input_length=self.config["sequence"]["input_length"],
            output_length=self.config["sequence"]["output_length"],
        )

        sequence_index = SequenceIndex(
            sequences=sequences,
            n_groups=len(time_series),
            input_length=self.config["sequence"]["input_length"],
            output_length=self.config["sequence"]["output_length"],
            validate=True,
        )

        logger.info(f"Found {len(sequence_index)} valid sequences for {split} split")

        # Create and return container
        return LSHDataContainer(
            time_series=time_series,
            static_attributes=static_attributes,
            sequence_index=sequence_index,
            config=dataset_config,
        )

    def _build_dataset_config(self, feature_names: list[str]) -> DatasetConfig:
        """
        Build DatasetConfig from configuration and detected features.

        Args:
            feature_names: List of feature names from time series data

        Returns:
            DatasetConfig with computed indices
        """
        config = self.config

        # Find target index
        target_name = config["features"]["target"]
        if target_name not in feature_names:
            raise ValueError(f"Target '{target_name}' not found in features: {feature_names}")
        target_idx = feature_names.index(target_name)

        # Find forcing feature indices
        forcing_names = config["features"]["forcing"]
        forcing_indices = []
        for name in forcing_names:
            if name not in feature_names:
                raise ValueError(f"Forcing feature '{name}' not found in features: {feature_names}")
            forcing_indices.append(feature_names.index(name))

        # Find future feature indices (if any)
        future_names = config["features"].get("future", [])
        future_indices = []
        for name in future_names:
            if name not in feature_names:
                raise ValueError(f"Future feature '{name}' not found in features: {feature_names}")
            future_indices.append(feature_names.index(name))

        # Determine input features based on autoregressive mode
        if config["model"]["is_autoregressive"]:
            # In autoregressive mode, all forcing features are inputs
            input_feature_indices = forcing_indices
        else:
            # In non-autoregressive mode, exclude target from input features
            input_feature_indices = [idx for idx in forcing_indices if idx != target_idx]

        return DatasetConfig(
            input_length=config["sequence"]["input_length"],
            output_length=config["sequence"]["output_length"],
            target_name=target_name,
            forcing_features=forcing_names,
            static_features=config["features"]["static"],
            future_features=future_names,
            target_idx=target_idx,
            forcing_indices=forcing_indices,
            future_indices=future_indices if future_indices else None,
            input_feature_indices=input_feature_indices,
            is_autoregressive=config["model"]["is_autoregressive"],
            include_dates=config["model"].get("include_dates", False),
            group_identifier_name="gauge_id",
        )

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("setup() must be called before train_dataloader()")

        return self._create_dataloader(self.train_dataset, shuffle=self.config["dataloader"].get("shuffle_train", True))

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        if self.val_dataset is None:
            raise RuntimeError("setup() must be called before val_dataloader()")

        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        if self.test_dataset is None:
            raise RuntimeError("setup() must be called before test_dataloader()")

        return self._create_dataloader(self.test_dataset, shuffle=False)

    def _create_dataloader(self, dataset: LSHDataset, shuffle: bool) -> DataLoader:
        """
        Create a DataLoader with configuration.

        Args:
            dataset: Dataset to wrap
            shuffle: Whether to shuffle data

        Returns:
            Configured DataLoader
        """
        dl_config = self.config["dataloader"]

        return DataLoader(
            dataset,
            batch_size=dl_config["batch_size"],
            shuffle=shuffle,
            num_workers=dl_config.get("num_workers", 0),
            pin_memory=dl_config.get("pin_memory", False),
            persistent_workers=dl_config.get("persistent_workers", False) and dl_config.get("num_workers", 0) > 0,
            collate_fn=collate_fn,
        )

    def inverse_transform(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform model predictions to original scale.

        Lazy loads the preprocessing pipeline when first called.

        Args:
            predictions: Model predictions in transformed space

        Returns:
            Predictions in original scale
        """
        if self._pipeline is None:
            if self._pipeline_path is None:
                raise ValueError("No pipeline path configured for inverse transform")

            import joblib

            self._pipeline = joblib.load(self._pipeline_path)
            logger.info(f"Loaded pipeline from {self._pipeline_path}")

        # Implementation depends on CompositePipeline's interface
        # This is a placeholder for the actual implementation
        return self._pipeline.inverse_transform_target(predictions)

    @property
    def num_features(self) -> int:
        """Number of input features."""
        return len(self.config["features"]["forcing"])

    @property
    def num_static_features(self) -> int:
        """Number of static features."""
        return len(self.config["features"]["static"])

    @property
    def num_future_features(self) -> int:
        """Number of future covariates."""
        return len(self.config["features"].get("future", []))

    @property
    def input_length(self) -> int:
        """Input sequence length."""
        return self.config["sequence"]["input_length"]

    @property
    def output_length(self) -> int:
        """Output sequence length."""
        return self.config["sequence"]["output_length"]

    @property
    def target_name(self) -> str:
        """Name of the target variable."""
        return self.config["features"]["target"]

    @property
    def is_autoregressive(self) -> bool:
        """Whether the model is autoregressive."""
        return self.config["model"]["is_autoregressive"]
