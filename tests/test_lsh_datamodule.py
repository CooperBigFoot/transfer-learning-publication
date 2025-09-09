import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from transfer_learning_publication.containers import DatasetConfig, LSHDataContainer, SequenceIndex
from transfer_learning_publication.contracts import Batch
from transfer_learning_publication.data import LSHDataModule


class TestLSHDataModule:
    """Tests for LSHDataModule class."""

    def test_init_with_valid_config(self, tmp_path):
        """Test initialization with valid configuration file."""
        # Create test config
        config = {
            "data": {"base_path": "/data", "region": "test"},
            "features": {
                "forcing": ["feature1", "feature2"],
                "static": ["static1"],
                "target": "feature1",
            },
            "sequence": {"input_length": 10, "output_length": 1},
            "model": {"is_autoregressive": True},
            "dataloader": {"batch_size": 32},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Initialize datamodule
        dm = LSHDataModule(config_path)

        assert dm.config == config
        assert dm.train_dataset is None
        assert dm.val_dataset is None
        assert dm.test_dataset is None

    def test_init_missing_config_file(self, tmp_path):
        """Test initialization with missing configuration file."""
        config_path = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            LSHDataModule(config_path)

    def test_config_validation_missing_sections(self, tmp_path):
        """Test configuration validation for missing required sections."""
        # Test missing data section
        config = {
            "features": {"forcing": ["f1"], "static": ["s1"], "target": "f1"},
            "sequence": {"input_length": 10, "output_length": 1},
            "model": {"is_autoregressive": True},
            "dataloader": {"batch_size": 32},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(ValueError, match="Missing required config section: data"):
            LSHDataModule(config_path)

    def test_config_validation_missing_keys(self, tmp_path):
        """Test configuration validation for missing required keys."""
        # Test missing base_path
        config = {
            "data": {"region": "test"},
            "features": {"forcing": ["f1"], "static": ["s1"], "target": "f1"},
            "sequence": {"input_length": 10, "output_length": 1},
            "model": {"is_autoregressive": True},
            "dataloader": {"batch_size": 32},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(ValueError, match="Missing required config: data.base_path"):
            LSHDataModule(config_path)

    def test_build_dataset_config_autoregressive(self, tmp_path):
        """Test _build_dataset_config for autoregressive mode."""
        # Create test config
        config = {
            "data": {"base_path": "/data", "region": "test"},
            "features": {
                "forcing": ["streamflow", "precipitation", "temperature"],
                "static": ["area"],
                "target": "streamflow",
                "future": ["temperature"],
            },
            "sequence": {"input_length": 10, "output_length": 1},
            "model": {"is_autoregressive": True, "include_dates": True},
            "dataloader": {"batch_size": 32},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        dm = LSHDataModule(config_path)

        # Test dataset config generation
        feature_names = ["streamflow", "precipitation", "temperature"]
        dataset_config = dm._build_dataset_config(feature_names)

        assert dataset_config.input_length == 10
        assert dataset_config.output_length == 1
        assert dataset_config.target_name == "streamflow"
        assert dataset_config.target_idx == 0
        assert dataset_config.forcing_features == ["streamflow", "precipitation", "temperature"]
        assert dataset_config.forcing_indices == [0, 1, 2]
        assert dataset_config.future_features == ["temperature"]
        assert dataset_config.future_indices == [2]
        assert dataset_config.input_feature_indices == [0, 1, 2]  # Target first, then others
        assert dataset_config.is_autoregressive is True
        assert dataset_config.include_dates is True

    def test_autoregressive_target_first_reordering(self, tmp_path):
        """Test that target is reordered to index 0 for autoregressive models."""
        # Create test config where target is NOT first in forcing list
        config = {
            "data": {"base_path": "/data", "region": "test"},
            "features": {
                "forcing": ["precipitation", "temperature", "streamflow"],  # Target is last
                "static": ["area"],
                "target": "streamflow",
            },
            "sequence": {"input_length": 10, "output_length": 1},
            "model": {"is_autoregressive": True},
            "dataloader": {"batch_size": 32},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        dm = LSHDataModule(config_path)

        # Test with feature order matching config (target at index 2)
        feature_names = ["precipitation", "temperature", "streamflow"]
        dataset_config = dm._build_dataset_config(feature_names)

        assert dataset_config.target_name == "streamflow"
        assert dataset_config.target_idx == 2  # Original position in feature_names
        assert dataset_config.forcing_indices == [0, 1, 2]  # Original order preserved
        
        # CRITICAL: Check that input_feature_indices has target FIRST
        assert dataset_config.input_feature_indices == [2, 0, 1]  # Target (2) moved to first position
        assert dataset_config.input_feature_indices[0] == dataset_config.target_idx  # Verify target is first
        
    def test_build_dataset_config_non_autoregressive_validation(self, tmp_path):
        """Test that non-autoregressive mode validates target not in forcing."""
        # Test that it raises error when target is in forcing for non-autoregressive
        config = {
            "data": {"base_path": "/data", "region": "test"},
            "features": {
                "forcing": ["streamflow", "precipitation", "temperature"],  # Target in forcing
                "static": ["area"],
                "target": "streamflow",
            },
            "sequence": {"input_length": 10, "output_length": 1},
            "model": {"is_autoregressive": False},  # Non-autoregressive with target in forcing
            "dataloader": {"batch_size": 32},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        dm = LSHDataModule(config_path)

        # This should raise an error because target is in forcing for non-autoregressive
        feature_names = ["streamflow", "precipitation", "temperature"]
        with pytest.raises(ValueError, match="Target 'streamflow' cannot be in forcing_features"):
            dm._build_dataset_config(feature_names)

    def test_build_dataset_config_missing_target(self, tmp_path):
        """Test _build_dataset_config with missing target in features."""
        config = {
            "data": {"base_path": "/data", "region": "test"},
            "features": {
                "forcing": ["precipitation", "temperature"],
                "static": ["area"],
                "target": "streamflow",  # Not in forcing
            },
            "sequence": {"input_length": 10, "output_length": 1},
            "model": {"is_autoregressive": True},
            "dataloader": {"batch_size": 32},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        dm = LSHDataModule(config_path)

        feature_names = ["precipitation", "temperature"]
        with pytest.raises(ValueError, match="Target 'streamflow' not found in features"):
            dm._build_dataset_config(feature_names)

    @patch("transfer_learning_publication.data.lsh_datamodule.CaravanDataSource")
    def test_build_container_success(self, mock_caravan_class, tmp_path):
        """Test _build_container with successful data loading."""
        # Setup config
        config = {
            "data": {"base_path": str(tmp_path), "region": "test"},
            "features": {
                "forcing": ["streamflow", "precipitation"],
                "static": ["area"],
                "target": "streamflow",
            },
            "sequence": {"input_length": 10, "output_length": 1},
            "model": {"is_autoregressive": True},
            "dataloader": {"batch_size": 32},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Create data directories
        (tmp_path / "train").mkdir()
        (tmp_path / "val").mkdir()
        (tmp_path / "test").mkdir()

        # Setup mocks
        mock_caravan = MagicMock()
        mock_caravan_class.return_value = mock_caravan

        # Mock gauge IDs
        mock_caravan.list_gauge_ids.return_value = ["gauge1", "gauge2"]

        # Mock LazyFrames
        mock_ts_lf = MagicMock()
        mock_static_lf = MagicMock()
        mock_caravan.get_timeseries.return_value = mock_ts_lf
        mock_caravan.get_static_attributes.return_value = mock_static_lf

        # Mock collections
        mock_time_series = MagicMock()
        mock_time_series.feature_names = ["streamflow", "precipitation"]
        mock_time_series.__len__.return_value = 2
        mock_time_series.get_n_features.return_value = 2

        mock_static_attrs = MagicMock()
        mock_static_attrs.attribute_names = ["area"]
        mock_static_attrs.__len__.return_value = 2
        mock_static_attrs.get_n_attributes.return_value = 1

        mock_caravan.to_time_series_collection.return_value = mock_time_series
        mock_caravan.to_static_attribute_collection.return_value = mock_static_attrs

        # Initialize datamodule
        dm = LSHDataModule(config_path)

        # Test container building
        with patch.object(SequenceIndex, "find_valid_sequences") as mock_find:
            mock_find.return_value = torch.tensor([[0, 0, 11], [0, 1, 12], [1, 0, 11]])

            container = dm._build_container("train")

        assert isinstance(container, LSHDataContainer)
        assert container.time_series == mock_time_series
        assert container.static_attributes == mock_static_attrs
        assert isinstance(container.sequence_index, SequenceIndex)
        assert isinstance(container.config, DatasetConfig)

    def test_build_container_missing_split_directory(self, tmp_path):
        """Test _build_container with missing split directory."""
        config = {
            "data": {"base_path": str(tmp_path), "region": "test"},
            "features": {
                "forcing": ["streamflow"],
                "static": ["area"],
                "target": "streamflow",
            },
            "sequence": {"input_length": 10, "output_length": 1},
            "model": {"is_autoregressive": True},
            "dataloader": {"batch_size": 32},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        dm = LSHDataModule(config_path)

        with pytest.raises(FileNotFoundError, match="Data path not found for split 'train'"):
            dm._build_container("train")

    @patch("transfer_learning_publication.data.lsh_datamodule.LSHDataset")
    @patch.object(LSHDataModule, "_build_container")
    def test_setup_fit_stage(self, mock_build_container, mock_dataset_class, tmp_path):
        """Test setup method for fit stage."""
        config = {
            "data": {"base_path": str(tmp_path), "region": "test"},
            "features": {
                "forcing": ["streamflow"],
                "static": ["area"],
                "target": "streamflow",
            },
            "sequence": {"input_length": 10, "output_length": 1},
            "model": {"is_autoregressive": True},
            "dataloader": {"batch_size": 32},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Setup mocks
        mock_container = MagicMock()
        mock_build_container.return_value = mock_container

        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset_class.return_value = mock_dataset

        dm = LSHDataModule(config_path)

        # Test setup for fit stage
        dm.setup(stage="fit")

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is None
        assert mock_build_container.call_count == 2  # train and val

    def test_dataloader_creation(self, tmp_path):
        """Test DataLoader creation methods."""
        config = {
            "data": {"base_path": str(tmp_path), "region": "test"},
            "features": {
                "forcing": ["streamflow"],
                "static": ["area"],
                "target": "streamflow",
            },
            "sequence": {"input_length": 10, "output_length": 1},
            "model": {"is_autoregressive": True},
            "dataloader": {
                "batch_size": 16,
                "num_workers": 2,
                "pin_memory": True,
                "persistent_workers": True,
                "shuffle_train": True,
            },
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        dm = LSHDataModule(config_path)

        # Mock datasets with __len__ method
        dm.train_dataset = MagicMock()
        dm.train_dataset.__len__.return_value = 100
        dm.val_dataset = MagicMock()
        dm.val_dataset.__len__.return_value = 50
        dm.test_dataset = MagicMock()
        dm.test_dataset.__len__.return_value = 25

        # Test train dataloader
        train_dl = dm.train_dataloader()
        assert isinstance(train_dl, DataLoader)
        assert train_dl.batch_size == 16
        assert train_dl.num_workers == 2
        assert train_dl.pin_memory is True

        # Test val dataloader
        val_dl = dm.val_dataloader()
        assert isinstance(val_dl, DataLoader)
        assert val_dl.batch_size == 16

        # Test test dataloader
        test_dl = dm.test_dataloader()
        assert isinstance(test_dl, DataLoader)
        assert test_dl.batch_size == 16

    def test_dataloader_without_setup(self, tmp_path):
        """Test DataLoader creation without calling setup first."""
        config = {
            "data": {"base_path": str(tmp_path), "region": "test"},
            "features": {
                "forcing": ["streamflow"],
                "static": ["area"],
                "target": "streamflow",
            },
            "sequence": {"input_length": 10, "output_length": 1},
            "model": {"is_autoregressive": True},
            "dataloader": {"batch_size": 32},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        dm = LSHDataModule(config_path)

        with pytest.raises(RuntimeError, match="setup\\(\\) must be called"):
            dm.train_dataloader()

    def test_properties(self, tmp_path):
        """Test DataModule properties."""
        config = {
            "data": {"base_path": str(tmp_path), "region": "test"},
            "features": {
                "forcing": ["streamflow", "precipitation", "temperature"],
                "static": ["area", "elevation"],
                "future": ["temperature"],
                "target": "streamflow",
            },
            "sequence": {"input_length": 365, "output_length": 7},
            "model": {"is_autoregressive": True},
            "dataloader": {"batch_size": 32},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        dm = LSHDataModule(config_path)

        assert dm.num_features == 3
        assert dm.num_static_features == 2
        assert dm.num_future_features == 1
        assert dm.input_length == 365
        assert dm.output_length == 7
        assert dm.target_name == "streamflow"
        assert dm.is_autoregressive is True

    @patch("joblib.load")
    def test_inverse_transform_with_pipeline(self, mock_joblib_load, tmp_path):
        """Test inverse_transform with pipeline loaded."""
        config = {
            "data": {
                "base_path": str(tmp_path),
                "region": "test",
                "pipeline_path": "/path/to/pipeline.joblib",
            },
            "features": {
                "forcing": ["streamflow"],
                "static": ["area"],
                "target": "streamflow",
            },
            "sequence": {"input_length": 10, "output_length": 1},
            "model": {"is_autoregressive": True},
            "dataloader": {"batch_size": 32},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.inverse_transform_target.return_value = torch.tensor([1.0, 2.0])
        mock_joblib_load.return_value = mock_pipeline

        dm = LSHDataModule(config_path)

        predictions = torch.tensor([0.5, 1.5])
        result = dm.inverse_transform(predictions)

        assert torch.equal(result, torch.tensor([1.0, 2.0]))
        mock_joblib_load.assert_called_once_with("/path/to/pipeline.joblib")
        mock_pipeline.inverse_transform_target.assert_called_once()

    def test_inverse_transform_without_pipeline(self, tmp_path):
        """Test inverse_transform without pipeline configured."""
        config = {
            "data": {"base_path": str(tmp_path), "region": "test"},
            "features": {
                "forcing": ["streamflow"],
                "static": ["area"],
                "target": "streamflow",
            },
            "sequence": {"input_length": 10, "output_length": 1},
            "model": {"is_autoregressive": True},
            "dataloader": {"batch_size": 32},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        dm = LSHDataModule(config_path)

        predictions = torch.tensor([0.5, 1.5])
        with pytest.raises(ValueError, match="No pipeline path configured"):
            dm.inverse_transform(predictions)