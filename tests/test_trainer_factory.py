"""Tests for the trainer factory module."""

from unittest.mock import MagicMock, patch

import pytest
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from transfer_learning_publication.callbacks.progress import MinimalProgressCallback
from transfer_learning_publication.training_cli.trainer_factory import (
    create_trainer,
    train_single_model,
)


class TestCreateTrainer:
    """Test trainer creation with proper configuration."""

    def test_creates_trainer_with_correct_callbacks(self, tmp_path):
        """Test that trainer is created with all expected callbacks."""
        checkpoint_dir = tmp_path / "checkpoints"
        tensorboard_dir = tmp_path / "tensorboard"

        trainer = create_trainer(
            model_name="tide", seed=42, checkpoint_dir=checkpoint_dir, tensorboard_dir=tensorboard_dir
        )

        # Check callbacks
        callback_types = [type(cb) for cb in trainer.callbacks]
        assert ModelCheckpoint in callback_types
        assert EarlyStopping in callback_types
        assert MinimalProgressCallback in callback_types

    def test_model_checkpoint_configuration(self, tmp_path):
        """Test ModelCheckpoint callback configuration."""
        checkpoint_dir = tmp_path / "checkpoints"

        trainer = create_trainer(model_name="tide", seed=42, checkpoint_dir=checkpoint_dir)

        # Find ModelCheckpoint callback
        checkpoint_cb = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                checkpoint_cb = cb
                break

        assert checkpoint_cb is not None
        assert checkpoint_cb.monitor == "val_loss"
        assert checkpoint_cb.mode == "min"
        assert checkpoint_cb.save_top_k == 1
        assert checkpoint_cb.save_last
        from pathlib import Path

        assert Path(checkpoint_cb.dirpath) == checkpoint_dir / "checkpoints"
        assert "val_loss" in checkpoint_cb.filename

    def test_early_stopping_configuration(self, tmp_path):
        """Test EarlyStopping callback configuration."""
        trainer = create_trainer(model_name="tide", seed=42, checkpoint_dir=tmp_path / "checkpoints")

        # Find EarlyStopping callback
        early_stop_cb = None
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                early_stop_cb = cb
                break

        assert early_stop_cb is not None
        assert early_stop_cb.monitor == "val_loss"
        assert early_stop_cb.mode == "min"
        assert early_stop_cb.patience == 10

    def test_minimal_progress_callback_configuration(self, tmp_path):
        """Test MinimalProgressCallback configuration."""
        trainer = create_trainer(model_name="ealstm", seed=123, checkpoint_dir=tmp_path / "checkpoints")

        # Find MinimalProgressCallback
        progress_cb = None
        for cb in trainer.callbacks:
            if isinstance(cb, MinimalProgressCallback):
                progress_cb = cb
                break

        assert progress_cb is not None
        assert progress_cb.model_name == "ealstm"
        assert progress_cb.seed == 123

    def test_logger_configuration(self, tmp_path):
        """Test logger configuration."""
        checkpoint_dir = tmp_path / "checkpoints"
        tensorboard_dir = tmp_path / "tensorboard"

        trainer = create_trainer(
            model_name="tide", seed=42, checkpoint_dir=checkpoint_dir, tensorboard_dir=tensorboard_dir
        )

        # Check loggers
        logger_types = [type(logger) for logger in trainer.loggers]
        assert CSVLogger in logger_types
        assert TensorBoardLogger in logger_types

        # Check TensorBoard logger configuration
        tb_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger
                break

        assert tb_logger is not None
        assert "tide_seed42" in tb_logger.name

    def test_trainer_basic_configuration(self, tmp_path):
        """Test basic trainer configuration."""
        trainer = create_trainer(model_name="tide", seed=42, checkpoint_dir=tmp_path / "checkpoints", max_epochs=50)

        assert trainer.max_epochs == 50
        # The trainer is created successfully with the expected configuration

    def test_default_max_epochs(self, tmp_path):
        """Test default max_epochs when not specified."""
        trainer = create_trainer(model_name="tide", seed=42, checkpoint_dir=tmp_path / "checkpoints")

        assert trainer.max_epochs == 100

    def test_creates_directories_if_not_exist(self, tmp_path):
        """Test that directories are created if they don't exist."""
        checkpoint_dir = tmp_path / "new_checkpoints"
        tensorboard_dir = tmp_path / "new_tensorboard"

        assert not checkpoint_dir.exists()
        assert not tensorboard_dir.exists()

        create_trainer(model_name="tide", seed=42, checkpoint_dir=checkpoint_dir, tensorboard_dir=tensorboard_dir)

        assert checkpoint_dir.exists()
        assert tensorboard_dir.exists()

    def test_trainer_creation_with_seed(self, tmp_path):
        """Test that trainer is created successfully with a seed."""
        trainer = create_trainer(model_name="tide", seed=42, checkpoint_dir=tmp_path / "checkpoints")
        
        # Verify trainer was created
        assert trainer is not None
        assert trainer.max_epochs == 100  # Default when not specified


class TestTrainSingleModel:
    """Test single model training."""

    @pytest.fixture
    def config_path(self, tmp_path):
        """Create a temporary config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model:\n  type: tide\n")
        return config_file

    @patch("transfer_learning_publication.training_cli.trainer_factory.pl.seed_everything")
    @patch("transfer_learning_publication.training_cli.trainer_factory.ModelFactory")
    @patch("transfer_learning_publication.training_cli.trainer_factory.LSHDataModule")
    @patch("transfer_learning_publication.training_cli.trainer_factory.create_trainer")
    def test_sets_seed_before_model_creation(self, mock_create_trainer, mock_datamodule, mock_factory, mock_seed, config_path, tmp_path):
        """Test that seed is set before model creation for reproducibility."""
        # Setup mocks
        mock_model = MagicMock()
        mock_factory.create_from_config.return_value = mock_model
        mock_dm = MagicMock()
        mock_datamodule.return_value = mock_dm
        mock_trainer = MagicMock()
        mock_create_trainer.return_value = mock_trainer

        # Call function
        train_single_model(
            config_path=config_path,
            model_name="tide",
            seed=42,
            checkpoint_dir=tmp_path / "checkpoints",
            tensorboard_dir=tmp_path / "tensorboard",
        )

        # Verify seed was set with correct arguments
        mock_seed.assert_called_once_with(42, workers=True)
        
        # Verify seed was set before model creation
        assert mock_seed.call_args_list[0][0] == (42,)
        assert mock_factory.create_from_config.called

    @patch("transfer_learning_publication.training_cli.trainer_factory.ModelFactory")
    @patch("transfer_learning_publication.training_cli.trainer_factory.LSHDataModule")
    @patch("transfer_learning_publication.training_cli.trainer_factory.create_trainer")
    def test_successful_training(self, mock_create_trainer, mock_datamodule, mock_factory, config_path, tmp_path):
        """Test successful model training."""
        # Setup mocks
        mock_model = MagicMock()
        mock_factory.create_from_config.return_value = mock_model

        mock_dm = MagicMock()
        mock_datamodule.return_value = mock_dm

        mock_trainer = MagicMock()
        mock_create_trainer.return_value = mock_trainer

        # Call function
        result = train_single_model(
            config_path=config_path,
            model_name="tide",
            seed=42,
            checkpoint_dir=tmp_path / "checkpoints",
            tensorboard_dir=tmp_path / "tensorboard",
        )

        # Assertions
        assert result is True
        mock_factory.create_from_config.assert_called_once_with(config_path)
        mock_datamodule.assert_called_once_with(config_path)
        mock_trainer.fit.assert_called_once_with(mock_model, mock_dm)

    @patch("transfer_learning_publication.training_cli.trainer_factory.ModelFactory")
    def test_handles_model_creation_failure(self, mock_factory, config_path, tmp_path):
        """Test handling of model creation failure."""
        mock_factory.create_from_config.side_effect = Exception("Model creation failed")

        result = train_single_model(
            config_path=config_path,
            model_name="tide",
            seed=42,
            checkpoint_dir=tmp_path / "checkpoints",
            tensorboard_dir=tmp_path / "tensorboard",
        )

        assert result is False

    @patch("transfer_learning_publication.training_cli.trainer_factory.ModelFactory")
    @patch("transfer_learning_publication.training_cli.trainer_factory.LSHDataModule")
    def test_handles_datamodule_creation_failure(self, mock_datamodule, mock_factory, config_path, tmp_path):
        """Test handling of datamodule creation failure."""
        mock_factory.create_from_config.return_value = MagicMock()
        mock_datamodule.side_effect = Exception("DataModule creation failed")

        result = train_single_model(
            config_path=config_path,
            model_name="tide",
            seed=42,
            checkpoint_dir=tmp_path / "checkpoints",
            tensorboard_dir=tmp_path / "tensorboard",
        )

        assert result is False

    @patch("transfer_learning_publication.training_cli.trainer_factory.ModelFactory")
    @patch("transfer_learning_publication.training_cli.trainer_factory.LSHDataModule")
    @patch("transfer_learning_publication.training_cli.trainer_factory.create_trainer")
    def test_handles_training_failure(self, mock_create_trainer, mock_datamodule, mock_factory, config_path, tmp_path):
        """Test handling of training failure."""
        mock_factory.create_from_config.return_value = MagicMock()
        mock_datamodule.return_value = MagicMock()

        mock_trainer = MagicMock()
        mock_trainer.fit.side_effect = Exception("Training failed")
        mock_create_trainer.return_value = mock_trainer

        result = train_single_model(
            config_path=config_path,
            model_name="tide",
            seed=42,
            checkpoint_dir=tmp_path / "checkpoints",
            tensorboard_dir=tmp_path / "tensorboard",
        )

        assert result is False

    @patch("transfer_learning_publication.training_cli.trainer_factory.ModelFactory")
    @patch("transfer_learning_publication.training_cli.trainer_factory.LSHDataModule")
    @patch("transfer_learning_publication.training_cli.trainer_factory.create_trainer")
    def test_passes_max_epochs_to_trainer(
        self, mock_create_trainer, mock_datamodule, mock_factory, config_path, tmp_path
    ):
        """Test that max_epochs is passed to trainer creation."""
        mock_factory.create_from_config.return_value = MagicMock()
        mock_datamodule.return_value = MagicMock()
        mock_create_trainer.return_value = MagicMock()

        train_single_model(
            config_path=config_path,
            model_name="tide",
            seed=42,
            checkpoint_dir=tmp_path / "checkpoints",
            tensorboard_dir=tmp_path / "tensorboard",
            max_epochs=75,
        )

        mock_create_trainer.assert_called_once_with(
            model_name="tide",
            seed=42,
            checkpoint_dir=tmp_path / "checkpoints",
            tensorboard_dir=tmp_path / "tensorboard",
            max_epochs=75,
        )

    @patch("transfer_learning_publication.training_cli.trainer_factory.ModelFactory")
    @patch("transfer_learning_publication.training_cli.trainer_factory.LSHDataModule")
    @patch("transfer_learning_publication.training_cli.trainer_factory.create_trainer")
    def test_successful_training_flow(
        self, mock_create_trainer, mock_datamodule, mock_factory, config_path, tmp_path
    ):
        """Test that training flow executes successfully."""
        mock_model = MagicMock()
        mock_factory.create_from_config.return_value = mock_model
        
        mock_dm = MagicMock()
        mock_datamodule.return_value = mock_dm
        
        mock_trainer = MagicMock()
        mock_create_trainer.return_value = mock_trainer

        result = train_single_model(
            config_path=config_path,
            model_name="tide",
            seed=42,
            checkpoint_dir=tmp_path / "checkpoints",
            tensorboard_dir=tmp_path / "tensorboard",
        )

        # Verify the flow executed correctly
        assert result is True
        mock_factory.create_from_config.assert_called_once_with(config_path)
        mock_datamodule.assert_called_once_with(config_path)
        mock_trainer.fit.assert_called_once_with(mock_model, mock_dm)
