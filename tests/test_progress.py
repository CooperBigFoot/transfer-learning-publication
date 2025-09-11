"""Tests for the progress reporting module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import torch

from transfer_learning_publication.callbacks.progress import (
    MinimalProgressCallback,
    print_experiment_summary,
)


class TestMinimalProgressCallback:
    """Test the minimal progress callback."""

    @pytest.fixture
    def callback(self):
        """Create a callback instance."""
        return MinimalProgressCallback(model_name="tide", seed=42)

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer."""
        trainer = MagicMock()
        trainer.current_epoch = 0
        trainer.max_epochs = 100
        trainer.callback_metrics = {}
        return trainer

    @pytest.fixture
    def mock_module(self):
        """Create a mock Lightning module."""
        return MagicMock()

    def test_initialization(self):
        """Test callback initialization."""
        callback = MinimalProgressCallback(model_name="ealstm", seed=123)
        assert callback.model_name == "ealstm"
        assert callback.seed == 123
        assert callback.start_time is None

    def test_on_train_start_records_time(self, callback, mock_trainer, mock_module):
        """Test that training start time is recorded."""
        callback.on_train_start(mock_trainer, mock_module)
        assert callback.start_time is not None
        assert isinstance(callback.start_time, datetime)

    @patch("transfer_learning_publication.callbacks.progress.logger")
    @patch("transfer_learning_publication.callbacks.progress.datetime")
    def test_on_train_epoch_start_logs_progress(self, mock_datetime, mock_logger, callback, mock_trainer, mock_module):
        """Test that epoch start logs progress."""
        mock_datetime.now.return_value.strftime.return_value = "2024-11-20 14:30:00"
        mock_trainer.current_epoch = 5
        mock_trainer.max_epochs = 100

        callback.on_train_epoch_start(mock_trainer, mock_module)

        mock_logger.info.assert_called_once()
        logged = mock_logger.info.call_args[0][0]
        assert "[2024-11-20 14:30:00]" in logged
        assert "model_name=tide" in logged
        assert "seed=42" in logged
        assert "epoch 6/100" in logged  # current_epoch + 1

    @patch("transfer_learning_publication.callbacks.progress.logger")
    @patch("transfer_learning_publication.callbacks.progress.datetime")
    def test_on_validation_epoch_end_logs_val_loss(
        self, mock_datetime, mock_logger, callback, mock_trainer, mock_module
    ):
        """Test that validation epoch end logs validation loss."""
        mock_datetime.now.return_value.strftime.return_value = "2024-11-20 14:31:00"
        mock_trainer.current_epoch = 5
        mock_trainer.callback_metrics = {"val_loss": torch.tensor(0.0234)}

        callback.on_validation_epoch_end(mock_trainer, mock_module)

        mock_logger.info.assert_called_once()
        logged = mock_logger.info.call_args[0][0]
        assert "[2024-11-20 14:31:00]" in logged
        assert "Validation:" in logged
        assert "model_name=tide" in logged
        assert "seed=42" in logged
        assert "epoch 6" in logged
        assert "val_loss=0.0234" in logged

    @patch("transfer_learning_publication.callbacks.progress.logger")
    def test_on_validation_epoch_end_no_log_without_val_loss(self, mock_logger, callback, mock_trainer, mock_module):
        """Test that nothing is logged if val_loss is not available."""
        mock_trainer.callback_metrics = {}  # No val_loss

        callback.on_validation_epoch_end(mock_trainer, mock_module)

        mock_logger.info.assert_not_called()

    @patch("transfer_learning_publication.callbacks.progress.logger")
    def test_on_validation_epoch_end_handles_none_val_loss(self, mock_logger, callback, mock_trainer, mock_module):
        """Test handling of None val_loss."""
        mock_trainer.callback_metrics = {"val_loss": None}

        callback.on_validation_epoch_end(mock_trainer, mock_module)

        mock_logger.info.assert_not_called()


class TestPrintExperimentSummary:
    """Test the experiment summary printing function."""

    @patch("transfer_learning_publication.callbacks.progress.logger")
    def test_logs_complete_summary(self, mock_logger):
        """Test logging a complete experiment summary."""
        results = {
            "total_time": "01:23:45",
            "timestamp": "2024-11-20",
            "models": {
                "tide": {"total": 3, "successful": 3, "failed_seeds": []},
                "ealstm": {"total": 3, "successful": 2, "failed_seeds": [44]},
            },
        }

        print_experiment_summary(results)

        # Check that logger.info was called multiple times
        assert mock_logger.info.call_count > 0

        # Collect all logged strings
        logged_lines = [call[0][0] if call[0] else "" for call in mock_logger.info.call_args_list]
        all_output = "\n".join(logged_lines)

        # Check for expected content
        assert "Experiment Complete!" in all_output
        assert "✓ tide: 3/3 runs successful" in all_output
        assert "⚠ ealstm: 2/3 runs successful (failed: seed44)" in all_output
        assert "Total time: 01:23:45" in all_output
        assert "Experiment timestamp: 2024-11-20" in all_output
        assert "Checkpoints saved to: checkpoints/training/" in all_output
        assert "TensorBoard logs: tensorboard/" in all_output
        assert "tensorboard --logdir tensorboard/" in all_output

    @patch("transfer_learning_publication.callbacks.progress.logger")
    def test_handles_all_successful_runs(self, mock_logger):
        """Test summary when all runs are successful."""
        results = {
            "total_time": "00:15:30",
            "timestamp": "2024-11-20",
            "models": {"tide": {"total": 5, "successful": 5, "failed_seeds": []}},
        }

        print_experiment_summary(results)

        logged_lines = [call[0][0] if call[0] else "" for call in mock_logger.info.call_args_list]
        all_output = "\n".join(logged_lines)

        assert "✓ tide: 5/5 runs successful" in all_output
        assert "failed" not in all_output.lower() or "failed_seeds" in all_output.lower()

    @patch("transfer_learning_publication.callbacks.progress.logger")
    def test_handles_all_failed_runs(self, mock_logger):
        """Test summary when all runs fail."""
        results = {
            "total_time": "00:02:15",
            "timestamp": "2024-11-20",
            "models": {"tide": {"total": 3, "successful": 0, "failed_seeds": [42, 43, 44]}},
        }

        print_experiment_summary(results)

        logged_lines = [call[0][0] if call[0] else "" for call in mock_logger.info.call_args_list]
        all_output = "\n".join(logged_lines)

        assert "⚠ tide: 0/3 runs successful (failed: seed42, seed43, seed44)" in all_output

    @patch("transfer_learning_publication.callbacks.progress.logger")
    def test_handles_multiple_models(self, mock_logger):
        """Test summary with multiple models."""
        results = {
            "total_time": "02:45:00",
            "timestamp": "2024-11-20",
            "models": {
                "tide": {"total": 2, "successful": 2, "failed_seeds": []},
                "ealstm": {"total": 2, "successful": 1, "failed_seeds": [43]},
                "tsmixer": {"total": 2, "successful": 2, "failed_seeds": []},
                "tft": {"total": 2, "successful": 0, "failed_seeds": [42, 43]},
            },
        }

        print_experiment_summary(results)

        logged_lines = [call[0][0] if call[0] else "" for call in mock_logger.info.call_args_list]
        all_output = "\n".join(logged_lines)

        # Check all models are mentioned
        assert "tide" in all_output
        assert "ealstm" in all_output
        assert "tsmixer" in all_output
        assert "tft" in all_output

        # Check statuses
        assert "✓ tide: 2/2" in all_output
        assert "⚠ ealstm: 1/2" in all_output
        assert "✓ tsmixer: 2/2" in all_output
        assert "⚠ tft: 0/2" in all_output

    @patch("transfer_learning_publication.callbacks.progress.logger")
    def test_handles_missing_failed_seeds_key(self, mock_logger):
        """Test handling of missing failed_seeds key (backward compatibility)."""
        results = {
            "total_time": "00:30:00",
            "timestamp": "2024-11-20",
            "models": {
                "tide": {
                    "total": 2,
                    "successful": 2,
                    # No failed_seeds key
                }
            },
        }

        # Should not raise an error
        print_experiment_summary(results)

        logged_lines = [call[0][0] if call[0] else "" for call in mock_logger.info.call_args_list]
        all_output = "\n".join(logged_lines)

        assert "✓ tide: 2/2 runs successful" in all_output

    @patch("transfer_learning_publication.callbacks.progress.logger")
    def test_formatting_of_output(self, mock_logger):
        """Test that output is properly formatted with separators."""
        results = {
            "total_time": "00:05:00",
            "timestamp": "2024-11-20",
            "models": {"tide": {"total": 1, "successful": 1, "failed_seeds": []}},
        }

        print_experiment_summary(results)

        logged_lines = [call[0][0] if call[0] else "" for call in mock_logger.info.call_args_list]

        # Check for separator lines
        assert any("=" * 60 in line for line in logged_lines)

        # Check structure
        assert logged_lines[0] == "\n" + "=" * 60  # Empty line before separator
        assert "Experiment Complete!" in logged_lines[1]
        assert "=" * 60 in logged_lines[2]  # Second separator
