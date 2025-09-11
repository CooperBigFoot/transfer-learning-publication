"""Tests for the fine-tuning CLI main module."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from transfer_learning_publication.finetuning_cli.__main__ import main


class TestFinetuningCLIMain:
    """Test the fine-tuning CLI interface."""

    @pytest.fixture
    def runner(self):
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def experiment_file(self, tmp_path):
        """Create a temporary experiment file."""
        exp_file = tmp_path / "experiment.yaml"
        exp_file.write_text("models:\n  tide: configs/tide.yaml\n")
        return str(exp_file)

    def test_help_option(self, runner):
        """Test that --help works."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Fine-tune models from experiment configuration" in result.output
        assert "--models" in result.output
        assert "--lr-reduction" in result.output
        assert "--base-seed" in result.output
        assert "--n-runs" in result.output
        assert "--start-seed" in result.output
        assert "--max-epochs" in result.output
        assert "--fresh" in result.output

    def test_missing_experiment_file(self, runner):
        """Test error when experiment file doesn't exist."""
        result = runner.invoke(main, ["nonexistent.yaml"])
        assert result.exit_code != 0
        assert "does not exist" in result.output or "Error" in result.output

    @patch("transfer_learning_publication.finetuning_cli.__main__.run_finetuning")
    @patch("transfer_learning_publication.finetuning_cli.__main__._print_finetuning_summary")
    def test_basic_command(self, mock_summary, mock_run, runner, experiment_file):
        """Test basic command execution."""
        mock_run.return_value = {
            "total_time": "00:05:00",
            "timestamp": "2024-11-20",
            "lr_reduction": 25.0,
            "models": {"tide": {"total": 1, "successful": 1, "failed_seeds": []}},
        }

        result = runner.invoke(main, [experiment_file])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            experiment_path=Path(experiment_file),
            models=None,
            lr_reduction=25.0,
            base_seed=None,
            n_runs=1,
            start_seed=42,
            max_epochs=None,
            fresh=False,
        )
        mock_summary.assert_called_once()

    @patch("transfer_learning_publication.finetuning_cli.__main__.run_finetuning")
    @patch("transfer_learning_publication.finetuning_cli.__main__._print_finetuning_summary")
    def test_lr_reduction_option(self, mock_summary, mock_run, runner, experiment_file):
        """Test --lr-reduction option."""
        mock_run.return_value = {
            "total_time": "00:05:00",
            "timestamp": "2024-11-20",
            "lr_reduction": 10.0,
            "models": {},
        }

        result = runner.invoke(main, [experiment_file, "--lr-reduction", "10"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            experiment_path=Path(experiment_file),
            models=None,
            lr_reduction=10.0,
            base_seed=None,
            n_runs=1,
            start_seed=42,
            max_epochs=None,
            fresh=False,
        )

    @patch("transfer_learning_publication.finetuning_cli.__main__.run_finetuning")
    @patch("transfer_learning_publication.finetuning_cli.__main__._print_finetuning_summary")
    def test_base_seed_option(self, mock_summary, mock_run, runner, experiment_file):
        """Test --base-seed option."""
        mock_run.return_value = {
            "total_time": "00:05:00",
            "timestamp": "2024-11-20",
            "lr_reduction": 25.0,
            "models": {},
        }

        result = runner.invoke(main, [experiment_file, "--base-seed", "100"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            experiment_path=Path(experiment_file),
            models=None,
            lr_reduction=25.0,
            base_seed=100,
            n_runs=1,
            start_seed=42,
            max_epochs=None,
            fresh=False,
        )

    @patch("transfer_learning_publication.finetuning_cli.__main__.run_finetuning")
    @patch("transfer_learning_publication.finetuning_cli.__main__._print_finetuning_summary")
    def test_max_epochs_option(self, mock_summary, mock_run, runner, experiment_file):
        """Test --max-epochs option."""
        mock_run.return_value = {
            "total_time": "00:05:00",
            "timestamp": "2024-11-20",
            "lr_reduction": 25.0,
            "models": {},
        }

        result = runner.invoke(main, [experiment_file, "--max-epochs", "50"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            experiment_path=Path(experiment_file),
            models=None,
            lr_reduction=25.0,
            base_seed=None,
            n_runs=1,
            start_seed=42,
            max_epochs=50,
            fresh=False,
        )

    @patch("transfer_learning_publication.finetuning_cli.__main__.run_finetuning")
    @patch("transfer_learning_publication.finetuning_cli.__main__._print_finetuning_summary")
    def test_models_option(self, mock_summary, mock_run, runner, experiment_file):
        """Test --models option."""
        mock_run.return_value = {
            "total_time": "00:05:00",
            "timestamp": "2024-11-20",
            "lr_reduction": 25.0,
            "models": {},
        }

        result = runner.invoke(main, [experiment_file, "--models", "tide,ealstm"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            experiment_path=Path(experiment_file),
            models=["tide", "ealstm"],
            lr_reduction=25.0,
            base_seed=None,
            n_runs=1,
            start_seed=42,
            max_epochs=None,
            fresh=False,
        )

    @patch("transfer_learning_publication.finetuning_cli.__main__.run_finetuning")
    @patch("transfer_learning_publication.finetuning_cli.__main__._print_finetuning_summary")
    def test_single_model_option(self, mock_summary, mock_run, runner, experiment_file):
        """Test --models option with single model."""
        mock_run.return_value = {
            "total_time": "00:05:00",
            "timestamp": "2024-11-20",
            "lr_reduction": 25.0,
            "models": {},
        }

        result = runner.invoke(main, [experiment_file, "--models", "tide"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            experiment_path=Path(experiment_file),
            models=["tide"],
            lr_reduction=25.0,
            base_seed=None,
            n_runs=1,
            start_seed=42,
            max_epochs=None,
            fresh=False,
        )

    @patch("transfer_learning_publication.finetuning_cli.__main__.run_finetuning")
    @patch("transfer_learning_publication.finetuning_cli.__main__._print_finetuning_summary")
    def test_n_runs_option(self, mock_summary, mock_run, runner, experiment_file):
        """Test --n-runs option."""
        mock_run.return_value = {
            "total_time": "00:15:00",
            "timestamp": "2024-11-20",
            "lr_reduction": 25.0,
            "models": {},
        }

        result = runner.invoke(main, [experiment_file, "--n-runs", "5"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            experiment_path=Path(experiment_file),
            models=None,
            lr_reduction=25.0,
            base_seed=None,
            n_runs=5,
            start_seed=42,
            max_epochs=None,
            fresh=False,
        )

    @patch("transfer_learning_publication.finetuning_cli.__main__.run_finetuning")
    @patch("transfer_learning_publication.finetuning_cli.__main__._print_finetuning_summary")
    def test_start_seed_option(self, mock_summary, mock_run, runner, experiment_file):
        """Test --start-seed option."""
        mock_run.return_value = {
            "total_time": "00:05:00",
            "timestamp": "2024-11-20",
            "lr_reduction": 25.0,
            "models": {},
        }

        result = runner.invoke(main, [experiment_file, "--start-seed", "123"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            experiment_path=Path(experiment_file),
            models=None,
            lr_reduction=25.0,
            base_seed=None,
            n_runs=1,
            start_seed=123,
            max_epochs=None,
            fresh=False,
        )

    @patch("transfer_learning_publication.finetuning_cli.__main__.run_finetuning")
    @patch("transfer_learning_publication.finetuning_cli.__main__._print_finetuning_summary")
    def test_fresh_flag(self, mock_summary, mock_run, runner, experiment_file):
        """Test --fresh flag."""
        mock_run.return_value = {
            "total_time": "00:05:00",
            "timestamp": "2024-11-20",
            "lr_reduction": 25.0,
            "models": {},
        }

        result = runner.invoke(main, [experiment_file, "--fresh"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            experiment_path=Path(experiment_file),
            models=None,
            lr_reduction=25.0,
            base_seed=None,
            n_runs=1,
            start_seed=42,
            max_epochs=None,
            fresh=True,
        )

    @patch("transfer_learning_publication.finetuning_cli.__main__.run_finetuning")
    @patch("transfer_learning_publication.finetuning_cli.__main__._print_finetuning_summary")
    def test_combined_options(self, mock_summary, mock_run, runner, experiment_file):
        """Test combining multiple options."""
        mock_run.return_value = {
            "total_time": "01:00:00",
            "timestamp": "2024-11-20",
            "lr_reduction": 10.0,
            "models": {},
        }

        result = runner.invoke(
            main,
            [
                experiment_file,
                "--models",
                "tide,ealstm",
                "--lr-reduction",
                "10",
                "--base-seed",
                "42",
                "--n-runs",
                "3",
                "--start-seed",
                "100",
                "--max-epochs",
                "50",
                "--fresh",
            ],
        )

        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            experiment_path=Path(experiment_file),
            models=["tide", "ealstm"],
            lr_reduction=10.0,
            base_seed=42,
            n_runs=3,
            start_seed=100,
            max_epochs=50,
            fresh=True,
        )

    @patch("transfer_learning_publication.finetuning_cli.__main__.run_finetuning")
    def test_handles_run_finetuning_exception(self, mock_run, runner, experiment_file):
        """Test that exceptions from run_finetuning are handled."""
        mock_run.side_effect = ValueError("Invalid configuration")

        result = runner.invoke(main, [experiment_file])

        # Click should handle the exception and return non-zero exit code
        assert result.exit_code != 0

    @patch("transfer_learning_publication.finetuning_cli.__main__.run_finetuning")
    @patch("transfer_learning_publication.finetuning_cli.__main__._print_finetuning_summary")
    def test_empty_models_string(self, mock_summary, mock_run, runner, experiment_file):
        """Test that empty models string is handled correctly."""
        mock_run.return_value = {
            "total_time": "00:05:00",
            "timestamp": "2024-11-20",
            "lr_reduction": 25.0,
            "models": {},
        }

        result = runner.invoke(main, [experiment_file, "--models", ""])

        assert result.exit_code == 0
        # Empty string should result in None (all models) after filtering
        mock_run.assert_called_once_with(
            experiment_path=Path(experiment_file),
            models=None,  # Empty string results in None after filtering
            lr_reduction=25.0,
            base_seed=None,
            n_runs=1,
            start_seed=42,
            max_epochs=None,
            fresh=False,
        )

    @patch("transfer_learning_publication.finetuning_cli.__main__.run_finetuning")
    @patch("transfer_learning_publication.finetuning_cli.__main__._print_finetuning_summary")
    def test_models_with_spaces(self, mock_summary, mock_run, runner, experiment_file):
        """Test that models string with spaces is handled correctly."""
        mock_run.return_value = {
            "total_time": "00:05:00",
            "timestamp": "2024-11-20",
            "lr_reduction": 25.0,
            "models": {},
        }

        result = runner.invoke(main, [experiment_file, "--models", "tide, ealstm, tsmixer"])

        assert result.exit_code == 0
        # Spaces are stripped after split
        mock_run.assert_called_once_with(
            experiment_path=Path(experiment_file),
            models=["tide", "ealstm", "tsmixer"],  # Spaces are stripped
            lr_reduction=25.0,
            base_seed=None,
            n_runs=1,
            start_seed=42,
            max_epochs=None,
            fresh=False,
        )

    def test_invalid_lr_reduction_value(self, runner, experiment_file):
        """Test that invalid lr-reduction value is rejected."""
        result = runner.invoke(main, [experiment_file, "--lr-reduction", "not_a_number"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "Error" in result.output

    def test_invalid_n_runs_value(self, runner, experiment_file):
        """Test that invalid n-runs value is rejected."""
        result = runner.invoke(main, [experiment_file, "--n-runs", "not_a_number"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "Error" in result.output

    def test_invalid_start_seed_value(self, runner, experiment_file):
        """Test that invalid start-seed value is rejected."""
        result = runner.invoke(main, [experiment_file, "--start-seed", "not_a_number"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "Error" in result.output

    def test_invalid_base_seed_value(self, runner, experiment_file):
        """Test that invalid base-seed value is rejected."""
        result = runner.invoke(main, [experiment_file, "--base-seed", "not_a_number"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "Error" in result.output

    def test_invalid_max_epochs_value(self, runner, experiment_file):
        """Test that invalid max-epochs value is rejected."""
        result = runner.invoke(main, [experiment_file, "--max-epochs", "not_a_number"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "Error" in result.output
