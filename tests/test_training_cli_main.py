"""Tests for the training CLI main module."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from transfer_learning_publication.training_cli.__main__ import main


class TestCLIMain:
    """Test the main CLI interface."""

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
        assert "Train models from experiment configuration" in result.output
        assert "--models" in result.output
        assert "--n-runs" in result.output
        assert "--start-seed" in result.output
        assert "--fresh" in result.output

    def test_missing_experiment_file(self, runner):
        """Test error when experiment file doesn't exist."""
        result = runner.invoke(main, ["nonexistent.yaml"])
        assert result.exit_code != 0
        assert "does not exist" in result.output or "Error" in result.output

    @patch("transfer_learning_publication.training_cli.__main__.run_experiment")
    @patch("transfer_learning_publication.training_cli.__main__.print_experiment_summary")
    def test_basic_command(self, mock_summary, mock_run, runner, experiment_file):
        """Test basic command execution."""
        mock_run.return_value = {
            "total_time": "00:05:00",
            "timestamp": "2024-11-20",
            "models": {"tide": {"total": 1, "successful": 1, "failed_seeds": []}},
        }

        result = runner.invoke(main, [experiment_file])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(Path(experiment_file), models=None, n_runs=1, start_seed=42, fresh=False)
        mock_summary.assert_called_once()

    @patch("transfer_learning_publication.training_cli.__main__.run_experiment")
    @patch("transfer_learning_publication.training_cli.__main__.print_experiment_summary")
    def test_models_option(self, mock_summary, mock_run, runner, experiment_file):
        """Test --models option."""
        mock_run.return_value = {"total_time": "00:05:00", "timestamp": "2024-11-20", "models": {}}

        result = runner.invoke(main, [experiment_file, "--models", "tide,ealstm"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            Path(experiment_file), models=["tide", "ealstm"], n_runs=1, start_seed=42, fresh=False
        )

    @patch("transfer_learning_publication.training_cli.__main__.run_experiment")
    @patch("transfer_learning_publication.training_cli.__main__.print_experiment_summary")
    def test_single_model_option(self, mock_summary, mock_run, runner, experiment_file):
        """Test --models option with single model."""
        mock_run.return_value = {"total_time": "00:05:00", "timestamp": "2024-11-20", "models": {}}

        result = runner.invoke(main, [experiment_file, "--models", "tide"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(Path(experiment_file), models=["tide"], n_runs=1, start_seed=42, fresh=False)

    @patch("transfer_learning_publication.training_cli.__main__.run_experiment")
    @patch("transfer_learning_publication.training_cli.__main__.print_experiment_summary")
    def test_n_runs_option(self, mock_summary, mock_run, runner, experiment_file):
        """Test --n-runs option."""
        mock_run.return_value = {"total_time": "00:15:00", "timestamp": "2024-11-20", "models": {}}

        result = runner.invoke(main, [experiment_file, "--n-runs", "5"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(Path(experiment_file), models=None, n_runs=5, start_seed=42, fresh=False)

    @patch("transfer_learning_publication.training_cli.__main__.run_experiment")
    @patch("transfer_learning_publication.training_cli.__main__.print_experiment_summary")
    def test_start_seed_option(self, mock_summary, mock_run, runner, experiment_file):
        """Test --start-seed option."""
        mock_run.return_value = {"total_time": "00:05:00", "timestamp": "2024-11-20", "models": {}}

        result = runner.invoke(main, [experiment_file, "--start-seed", "123"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(Path(experiment_file), models=None, n_runs=1, start_seed=123, fresh=False)

    @patch("transfer_learning_publication.training_cli.__main__.run_experiment")
    @patch("transfer_learning_publication.training_cli.__main__.print_experiment_summary")
    def test_fresh_flag(self, mock_summary, mock_run, runner, experiment_file):
        """Test --fresh flag."""
        mock_run.return_value = {"total_time": "00:05:00", "timestamp": "2024-11-20", "models": {}}

        result = runner.invoke(main, [experiment_file, "--fresh"])

        assert result.exit_code == 0
        mock_run.assert_called_once_with(Path(experiment_file), models=None, n_runs=1, start_seed=42, fresh=True)

    @patch("transfer_learning_publication.training_cli.__main__.run_experiment")
    @patch("transfer_learning_publication.training_cli.__main__.print_experiment_summary")
    def test_combined_options(self, mock_summary, mock_run, runner, experiment_file):
        """Test combining multiple options."""
        mock_run.return_value = {"total_time": "01:00:00", "timestamp": "2024-11-20", "models": {}}

        result = runner.invoke(
            main,
            [experiment_file, "--models", "tide,ealstm,tsmixer", "--n-runs", "10", "--start-seed", "100", "--fresh"],
        )

        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            Path(experiment_file), models=["tide", "ealstm", "tsmixer"], n_runs=10, start_seed=100, fresh=True
        )

    @patch("transfer_learning_publication.training_cli.__main__.run_experiment")
    def test_handles_run_experiment_exception(self, mock_run, runner, experiment_file):
        """Test that exceptions from run_experiment are handled."""
        mock_run.side_effect = ValueError("Invalid configuration")

        result = runner.invoke(main, [experiment_file])

        # Click should handle the exception and return non-zero exit code
        assert result.exit_code != 0
        # The exception is raised but not necessarily shown in output

    @patch("transfer_learning_publication.training_cli.__main__.run_experiment")
    @patch("transfer_learning_publication.training_cli.__main__.print_experiment_summary")
    def test_empty_models_string(self, mock_summary, mock_run, runner, experiment_file):
        """Test that empty models string is handled correctly."""
        mock_run.return_value = {"total_time": "00:05:00", "timestamp": "2024-11-20", "models": {}}

        result = runner.invoke(main, [experiment_file, "--models", ""])

        assert result.exit_code == 0
        # Empty string should result in None (all models) after filtering
        mock_run.assert_called_once_with(
            Path(experiment_file),
            models=None,  # Empty string results in None after filtering
            n_runs=1,
            start_seed=42,
            fresh=False,
        )

    @patch("transfer_learning_publication.training_cli.__main__.run_experiment")
    @patch("transfer_learning_publication.training_cli.__main__.print_experiment_summary")
    def test_models_with_spaces(self, mock_summary, mock_run, runner, experiment_file):
        """Test that models string with spaces is handled correctly."""
        mock_run.return_value = {"total_time": "00:05:00", "timestamp": "2024-11-20", "models": {}}

        result = runner.invoke(main, [experiment_file, "--models", "tide, ealstm, tsmixer"])

        assert result.exit_code == 0
        # Spaces are stripped after split
        mock_run.assert_called_once_with(
            Path(experiment_file),
            models=["tide", "ealstm", "tsmixer"],  # Spaces are stripped
            n_runs=1,
            start_seed=42,
            fresh=False,
        )

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
