"""Tests for the training orchestrator module."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from transfer_learning_publication.training_cli.orchestrator import (
    check_existing_runs,
    create_checkpoint_path,
    is_run_complete,
    run_experiment,
)


class TestCreateCheckpointPath:
    """Test checkpoint path creation."""

    def test_creates_correct_hive_partitioned_path(self):
        """Test that checkpoint path follows hive-partitioned structure."""
        path = create_checkpoint_path(
            model_name="tide", timestamp="2024-11-20", seed=42, base_dir="checkpoints/training"
        )

        expected = Path("checkpoints/training/model_name=tide/run_2024-11-20_seed42")
        assert path == expected

    def test_returns_correct_path_structure(self, tmp_path):
        """Test that correct path structure is returned."""
        base_dir = tmp_path / "checkpoints"

        path = create_checkpoint_path(model_name="ealstm", timestamp="2024-11-20", seed=123, base_dir=str(base_dir))

        expected = base_dir / "model_name=ealstm" / "run_2024-11-20_seed123"
        assert path == expected

    def test_handles_special_characters_in_model_name(self, tmp_path):
        """Test handling of special characters in model names."""
        path = create_checkpoint_path(model_name="tide_v2.0", timestamp="2024-11-20", seed=42, base_dir=str(tmp_path))

        assert "tide_v2.0" in str(path)
        expected = tmp_path / "model_name=tide_v2.0" / "run_2024-11-20_seed42"
        assert path == expected


class TestIsRunComplete:
    """Test run completion checking."""

    def test_returns_false_for_nonexistent_directory(self, tmp_path):
        """Test that non-existent directory returns False."""
        checkpoint_dir = tmp_path / "nonexistent"
        assert not is_run_complete(checkpoint_dir)

    def test_returns_false_when_no_checkpoints_dir(self, tmp_path):
        """Test that missing checkpoints subdirectory returns False."""
        checkpoint_dir = tmp_path / "run"
        checkpoint_dir.mkdir()
        assert not is_run_complete(checkpoint_dir)

    def test_returns_false_when_no_best_checkpoint(self, tmp_path):
        """Test that missing best checkpoint returns False."""
        checkpoint_dir = tmp_path / "run"
        checkpoints_dir = checkpoint_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)

        # Create only last.ckpt, not best checkpoint
        (checkpoints_dir / "last.ckpt").touch()

        assert not is_run_complete(checkpoint_dir)

    def test_returns_true_when_best_checkpoint_exists(self, tmp_path):
        """Test that existing best checkpoint returns True."""
        checkpoint_dir = tmp_path / "run"
        checkpoints_dir = checkpoint_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)

        # Create best checkpoint
        (checkpoints_dir / "best_val_loss_0.0234.ckpt").touch()

        assert is_run_complete(checkpoint_dir)

    def test_handles_multiple_best_checkpoints(self, tmp_path):
        """Test handling of multiple best checkpoints (shouldn't happen but test anyway)."""
        checkpoint_dir = tmp_path / "run"
        checkpoints_dir = checkpoint_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)

        # Create multiple best checkpoints
        (checkpoints_dir / "best_val_loss_0.0234.ckpt").touch()
        (checkpoints_dir / "best_val_loss_0.0199.ckpt").touch()

        assert is_run_complete(checkpoint_dir)


class TestCheckExistingRuns:
    """Test checking for existing completed runs."""

    def test_returns_all_seeds_when_no_runs_complete(self, tmp_path):
        """Test that all seeds are returned when no runs are complete."""
        seeds = [42, 43, 44]
        remaining = check_existing_runs(model_name="tide", timestamp="2024-11-20", seeds=seeds, base_dir=str(tmp_path))

        assert remaining == seeds

    def test_filters_out_completed_runs(self, tmp_path):
        """Test that completed runs are filtered out."""
        seeds = [42, 43, 44]

        # Create completed run for seed 43
        checkpoint_dir = tmp_path / "model_name=tide" / "run_2024-11-20_seed43"
        checkpoints_dir = checkpoint_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        (checkpoints_dir / "best_val_loss_0.0234.ckpt").touch()

        remaining = check_existing_runs(model_name="tide", timestamp="2024-11-20", seeds=seeds, base_dir=str(tmp_path))

        assert remaining == [42, 44]

    def test_returns_empty_list_when_all_complete(self, tmp_path):
        """Test that empty list is returned when all runs are complete."""
        seeds = [42, 43]

        # Create completed runs for all seeds
        for seed in seeds:
            checkpoint_dir = tmp_path / "model_name=tide" / f"run_2024-11-20_seed{seed}"
            checkpoints_dir = checkpoint_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True)
            (checkpoints_dir / f"best_val_loss_0.0{seed}.ckpt").touch()

        remaining = check_existing_runs(model_name="tide", timestamp="2024-11-20", seeds=seeds, base_dir=str(tmp_path))

        assert remaining == []


class TestRunExperiment:
    """Test the main experiment orchestration function."""

    @pytest.fixture
    def experiment_config_file(self, tmp_path):
        """Create a temporary experiment configuration file."""
        config = {"models": {"tide": "configs/tide.yaml", "ealstm": "configs/ealstm.yaml"}}

        config_path = tmp_path / "experiment.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    def test_loads_experiment_config(self, experiment_config_file):
        """Test that experiment configuration is loaded correctly."""
        with patch("transfer_learning_publication.training_cli.orchestrator.train_single_model") as mock_train:
            mock_train.return_value = True

            # Patch is_run_complete to return False so training actually happens
            with patch("transfer_learning_publication.training_cli.orchestrator.is_run_complete") as mock_complete:
                mock_complete.return_value = False

                results = run_experiment(experiment_config_file, models=["tide"], n_runs=1, start_seed=42)

                assert "models" in results
                assert "tide" in results["models"]
                mock_train.assert_called_once()

    def test_filters_models_when_specified(self, experiment_config_file):
        """Test that only specified models are trained."""
        with patch("transfer_learning_publication.training_cli.orchestrator.train_single_model") as mock_train:
            mock_train.return_value = True

            results = run_experiment(
                experiment_config_file,
                models=["tide"],  # Only train tide, not ealstm
                n_runs=1,
                start_seed=42,
            )

            assert "tide" in results["models"]
            assert "ealstm" not in results["models"]
            assert mock_train.call_count == 1

    def test_raises_error_for_invalid_model_names(self, experiment_config_file):
        """Test that invalid model names raise an error."""
        with pytest.raises(ValueError, match="Invalid model names"):
            run_experiment(experiment_config_file, models=["invalid_model"], n_runs=1, start_seed=42)

    def test_trains_all_models_when_none_specified(self, experiment_config_file):
        """Test that all models are trained when none are specified."""
        with patch("transfer_learning_publication.training_cli.orchestrator.train_single_model") as mock_train:
            mock_train.return_value = True

            results = run_experiment(
                experiment_config_file,
                models=None,  # Train all models
                n_runs=1,
                start_seed=42,
            )

            assert "tide" in results["models"]
            assert "ealstm" in results["models"]
            assert mock_train.call_count == 2

    def test_multiple_seeds_training(self, experiment_config_file):
        """Test training with multiple seeds."""
        with patch("transfer_learning_publication.training_cli.orchestrator.train_single_model") as mock_train:
            mock_train.return_value = True

            run_experiment(experiment_config_file, models=["tide"], n_runs=3, start_seed=42)

            assert mock_train.call_count == 3

            # Check that different seeds were used
            seeds_used = [call[1]["seed"] for call in mock_train.call_args_list]
            assert seeds_used == [42, 43, 44]

    def test_skips_completed_runs_when_not_fresh(self, experiment_config_file, tmp_path):
        """Test that completed runs are skipped when fresh=False."""
        with patch("transfer_learning_publication.training_cli.orchestrator.train_single_model") as mock_train:
            mock_train.return_value = True

            # Create a completed run for seed 42
            with patch("transfer_learning_publication.training_cli.orchestrator.is_run_complete") as mock_complete:
                mock_complete.side_effect = [True, False]  # First seed complete, second not

                results = run_experiment(experiment_config_file, models=["tide"], n_runs=2, start_seed=42, fresh=False)

                # Should only train once (seed 43)
                assert mock_train.call_count == 1
                assert results["models"]["tide"]["successful"] == 2  # One skipped, one trained

    def test_ignores_completed_runs_when_fresh(self, experiment_config_file):
        """Test that completed runs are ignored when fresh=True."""
        with patch("transfer_learning_publication.training_cli.orchestrator.train_single_model") as mock_train:
            mock_train.return_value = True

            with patch("transfer_learning_publication.training_cli.orchestrator.is_run_complete") as mock_complete:
                mock_complete.return_value = True  # All runs "complete"

                run_experiment(
                    experiment_config_file,
                    models=["tide"],
                    n_runs=2,
                    start_seed=42,
                    fresh=True,  # Force restart
                )

                # Should train all runs despite being "complete"
                assert mock_train.call_count == 2
                assert mock_complete.call_count == 0  # Shouldn't even check

    def test_handles_training_failures(self, experiment_config_file):
        """Test that training failures are handled gracefully."""
        with patch("transfer_learning_publication.training_cli.orchestrator.train_single_model") as mock_train:
            # First run succeeds, second fails
            mock_train.side_effect = [True, False]

            results = run_experiment(experiment_config_file, models=["tide"], n_runs=2, start_seed=42)

            assert results["models"]["tide"]["successful"] == 1
            assert results["models"]["tide"]["failed_seeds"] == [43]

    def test_handles_training_exceptions(self, experiment_config_file):
        """Test that training exceptions are handled gracefully."""
        with patch("transfer_learning_publication.training_cli.orchestrator.train_single_model") as mock_train:
            mock_train.side_effect = Exception("Training error")

            results = run_experiment(experiment_config_file, models=["tide"], n_runs=1, start_seed=42)

            assert results["models"]["tide"]["successful"] == 0
            assert results["models"]["tide"]["failed_seeds"] == [42]

    def test_calculates_total_time(self, experiment_config_file):
        """Test that total time is calculated and formatted correctly."""
        with patch("transfer_learning_publication.training_cli.orchestrator.train_single_model") as mock_train:
            mock_train.return_value = True

            with patch("time.time") as mock_time:
                # Simulate 1 hour, 23 minutes, 45 seconds
                mock_time.side_effect = [0, 5025]

                results = run_experiment(experiment_config_file, models=["tide"], n_runs=1, start_seed=42)

                assert results["total_time"] == "01:23:45"

    def test_raises_error_for_missing_models_section(self, tmp_path):
        """Test that missing models section raises an error."""
        config_path = tmp_path / "bad_experiment.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"not_models": {}}, f)

        with pytest.raises(ValueError, match="must have a 'models' section"):
            run_experiment(config_path)

    def test_seed_consistency_in_checkpoint_paths(self, experiment_config_file):
        """Test that checkpoint paths correctly include seed values."""
        with patch("transfer_learning_publication.training_cli.orchestrator.train_single_model") as mock_train:
            mock_train.return_value = True

            # Run with multiple seeds
            run_experiment(experiment_config_file, models=["tide"], n_runs=3, start_seed=100)

            # Verify checkpoint paths include correct seeds
            for call in mock_train.call_args_list:
                checkpoint_dir = call[1]["checkpoint_dir"]
                seed = call[1]["seed"]
                assert f"seed{seed}" in str(checkpoint_dir)

    def test_deterministic_seed_sequence(self, experiment_config_file):
        """Test that seeds are generated deterministically."""
        with patch("transfer_learning_publication.training_cli.orchestrator.train_single_model") as mock_train:
            mock_train.return_value = True
            
            # Mock is_run_complete to return False so training happens
            with patch("transfer_learning_publication.training_cli.orchestrator.is_run_complete") as mock_complete:
                mock_complete.return_value = False

                # First run
                run_experiment(experiment_config_file, models=["tide"], n_runs=5, start_seed=42)
                seeds_run1 = [call[1]["seed"] for call in mock_train.call_args_list]

                # Reset mock
                mock_train.reset_mock()

                # Second run with same parameters
                run_experiment(experiment_config_file, models=["tide"], n_runs=5, start_seed=42)
                seeds_run2 = [call[1]["seed"] for call in mock_train.call_args_list]

                # Seeds should be identical and sequential
                assert seeds_run1 == seeds_run2 == [42, 43, 44, 45, 46]

    def test_fresh_flag_overwrites_same_seed_runs(self, experiment_config_file, tmp_path):
        """Test that fresh flag allows rerunning with the same seed."""
        with patch("transfer_learning_publication.training_cli.orchestrator.train_single_model") as mock_train:
            mock_train.return_value = True

            # Mock is_run_complete to always return True (simulating existing runs)
            with patch("transfer_learning_publication.training_cli.orchestrator.is_run_complete") as mock_complete:
                mock_complete.return_value = True

                # Without fresh flag - should skip
                run_experiment(experiment_config_file, models=["tide"], n_runs=1, start_seed=42, fresh=False)
                assert mock_train.call_count == 0

                # With fresh flag - should run despite existing
                run_experiment(experiment_config_file, models=["tide"], n_runs=1, start_seed=42, fresh=True)
                assert mock_train.call_count == 1
                assert mock_train.call_args[1]["seed"] == 42
