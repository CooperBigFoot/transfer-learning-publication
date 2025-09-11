"""Tests for the fine-tuning orchestrator module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from transfer_learning_publication.checkpoint_utils import Checkpoint
from transfer_learning_publication.finetuning_cli.finetune_orchestrator import (
    create_finetuning_checkpoint_path,
    finetune_single_model,
    is_finetuning_complete,
    run_finetuning,
    save_finetune_metadata,
)


class TestCreateFinetuningCheckpointPath:
    """Test fine-tuning checkpoint path creation."""

    def test_creates_correct_hive_partitioned_path(self):
        """Test that checkpoint path follows hive-partitioned structure."""
        path = create_finetuning_checkpoint_path(
            model_name="tide", timestamp="2024-11-20", seed=42, base_dir="checkpoints/finetuning"
        )

        expected = Path("checkpoints/finetuning/model_name=tide/run_2024-11-20_seed42")
        assert path == expected

    def test_returns_correct_path_structure(self, tmp_path):
        """Test that correct path structure is returned."""
        base_dir = tmp_path / "checkpoints"

        path = create_finetuning_checkpoint_path(
            model_name="ealstm", timestamp="2024-11-20", seed=123, base_dir=str(base_dir)
        )

        expected = base_dir / "model_name=ealstm" / "run_2024-11-20_seed123"
        assert path == expected

    def test_handles_special_characters_in_model_name(self, tmp_path):
        """Test handling of special characters in model names."""
        path = create_finetuning_checkpoint_path(
            model_name="tide_v2.0", timestamp="2024-11-20", seed=42, base_dir=str(tmp_path)
        )

        assert "tide_v2.0" in str(path)
        expected = tmp_path / "model_name=tide_v2.0" / "run_2024-11-20_seed42"
        assert path == expected


class TestIsFinetuningComplete:
    """Test fine-tuning completion checking."""

    def test_returns_false_for_nonexistent_directory(self, tmp_path):
        """Test that non-existent directory returns False."""
        checkpoint_dir = tmp_path / "nonexistent"
        assert not is_finetuning_complete(checkpoint_dir)

    def test_returns_false_when_no_checkpoints_dir(self, tmp_path):
        """Test that missing checkpoints subdirectory returns False."""
        checkpoint_dir = tmp_path / "run"
        checkpoint_dir.mkdir()
        assert not is_finetuning_complete(checkpoint_dir)

    def test_returns_false_when_no_best_checkpoint(self, tmp_path):
        """Test that missing best checkpoint returns False."""
        checkpoint_dir = tmp_path / "run"
        checkpoints_dir = checkpoint_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)

        # Create only last.ckpt, not best checkpoint
        (checkpoints_dir / "last.ckpt").touch()

        assert not is_finetuning_complete(checkpoint_dir)

    def test_returns_true_when_best_checkpoint_exists(self, tmp_path):
        """Test that existing best checkpoint returns True."""
        checkpoint_dir = tmp_path / "run"
        checkpoints_dir = checkpoint_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)

        # Create best checkpoint
        (checkpoints_dir / "best_val_loss_0.0234.ckpt").touch()

        assert is_finetuning_complete(checkpoint_dir)

    def test_handles_multiple_best_checkpoints(self, tmp_path):
        """Test handling of multiple best checkpoints."""
        checkpoint_dir = tmp_path / "run"
        checkpoints_dir = checkpoint_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)

        # Create multiple best checkpoints
        (checkpoints_dir / "best_val_loss_0.0234.ckpt").touch()
        (checkpoints_dir / "best_val_loss_0.0199.ckpt").touch()

        assert is_finetuning_complete(checkpoint_dir)


class TestSaveFinetuneMetadata:
    """Test fine-tuning metadata saving."""

    def test_creates_metadata_file(self, tmp_path):
        """Test that metadata file is created."""
        checkpoint_dir = tmp_path / "checkpoint"

        save_finetune_metadata(
            checkpoint_dir=checkpoint_dir,
            base_checkpoint_path=Path("/path/to/base.ckpt"),
            base_val_loss=0.025,
            lr_reduction_factor=10.0,
            original_lr=0.001,
            finetuned_lr=0.0001,
            seed=42,
            experiment_config="experiment.yaml",
        )

        metadata_file = checkpoint_dir / "finetune_metadata.yaml"
        assert metadata_file.exists()

    def test_metadata_contains_all_fields(self, tmp_path):
        """Test that metadata contains all required fields."""
        checkpoint_dir = tmp_path / "checkpoint"

        save_finetune_metadata(
            checkpoint_dir=checkpoint_dir,
            base_checkpoint_path=Path("/path/to/base.ckpt"),
            base_val_loss=0.025,
            lr_reduction_factor=10.0,
            original_lr=0.001,
            finetuned_lr=0.0001,
            seed=42,
            experiment_config="experiment.yaml",
        )

        metadata_file = checkpoint_dir / "finetune_metadata.yaml"
        with open(metadata_file) as f:
            metadata = yaml.safe_load(f)

        assert metadata["base_checkpoint"] == "/path/to/base.ckpt"
        assert metadata["base_val_loss"] == 0.025
        assert metadata["lr_reduction_factor"] == 10.0
        assert metadata["original_lr"] == 0.001
        assert metadata["finetuned_lr"] == 0.0001
        assert metadata["seed"] == 42
        assert metadata["experiment_config"] == "experiment.yaml"
        assert "timestamp" in metadata

    def test_creates_directory_if_not_exists(self, tmp_path):
        """Test that directory is created if it doesn't exist."""
        checkpoint_dir = tmp_path / "new" / "nested" / "dir"

        save_finetune_metadata(
            checkpoint_dir=checkpoint_dir,
            base_checkpoint_path=Path("/path/to/base.ckpt"),
            base_val_loss=None,
            lr_reduction_factor=25.0,
            original_lr=0.001,
            finetuned_lr=0.00004,
            seed=100,
            experiment_config="exp.yaml",
        )

        assert checkpoint_dir.exists()
        metadata_file = checkpoint_dir / "finetune_metadata.yaml"
        assert metadata_file.exists()


class TestFinetuneSingleModel:
    """Test single model fine-tuning function."""

    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.pl.seed_everything")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.ModelFactory")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.LSHDataModule")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.create_trainer")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.save_finetune_metadata")
    def test_loads_model_from_checkpoint(
        self, mock_save_metadata, mock_create_trainer, mock_datamodule, mock_factory, mock_seed, tmp_path
    ):
        """Test that model is loaded from checkpoint."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.config.learning_rate = 0.001
        mock_factory.create_from_checkpoint.return_value = mock_model

        mock_trainer = MagicMock()
        mock_create_trainer.return_value = mock_trainer

        # Call function
        result = finetune_single_model(
            base_checkpoint_path=Path("/path/to/checkpoint.ckpt"),
            config_path=Path("config.yaml"),
            model_name="tide",
            seed=42,
            checkpoint_dir=tmp_path / "checkpoint",
            lr_reduction=10.0,
            max_epochs=50,
            base_val_loss=0.025,
        )

        # Verify
        assert result is True
        mock_seed.assert_called_once_with(42, workers=True)
        mock_factory.create_from_checkpoint.assert_called_once_with("tide", Path("/path/to/checkpoint.ckpt"))

    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.pl.seed_everything")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.ModelFactory")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.LSHDataModule")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.create_trainer")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.save_finetune_metadata")
    def test_modifies_learning_rate_correctly(
        self, mock_save_metadata, mock_create_trainer, mock_datamodule, mock_factory, mock_seed, tmp_path
    ):
        """Test that learning rate is modified correctly."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.config.learning_rate = 0.001
        mock_factory.create_from_checkpoint.return_value = mock_model

        mock_trainer = MagicMock()
        mock_create_trainer.return_value = mock_trainer

        # Call function
        finetune_single_model(
            base_checkpoint_path=Path("/path/to/checkpoint.ckpt"),
            config_path=Path("config.yaml"),
            model_name="tide",
            seed=42,
            checkpoint_dir=tmp_path / "checkpoint",
            lr_reduction=25.0,
            max_epochs=50,
            base_val_loss=0.025,
        )

        # Verify learning rate was modified
        assert mock_model.config.learning_rate == 0.001 / 25.0

    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.pl.seed_everything")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.ModelFactory")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.LSHDataModule")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.create_trainer")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.save_finetune_metadata")
    def test_saves_metadata_before_training(
        self, mock_save_metadata, mock_create_trainer, mock_datamodule, mock_factory, mock_seed, tmp_path
    ):
        """Test that metadata is saved before training starts."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.config.learning_rate = 0.001
        mock_factory.create_from_checkpoint.return_value = mock_model

        mock_trainer = MagicMock()
        mock_create_trainer.return_value = mock_trainer

        checkpoint_dir = tmp_path / "checkpoint"

        # Call function
        finetune_single_model(
            base_checkpoint_path=Path("/path/to/checkpoint.ckpt"),
            config_path=Path("config.yaml"),
            model_name="tide",
            seed=42,
            checkpoint_dir=checkpoint_dir,
            lr_reduction=10.0,
            max_epochs=50,
            base_val_loss=0.025,
        )

        # Verify metadata was saved
        mock_save_metadata.assert_called_once_with(
            checkpoint_dir=checkpoint_dir,
            base_checkpoint_path=Path("/path/to/checkpoint.ckpt"),
            base_val_loss=0.025,
            lr_reduction_factor=10.0,
            original_lr=0.001,
            finetuned_lr=0.0001,
            seed=42,
            experiment_config="config.yaml",
        )

    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.pl.seed_everything")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.ModelFactory")
    def test_returns_false_on_failure(self, mock_factory, mock_seed, tmp_path):
        """Test that function returns False on failure."""
        # Setup mock to raise exception
        mock_factory.create_from_checkpoint.side_effect = RuntimeError("Failed to load")

        # Call function
        result = finetune_single_model(
            base_checkpoint_path=Path("/path/to/checkpoint.ckpt"),
            config_path=Path("config.yaml"),
            model_name="tide",
            seed=42,
            checkpoint_dir=tmp_path / "checkpoint",
            lr_reduction=10.0,
            max_epochs=50,
        )

        # Verify
        assert result is False

    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.pl.seed_everything")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.ModelFactory")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.LSHDataModule")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.create_trainer")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.save_finetune_metadata")
    def test_sets_seed_before_loading(
        self, mock_save_metadata, mock_create_trainer, mock_datamodule, mock_factory, mock_seed, tmp_path
    ):
        """Test that seed is set before loading model."""
        call_order = []

        def record_seed(*args, **kwargs):
            call_order.append("seed")

        def record_load(*args, **kwargs):
            call_order.append("load")
            mock_model = MagicMock()
            mock_model.config.learning_rate = 0.001
            return mock_model

        mock_seed.side_effect = record_seed
        mock_factory.create_from_checkpoint.side_effect = record_load

        mock_trainer = MagicMock()
        mock_create_trainer.return_value = mock_trainer

        # Call function
        finetune_single_model(
            base_checkpoint_path=Path("/path/to/checkpoint.ckpt"),
            config_path=Path("config.yaml"),
            model_name="tide",
            seed=42,
            checkpoint_dir=tmp_path / "checkpoint",
            lr_reduction=10.0,
            max_epochs=50,
        )

        # Verify seed was set before loading
        assert call_order == ["seed", "load"]


class TestRunFinetuning:
    """Test the main fine-tuning orchestration function."""

    @pytest.fixture
    def experiment_config_file(self, tmp_path):
        """Create a temporary experiment configuration file."""
        config = {
            "models": {"tide": "configs/tide.yaml", "ealstm": "configs/ealstm.yaml"},
            "trainer": {"max_epochs": 100},
        }

        config_path = tmp_path / "experiment.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    def test_loads_experiment_config(self, experiment_config_file):
        """Test that experiment config is loaded correctly."""
        with patch(
            "transfer_learning_publication.finetuning_cli.finetune_orchestrator.CheckpointDiscovery"
        ) as mock_discovery_class:
            mock_discovery = MagicMock()
            mock_discovery_class.return_value = mock_discovery

            # Mock checkpoint with proper val_loss attribute
            mock_checkpoint = Checkpoint(
                path=Path("/path/to/checkpoint.ckpt"),
                model_name="tide",
                timestamp="2024-11-20",
                seed=42,
                checkpoint_type="best_val_loss",
                val_loss=0.025,
            )
            mock_discovery.get_median_checkpoint.return_value = mock_checkpoint

            results = run_finetuning(experiment_config_file, models=[], n_runs=0)

            # Should have loaded the config (even if no models were processed)
            assert "models" in results
            assert "total_time" in results
            assert "timestamp" in results
            assert "lr_reduction" in results

    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.CheckpointDiscovery")
    def test_filters_models_correctly(self, mock_discovery_class, experiment_config_file):
        """Test that models are filtered correctly when specified."""
        mock_discovery = MagicMock()
        mock_discovery_class.return_value = mock_discovery
        mock_discovery.get_median_checkpoint.return_value = None

        results = run_finetuning(experiment_config_file, models=["tide"], n_runs=1)

        # Should only process tide model
        assert "tide" in results["models"]
        assert "ealstm" not in results["models"]

    def test_raises_error_for_invalid_models(self, experiment_config_file):
        """Test that error is raised for invalid model names."""
        with pytest.raises(ValueError, match="Invalid model names"):
            run_finetuning(experiment_config_file, models=["invalid_model"], n_runs=1)

    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.CheckpointDiscovery")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.finetune_single_model")
    def test_uses_checkpoint_discovery(self, mock_finetune, mock_discovery_class, experiment_config_file):
        """Test that checkpoint discovery is used to find base checkpoints."""
        # Setup mocks
        mock_discovery = MagicMock()
        mock_discovery_class.return_value = mock_discovery

        mock_checkpoint = Checkpoint(
            path=Path("/path/to/checkpoint.ckpt"),
            model_name="tide",
            timestamp="2024-11-20",
            seed=42,
            checkpoint_type="best_val_loss",
            val_loss=0.025,
        )
        mock_discovery.get_median_checkpoint.return_value = mock_checkpoint

        mock_finetune.return_value = True

        # Call function
        run_finetuning(experiment_config_file, models=["tide"], n_runs=1)

        # Verify discovery was used
        mock_discovery.get_median_checkpoint.assert_called_once_with("tide", stage="training")

    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.CheckpointDiscovery")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.finetune_single_model")
    def test_uses_specific_seed_when_provided(self, mock_finetune, mock_discovery_class, experiment_config_file):
        """Test that specific seed checkpoint is used when base_seed is provided."""
        # Setup mocks
        mock_discovery = MagicMock()
        mock_discovery_class.return_value = mock_discovery

        mock_checkpoint = Checkpoint(
            path=Path("/path/to/checkpoint.ckpt"),
            model_name="tide",
            timestamp="2024-11-20",
            seed=100,
            checkpoint_type="best_val_loss",
            val_loss=0.025,
        )
        mock_discovery.get_best_checkpoint.return_value = mock_checkpoint

        mock_finetune.return_value = True

        # Call function with base_seed
        run_finetuning(experiment_config_file, models=["tide"], n_runs=1, base_seed=100)

        # Verify specific seed was used
        mock_discovery.get_best_checkpoint.assert_called_once_with("tide", 100, stage="training")

    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.CheckpointDiscovery")
    def test_handles_missing_base_checkpoint(self, mock_discovery_class, experiment_config_file):
        """Test that missing base checkpoint is handled gracefully."""
        # Setup mocks
        mock_discovery = MagicMock()
        mock_discovery_class.return_value = mock_discovery
        mock_discovery.get_median_checkpoint.return_value = None

        # Call function
        results = run_finetuning(experiment_config_file, models=["tide"], n_runs=2, start_seed=42)

        # Should mark all seeds as failed
        assert results["models"]["tide"]["successful"] == 0
        assert results["models"]["tide"]["failed_seeds"] == [42, 43]

    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.CheckpointDiscovery")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.finetune_single_model")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.is_finetuning_complete")
    def test_skips_completed_runs_unless_fresh(
        self, mock_is_complete, mock_finetune, mock_discovery_class, experiment_config_file
    ):
        """Test that completed runs are skipped unless fresh flag is set."""
        # Setup mocks
        mock_discovery = MagicMock()
        mock_discovery_class.return_value = mock_discovery

        mock_checkpoint = Checkpoint(
            path=Path("/path/to/checkpoint.ckpt"),
            model_name="tide",
            timestamp="2024-11-20",
            seed=42,
            checkpoint_type="best_val_loss",
            val_loss=0.025,
        )
        mock_discovery.get_median_checkpoint.return_value = mock_checkpoint

        mock_is_complete.return_value = True
        mock_finetune.return_value = True

        # Call without fresh flag
        results = run_finetuning(experiment_config_file, models=["tide"], n_runs=1, fresh=False)

        # Should skip completed run
        mock_finetune.assert_not_called()
        assert results["models"]["tide"]["successful"] == 1

        # Reset mocks
        mock_finetune.reset_mock()

        # Call with fresh flag
        results = run_finetuning(experiment_config_file, models=["tide"], n_runs=1, fresh=True)

        # Should not skip
        mock_finetune.assert_called_once()

    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.CheckpointDiscovery")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.finetune_single_model")
    def test_tracks_results_correctly(self, mock_finetune, mock_discovery_class, experiment_config_file):
        """Test that results are tracked correctly."""
        # Setup mocks
        mock_discovery = MagicMock()
        mock_discovery_class.return_value = mock_discovery

        mock_checkpoint = Checkpoint(
            path=Path("/path/to/checkpoint.ckpt"),
            model_name="tide",
            timestamp="2024-11-20",
            seed=42,
            checkpoint_type="best_val_loss",
            val_loss=0.025,
        )
        mock_discovery.get_median_checkpoint.return_value = mock_checkpoint

        # First call succeeds, second fails
        mock_finetune.side_effect = [True, False, True]

        # Call function
        results = run_finetuning(experiment_config_file, models=["tide"], n_runs=3, start_seed=100)

        # Verify results
        assert results["models"]["tide"]["total"] == 3
        assert results["models"]["tide"]["successful"] == 2
        assert results["models"]["tide"]["failed_seeds"] == [101]

    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.CheckpointDiscovery")
    def test_calculates_total_time(self, mock_discovery_class, experiment_config_file):
        """Test that total time is calculated and formatted correctly."""
        mock_discovery = MagicMock()
        mock_discovery_class.return_value = mock_discovery
        mock_discovery.get_median_checkpoint.return_value = None

        results = run_finetuning(experiment_config_file, models=[], n_runs=0)

        # Should have a formatted time string
        assert "total_time" in results
        assert isinstance(results["total_time"], str)
        # Format should be HH:MM:SS
        assert results["total_time"].count(":") == 2

    def test_raises_error_for_missing_models_section(self, tmp_path):
        """Test that error is raised when experiment config has no models section."""
        config = {"trainer": {"max_epochs": 100}}  # No models section

        config_path = tmp_path / "bad_experiment.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(ValueError, match="must have a 'models' section"):
            run_finetuning(config_path, n_runs=1)

    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.CheckpointDiscovery")
    @patch("transfer_learning_publication.finetuning_cli.finetune_orchestrator.is_finetuning_complete")
    def test_uses_config_max_epochs_when_not_provided(
        self, mock_is_complete, mock_discovery_class, experiment_config_file
    ):
        """Test that max_epochs from config is used when not provided as argument."""
        mock_discovery = MagicMock()
        mock_discovery_class.return_value = mock_discovery
        mock_discovery.get_median_checkpoint.return_value = None
        mock_is_complete.return_value = False  # Ensure we don't skip

        with patch(
            "transfer_learning_publication.finetuning_cli.finetune_orchestrator.finetune_single_model"
        ) as mock_finetune:
            mock_checkpoint = Checkpoint(
                path=Path("/path/to/checkpoint.ckpt"),
                model_name="tide",
                timestamp="2024-11-20",
                seed=42,
                checkpoint_type="best_val_loss",
                val_loss=0.025,
            )
            mock_discovery.get_median_checkpoint.return_value = mock_checkpoint
            mock_finetune.return_value = True

            run_finetuning(experiment_config_file, models=["tide"], n_runs=1, max_epochs=None)

            # Should use config max_epochs (100)
            mock_finetune.assert_called_once()
            call_args = mock_finetune.call_args
            assert call_args[1]["max_epochs"] == 100
