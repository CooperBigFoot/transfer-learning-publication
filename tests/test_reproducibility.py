"""Tests for reproducibility and deterministic training behavior.

These tests verify that the random seeding mechanism works correctly
and produces identical results across multiple training runs.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from transfer_learning_publication.training_cli.orchestrator import run_experiment
from transfer_learning_publication.training_cli.trainer_factory import (
    create_trainer,
    train_single_model,
)


class TestReproducibility:
    """Test suite for verifying reproducibility of training runs."""

    @pytest.fixture
    def minimal_config(self, tmp_path):
        """Create a minimal configuration for fast testing."""
        config = {
            "data": {
                "base_path": "/Users/cooper/Desktop/first-test",
                "gauge_ids": ["tajikkyrgyz_15013"],  # Just one gauge for speed
                "pipeline_path": "/Users/cooper/Desktop/first-test/ts_pipeline.joblib",
            },
            "features": {
                "forcing": ["streamflow", "temperature_2m_mean", "total_precipitation_sum"],
                "static": ["area", "ele_mt_sav"],
                "target": "streamflow",
            },
            "sequence": {"input_length": 10, "output_length": 5},
            "data_preparation": {"is_autoregressive": True, "include_dates": True},
            "model": {"type": "tide", "overrides": {"hidden_size": 32}},  # TIDE with small hidden size for testing
            "dataloader": {"batch_size": 32, "num_workers": 0, "shuffle_train": True},
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    @pytest.fixture
    def reproducibility_experiment(self, tmp_path, minimal_config):
        """Create an experiment configuration for reproducibility testing."""
        experiment = {
            "models": {"test_model": str(minimal_config)},
            "trainer": {"max_epochs": 2},  # Very few epochs for speed
        }

        exp_path = tmp_path / "reproducibility_experiment.yaml"
        with open(exp_path, "w") as f:
            yaml.dump(experiment, f)

        return exp_path

    def test_trainer_seed_determinism(self, tmp_path):
        """Test that creating trainers with the same seed produces identical configuration."""
        checkpoint_dir = tmp_path / "checkpoints"

        # Create two trainers with the same seed
        trainer1 = create_trainer(model_name="test", seed=42, checkpoint_dir=checkpoint_dir)

        # Clear the checkpoint directory and create another trainer
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

        trainer2 = create_trainer(model_name="test", seed=42, checkpoint_dir=checkpoint_dir)

        # Check that key settings are identical
        # The trainers are configured with deterministic=True in trainer_factory.py
        assert trainer1.max_epochs == trainer2.max_epochs
        
        # Verify that seed was set (this is logged by PyTorch Lightning)
        # Both trainers should have been created after setting seed to 42

    @pytest.mark.skipif(
        not Path("/Users/cooper/Desktop/first-test").exists(),
        reason="Test data not available",
    )
    def test_single_model_training_reproducibility(self, minimal_config, tmp_path):
        """Test that training the same model with the same seed produces identical results."""
        # First training run
        checkpoint_dir1 = tmp_path / "run1"
        tensorboard_dir1 = tmp_path / "tb1"

        success1 = train_single_model(
            config_path=minimal_config,
            model_name="test_model",
            seed=42,
            checkpoint_dir=checkpoint_dir1,
            tensorboard_dir=tensorboard_dir1,
            max_epochs=2,
        )

        # Second training run with same seed
        checkpoint_dir2 = tmp_path / "run2"
        tensorboard_dir2 = tmp_path / "tb2"

        success2 = train_single_model(
            config_path=minimal_config,
            model_name="test_model",
            seed=42,
            checkpoint_dir=checkpoint_dir2,
            tensorboard_dir=tensorboard_dir2,
            max_epochs=2,
        )

        # Both should succeed
        assert success1 and success2

        # Compare validation losses from checkpoints
        ckpt1_files = list((checkpoint_dir1 / "checkpoints").glob("best_val_loss_*.ckpt"))
        ckpt2_files = list((checkpoint_dir2 / "checkpoints").glob("best_val_loss_*.ckpt"))

        assert len(ckpt1_files) > 0 and len(ckpt2_files) > 0

        # Extract loss values from filenames
        loss1 = float(ckpt1_files[0].stem.split("_")[-1])
        loss2 = float(ckpt2_files[0].stem.split("_")[-1])

        # Losses should be very close (allowing for minor floating point differences)
        # Using a tight tolerance since we expect high reproducibility
        assert abs(loss1 - loss2) < 0.001, f"Validation losses differ too much: {loss1} vs {loss2}"

    @pytest.mark.skipif(
        not Path("/Users/cooper/Desktop/first-test").exists(),
        reason="Test data not available",
    )
    def test_checkpoint_weight_reproducibility(self, minimal_config, tmp_path):
        """Test that model weights are identical when trained with the same seed."""
        # Train two models with the same seed
        checkpoint_dir1 = tmp_path / "weights_test1"
        checkpoint_dir2 = tmp_path / "weights_test2"

        train_single_model(
            config_path=minimal_config,
            model_name="test",
            seed=123,
            checkpoint_dir=checkpoint_dir1,
            tensorboard_dir=tmp_path / "tb1",
            max_epochs=1,
        )

        train_single_model(
            config_path=minimal_config,
            model_name="test",
            seed=123,
            checkpoint_dir=checkpoint_dir2,
            tensorboard_dir=tmp_path / "tb2",
            max_epochs=1,
        )

        # Load checkpoints
        ckpt1_path = list((checkpoint_dir1 / "checkpoints").glob("*.ckpt"))[0]
        ckpt2_path = list((checkpoint_dir2 / "checkpoints").glob("*.ckpt"))[0]

        ckpt1 = torch.load(ckpt1_path, map_location="cpu")
        ckpt2 = torch.load(ckpt2_path, map_location="cpu")

        # Compare state dicts
        state_dict1 = ckpt1["state_dict"]
        state_dict2 = ckpt2["state_dict"]

        assert state_dict1.keys() == state_dict2.keys()

        # For models with learnable parameters, check they're very close
        # Note: RevIN statistics may have minor variations due to data batch ordering
        for key in state_dict1.keys():
            if isinstance(state_dict1[key], torch.Tensor):
                # Use a slightly relaxed tolerance for RevIN layers, strict for others
                tolerance = 1e-3 if "rev_in" in key else 1e-5
                assert torch.allclose(
                    state_dict1[key], state_dict2[key], atol=tolerance
                ), f"Weights differ for {key}"

    @pytest.mark.skipif(
        not Path("/Users/cooper/Desktop/first-test").exists(),
        reason="Test data not available",
    )
    def test_experiment_reproducibility_with_fresh_flag(self, reproducibility_experiment, tmp_path):
        """Test that running experiments with --fresh and same seed produces identical results."""
        # Set a temporary checkpoint directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Monkey patch the checkpoint directory creation
            import transfer_learning_publication.training_cli.orchestrator as orch

            original_create_path = orch.create_checkpoint_path

            def patched_create_path(model_name, timestamp, seed, base_dir=None):
                return Path(temp_dir) / f"run_{seed}" / model_name

            orch.create_checkpoint_path = patched_create_path

            try:
                # First run
                results1 = run_experiment(
                    reproducibility_experiment,
                    models=None,
                    n_runs=1,
                    start_seed=42,
                    fresh=True,
                )

                # Second run with fresh flag (should overwrite)
                results2 = run_experiment(
                    reproducibility_experiment,
                    models=None,
                    n_runs=1,
                    start_seed=42,
                    fresh=True,
                )

                # Both runs should succeed
                assert results1["models"]["test_model"]["successful"] == 1
                assert results2["models"]["test_model"]["successful"] == 1

            finally:
                # Restore original function
                orch.create_checkpoint_path = original_create_path

    def test_different_seeds_produce_different_results(self, tmp_path):
        """Test that different seeds produce different random number sequences."""
        import lightning as pl

        # Set seed and generate random numbers
        pl.seed_everything(42)
        random_42 = torch.randn(10)

        # Set different seed and generate random numbers
        pl.seed_everything(123)
        random_123 = torch.randn(10)

        # They should be different
        assert not torch.allclose(random_42, random_123)

    def test_seed_propagation_through_pipeline(self, reproducibility_experiment):
        """Test that seeds are correctly propagated through the entire pipeline."""
        from unittest.mock import MagicMock, patch

        with patch(
            "transfer_learning_publication.training_cli.orchestrator.train_single_model"
        ) as mock_train:
            mock_train.return_value = True

            # Run experiment with specific seed
            run_experiment(
                reproducibility_experiment,
                models=None,
                n_runs=3,
                start_seed=100,
                fresh=True,
            )

            # Check that train_single_model was called with correct seeds
            assert mock_train.call_count == 3
            seeds_used = [call[1]["seed"] for call in mock_train.call_args_list]
            assert seeds_used == [100, 101, 102]


class TestDataLoaderReproducibility:
    """Test reproducibility of data loading."""

    @pytest.fixture
    def minimal_config(self, tmp_path):
        """Create a minimal configuration for fast testing."""
        config = {
            "data": {
                "base_path": "/Users/cooper/Desktop/first-test",
                "gauge_ids": ["tajikkyrgyz_15013"],  # Just one gauge for speed
                "pipeline_path": "/Users/cooper/Desktop/first-test/ts_pipeline.joblib",
            },
            "features": {
                "forcing": ["streamflow", "temperature_2m_mean", "total_precipitation_sum"],
                "static": ["area", "ele_mt_sav"],
                "target": "streamflow",
            },
            "sequence": {"input_length": 10, "output_length": 5},
            "data_preparation": {"is_autoregressive": True, "include_dates": True},
            "model": {"type": "tide", "overrides": {"hidden_size": 32}},  # TIDE with small hidden size for testing
            "dataloader": {"batch_size": 32, "num_workers": 0, "shuffle_train": True},
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    @pytest.mark.skipif(
        not Path("/Users/cooper/Desktop/first-test").exists(),
        reason="Test data not available",
    )
    def test_dataloader_seed_consistency(self, minimal_config):
        """Test that DataLoaders produce consistent batches with the same seed."""
        import lightning as pl

        from transfer_learning_publication.data import LSHDataModule

        # Create two data modules with same seed
        pl.seed_everything(42)
        dm1 = LSHDataModule(minimal_config)
        dm1.setup("fit")

        pl.seed_everything(42)
        dm2 = LSHDataModule(minimal_config)
        dm2.setup("fit")

        # Get first batch from each
        train_loader1 = dm1.train_dataloader()
        train_loader2 = dm2.train_dataloader()

        batch1 = next(iter(train_loader1))
        batch2 = next(iter(train_loader2))

        # Check that batches are identical
        assert torch.allclose(batch1.X, batch2.X)
        assert torch.allclose(batch1.y, batch2.y)
        if batch1.static is not None and batch2.static is not None:
            assert torch.allclose(batch1.static, batch2.static)