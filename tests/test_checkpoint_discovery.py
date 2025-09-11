"""
Comprehensive tests for the checkpoint discovery module.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from transfer_learning_publication.checkpoint_utils import Checkpoint, CheckpointDiscovery


class TestCheckpoint:
    """Tests for the Checkpoint dataclass."""

    def test_checkpoint_creation(self):
        """Test basic checkpoint creation."""
        ckpt = Checkpoint(
            path=Path("/path/to/checkpoint.ckpt"),
            model_name="tide",
            timestamp="2024-11-20",
            seed=42,
            checkpoint_type="best_val_loss",
            val_loss=0.0234,
            stage="training",
        )

        assert ckpt.model_name == "tide"
        assert ckpt.seed == 42
        assert ckpt.val_loss == 0.0234
        assert ckpt.checkpoint_type == "best_val_loss"
        assert ckpt.stage == "training"

    def test_checkpoint_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        ckpt = Checkpoint(
            path="/path/to/checkpoint.ckpt",
            model_name="tide",
            timestamp="2024-11-20",
            seed=42,
            checkpoint_type="last",
        )

        assert isinstance(ckpt.path, Path)
        assert str(ckpt.path) == "/path/to/checkpoint.ckpt"

    def test_checkpoint_validation(self):
        """Test validation of checkpoint_type and stage."""
        # Invalid checkpoint_type
        with pytest.raises(ValueError, match="Invalid checkpoint_type"):
            Checkpoint(
                path=Path("/path/to/checkpoint.ckpt"),
                model_name="tide",
                timestamp="2024-11-20",
                seed=42,
                checkpoint_type="invalid",
            )

        # Invalid stage
        with pytest.raises(ValueError, match="Invalid stage"):
            Checkpoint(
                path=Path("/path/to/checkpoint.ckpt"),
                model_name="tide",
                timestamp="2024-11-20",
                seed=42,
                checkpoint_type="best_val_loss",
                stage="invalid",
            )

    def test_checkpoint_comparison(self):
        """Test checkpoint comparison by validation loss."""
        ckpt1 = Checkpoint(
            path=Path("/path/1.ckpt"),
            model_name="tide",
            timestamp="2024-11-20",
            seed=42,
            checkpoint_type="best_val_loss",
            val_loss=0.025,
        )

        ckpt2 = Checkpoint(
            path=Path("/path/2.ckpt"),
            model_name="tide",
            timestamp="2024-11-20",
            seed=43,
            checkpoint_type="best_val_loss",
            val_loss=0.020,
        )

        ckpt3 = Checkpoint(
            path=Path("/path/3.ckpt"),
            model_name="tide",
            timestamp="2024-11-20",
            seed=44,
            checkpoint_type="last",
            val_loss=None,
        )

        # Lower val_loss is better
        assert ckpt2 < ckpt1
        assert ckpt1 > ckpt2

        # Checkpoint with val_loss is better than without
        assert ckpt1 < ckpt3
        assert ckpt2 < ckpt3

        # Sorting should work
        checkpoints = [ckpt1, ckpt3, ckpt2]
        checkpoints.sort()
        assert checkpoints == [ckpt2, ckpt1, ckpt3]

    def test_checkpoint_tie_breaking(self):
        """Test tie-breaking by seed when val_loss is equal."""
        ckpt1 = Checkpoint(
            path=Path("/path/1.ckpt"),
            model_name="tide",
            timestamp="2024-11-20",
            seed=43,
            checkpoint_type="best_val_loss",
            val_loss=0.020,
        )

        ckpt2 = Checkpoint(
            path=Path("/path/2.ckpt"),
            model_name="tide",
            timestamp="2024-11-20",
            seed=42,
            checkpoint_type="best_val_loss",
            val_loss=0.020,
        )

        # Same loss, lower seed wins
        assert ckpt2 < ckpt1

    def test_checkpoint_string_representation(self):
        """Test string representation for logging."""
        ckpt = Checkpoint(
            path=Path("/path/to/checkpoint.ckpt"),
            model_name="tide",
            timestamp="2024-11-20",
            seed=42,
            checkpoint_type="best_val_loss",
            val_loss=0.0234,
            rank=2,
            total_runs=5,
        )

        str_repr = str(ckpt)
        assert "tide" in str_repr
        assert "seed=42" in str_repr
        assert "0.0234" in str_repr
        assert "rank=2/5" in str_repr


class TestCheckpointDiscovery:
    """Tests for the CheckpointDiscovery class."""

    def create_test_directory_structure(self, tmp_path: Path) -> Path:
        """Create a test checkpoint directory structure."""
        # Create training checkpoints
        base = tmp_path / "checkpoints"

        # TiDE model - multiple seeds and timestamps
        tide_dir = base / "training" / "model_name=tide"

        # Seed 42, timestamp 2024-11-20
        run1 = tide_dir / "run_2024-11-20_seed42" / "checkpoints"
        run1.mkdir(parents=True)
        (run1 / "best_val_loss_0.0234.ckpt").touch()
        (run1 / "last.ckpt").touch()
        (run1.parent / "metrics.csv").touch()

        # Seed 43, timestamp 2024-11-20
        run2 = tide_dir / "run_2024-11-20_seed43" / "checkpoints"
        run2.mkdir(parents=True)
        (run2 / "best_val_loss_0.0245.ckpt").touch()
        (run2 / "last.ckpt").touch()

        # Seed 44, timestamp 2024-11-20
        run3 = tide_dir / "run_2024-11-20_seed44" / "checkpoints"
        run3.mkdir(parents=True)
        (run3 / "best_val_loss_0.0228.ckpt").touch()
        (run3 / "last.ckpt").touch()

        # Seed 42, newer timestamp 2024-11-21
        run4 = tide_dir / "run_2024-11-21_seed42" / "checkpoints"
        run4.mkdir(parents=True)
        (run4 / "best_val_loss_0.0220.ckpt").touch()
        (run4 / "last.ckpt").touch()

        # EALSTM model - one seed
        ealstm_dir = base / "training" / "model_name=ealstm"
        run5 = ealstm_dir / "run_2024-11-20_seed42" / "checkpoints"
        run5.mkdir(parents=True)
        (run5 / "best_val_loss_0.0312.ckpt").touch()
        (run5 / "last.ckpt").touch()

        # Create versioned checkpoints to test version filtering
        (run1 / "best_val_loss_0.0234-v1.ckpt").touch()
        (run1 / "best_val_loss_0.0234-v2.ckpt").touch()
        (run1 / "last-v1.ckpt").touch()

        # Create finetuning directory (empty for now)
        (base / "finetuning").mkdir(parents=True)

        return base

    def test_get_checkpoint_basic(self, tmp_path):
        """Test basic checkpoint retrieval."""
        base = self.create_test_directory_structure(tmp_path)
        discovery = CheckpointDiscovery(base)

        # Get specific checkpoint
        ckpt = discovery.get_checkpoint("tide", 42, timestamp="2024-11-20")
        assert ckpt is not None
        assert ckpt.model_name == "tide"
        assert ckpt.seed == 42
        assert ckpt.timestamp == "2024-11-20"
        assert ckpt.val_loss == 0.0234
        assert ckpt.checkpoint_type == "best_val_loss"

        # Get last checkpoint
        ckpt = discovery.get_checkpoint("tide", 42, checkpoint_type="last", timestamp="2024-11-20")
        assert ckpt is not None
        assert ckpt.checkpoint_type == "last"
        assert ckpt.val_loss is None

    def test_get_checkpoint_latest_timestamp(self, tmp_path):
        """Test getting checkpoint with most recent timestamp."""
        base = self.create_test_directory_structure(tmp_path)
        discovery = CheckpointDiscovery(base)

        # Should get the 2024-11-21 checkpoint (most recent)
        ckpt = discovery.get_checkpoint("tide", 42)
        assert ckpt is not None
        assert ckpt.timestamp == "2024-11-21"
        assert ckpt.val_loss == 0.0220

    def test_get_checkpoint_version_filtering(self, tmp_path):
        """Test that versioned checkpoints are filtered correctly."""
        base = self.create_test_directory_structure(tmp_path)
        discovery = CheckpointDiscovery(base)

        # Should get base version, not -v1 or -v2
        ckpt = discovery.get_checkpoint("tide", 42, timestamp="2024-11-20")
        assert ckpt is not None
        assert "-v" not in str(ckpt.path)
        assert ckpt.path.name == "best_val_loss_0.0234.ckpt"

    def test_get_best_checkpoint(self, tmp_path):
        """Test convenience method for getting best checkpoint."""
        base = self.create_test_directory_structure(tmp_path)
        discovery = CheckpointDiscovery(base)

        ckpt = discovery.get_best_checkpoint("tide", 42)
        assert ckpt is not None
        assert ckpt.checkpoint_type == "best_val_loss"
        assert ckpt.timestamp == "2024-11-21"  # Most recent

    def test_list_runs(self, tmp_path):
        """Test listing all available runs."""
        base = self.create_test_directory_structure(tmp_path)
        discovery = CheckpointDiscovery(base)

        # List all runs
        runs = discovery.list_runs()
        assert len(runs) == 5  # 4 tide runs + 1 ealstm run

        # List tide runs only
        tide_runs = discovery.list_runs(model_name="tide")
        assert len(tide_runs) == 4
        assert all(r["model_name"] == "tide" for r in tide_runs)

        # List specific seed
        seed42_runs = discovery.list_runs(seed=42)
        assert len(seed42_runs) == 3  # 2 tide + 1 ealstm
        assert all(r["seed"] == 42 for r in seed42_runs)

        # Check run details
        run = next(r for r in tide_runs if r["seed"] == 44)
        assert run["timestamp"] == "2024-11-20"
        assert run["has_best_checkpoint"] is True
        assert run["has_last_checkpoint"] is True
        assert run["best_val_loss"] == 0.0228

    def test_exists(self, tmp_path):
        """Test checking if checkpoint exists."""
        base = self.create_test_directory_structure(tmp_path)
        discovery = CheckpointDiscovery(base)

        # Existing checkpoint
        assert discovery.exists("tide", 42, "2024-11-20") is True
        assert discovery.exists("tide", 42, "2024-11-20", checkpoint_type="last") is True

        # Non-existing checkpoint
        assert discovery.exists("tide", 99, "2024-11-20") is False
        assert discovery.exists("nonexistent", 42, "2024-11-20") is False

    def test_get_median_checkpoint(self, tmp_path):
        """Test getting median performing checkpoint."""
        base = self.create_test_directory_structure(tmp_path)
        discovery = CheckpointDiscovery(base)

        # Get median across seeds 42, 43, 44 for timestamp 2024-11-20
        median = discovery.get_median_checkpoint("tide", seeds=[42, 43, 44], timestamp="2024-11-20")
        assert median is not None
        assert median.seed == 42  # 0.0234 is the median value
        assert median.val_loss == 0.0234
        assert median.rank == 2
        assert median.total_runs == 3

    def test_get_checkpoint_rankings(self, tmp_path):
        """Test getting ranked checkpoints."""
        base = self.create_test_directory_structure(tmp_path)
        discovery = CheckpointDiscovery(base)

        # Get rankings for specific timestamp
        rankings = discovery.get_checkpoint_rankings("tide", timestamp="2024-11-20")
        assert len(rankings) == 3

        # Check ordering (best to worst)
        assert rankings[0].val_loss == 0.0228  # Best
        assert rankings[0].rank == 1
        assert rankings[0].seed == 44

        assert rankings[1].val_loss == 0.0234  # Median
        assert rankings[1].rank == 2
        assert rankings[1].seed == 42

        assert rankings[2].val_loss == 0.0245  # Worst
        assert rankings[2].rank == 3
        assert rankings[2].seed == 43

        # Check percentiles
        assert rankings[0].percentile == 0.0
        assert rankings[1].percentile == 50.0
        assert rankings[2].percentile == 100.0

    def test_get_percentile_checkpoint(self, tmp_path):
        """Test getting checkpoint at specific percentile."""
        base = self.create_test_directory_structure(tmp_path)
        discovery = CheckpointDiscovery(base)

        # Get checkpoints at different percentiles
        best = discovery.get_percentile_checkpoint("tide", 0, timestamp="2024-11-20")
        median = discovery.get_percentile_checkpoint("tide", 50, timestamp="2024-11-20")
        worst = discovery.get_percentile_checkpoint("tide", 100, timestamp="2024-11-20")

        assert best.seed == 44
        assert median.seed == 42
        assert worst.seed == 43

    def test_get_run_statistics(self, tmp_path):
        """Test computing run statistics."""
        base = self.create_test_directory_structure(tmp_path)
        discovery = CheckpointDiscovery(base)

        stats = discovery.get_run_statistics("tide", timestamp="2024-11-20")

        assert stats["count"] == 3
        assert stats["min"] == 0.0228
        assert stats["max"] == 0.0245
        assert stats["median"] == 0.0234
        assert stats["best_seed"] == 44
        assert stats["median_seed"] == 42
        assert stats["worst_seed"] == 43

        # Check mean calculation
        expected_mean = (0.0228 + 0.0234 + 0.0245) / 3
        assert abs(stats["mean"] - expected_mean) < 0.0001

    def test_get_parent_checkpoint(self, tmp_path):
        """Test getting parent checkpoint for fine-tuning."""
        base = self.create_test_directory_structure(tmp_path)
        discovery = CheckpointDiscovery(base)

        parent = discovery.get_parent_checkpoint("tide", 42)
        assert parent is not None
        assert parent.stage == "training"
        assert parent.checkpoint_type == "best_val_loss"
        assert parent.timestamp == "2024-11-21"  # Most recent

    def test_get_checkpoints_for_model(self, tmp_path):
        """Test getting all checkpoints for a model."""
        base = self.create_test_directory_structure(tmp_path)
        discovery = CheckpointDiscovery(base)

        checkpoints = discovery.get_checkpoints_for_model("tide")

        assert 42 in checkpoints
        assert 43 in checkpoints
        assert 44 in checkpoints

        # Seed 42 has 2 timestamps, each with best and last
        assert len(checkpoints[42]) == 4

        # Seeds 43 and 44 have 1 timestamp each, with best and last
        assert len(checkpoints[43]) == 2
        assert len(checkpoints[44]) == 2

    def test_get_latest_timestamp(self, tmp_path):
        """Test finding most recent timestamp."""
        base = self.create_test_directory_structure(tmp_path)
        discovery = CheckpointDiscovery(base)

        latest = discovery.get_latest_timestamp("tide", 42)
        assert latest == "2024-11-21"

        latest = discovery.get_latest_timestamp("tide", 43)
        assert latest == "2024-11-20"

        latest = discovery.get_latest_timestamp("nonexistent", 42)
        assert latest is None

    def test_empty_directory(self, tmp_path):
        """Test behavior with empty checkpoint directory."""
        base = tmp_path / "checkpoints"
        base.mkdir()
        discovery = CheckpointDiscovery(base)

        assert discovery.get_checkpoint("tide", 42) is None
        assert discovery.list_runs() == []
        assert discovery.get_median_checkpoint("tide") is None

        stats = discovery.get_run_statistics("tide")
        assert stats["count"] == 0
        assert stats["mean"] is None

    def test_validate_checkpoint(self, tmp_path):
        """Test checkpoint validation."""
        base = self.create_test_directory_structure(tmp_path)
        discovery = CheckpointDiscovery(base)

        # Valid checkpoint
        ckpt = discovery.get_checkpoint("tide", 42, timestamp="2024-11-20")
        
        # Mock file size check
        with patch.object(Checkpoint, "get_file_size_mb", return_value=10.0):
            assert discovery.validate_checkpoint(ckpt) is True

        # Invalid checkpoint (doesn't exist)
        fake_ckpt = Checkpoint(
            path=Path("/nonexistent/checkpoint.ckpt"),
            model_name="tide",
            timestamp="2024-11-20",
            seed=42,
            checkpoint_type="best_val_loss",
        )
        assert discovery.validate_checkpoint(fake_ckpt) is False

        # Too small checkpoint
        with patch.object(Checkpoint, "exists", return_value=True):
            with patch.object(Checkpoint, "get_file_size_mb", return_value=0.0005):
                assert discovery.validate_checkpoint(ckpt) is False


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_paper_reporting_workflow(self, tmp_path):
        """Test complete workflow for paper reporting."""
        # Create test structure
        base = tmp_path / "checkpoints"
        discovery = CheckpointDiscovery(base)

        # Create checkpoints for multiple models and seeds
        models_data = {
            "tide": [0.0234, 0.0245, 0.0228, 0.0239, 0.0241],
            "ealstm": [0.0312, 0.0318, 0.0309, 0.0315, 0.0320],
            "tsmixer": [0.0198, 0.0205, 0.0195, 0.0201, 0.0199],
        }

        for model_name, losses in models_data.items():
            model_dir = base / "training" / f"model_name={model_name}"
            for i, loss in enumerate(losses):
                seed = 42 + i
                run_dir = model_dir / f"run_2024-11-20_seed{seed}" / "checkpoints"
                run_dir.mkdir(parents=True)
                (run_dir / f"best_val_loss_{loss:.4f}.ckpt").touch()

        # Get median checkpoint for each model
        results = {}
        for model_name in models_data.keys():
            median = discovery.get_median_checkpoint(model_name)
            stats = discovery.get_run_statistics(model_name)

            results[model_name] = {
                "median_loss": stats["median"],
                "mean_loss": stats["mean"],
                "std_loss": stats["std"],
                "median_seed": median.seed if median else None,
            }

        # Verify results
        # The median (3rd out of 5 sorted values) should be:
        assert results["tide"]["median_seed"] == 45  # seed with 0.0239
        assert results["ealstm"]["median_seed"] == 45  # seed with 0.0315
        assert results["tsmixer"]["median_seed"] == 46  # seed with 0.0199

    def test_fine_tuning_workflow(self, tmp_path):
        """Test fine-tuning parent/child relationship."""
        base = tmp_path / "checkpoints"
        discovery = CheckpointDiscovery(base)

        # Create training checkpoint
        train_dir = base / "training" / "model_name=tide" / "run_2024-11-20_seed42" / "checkpoints"
        train_dir.mkdir(parents=True)
        (train_dir / "best_val_loss_0.0234.ckpt").touch()

        # Create fine-tuned checkpoints
        ft_dir = base / "finetuning" / "model_name=tide"
        for i, loss in enumerate([0.0210, 0.0208, 0.0212]):
            run_dir = ft_dir / f"run_2024-11-2{i+1}_seed42" / "checkpoints"
            run_dir.mkdir(parents=True)
            (run_dir / f"best_val_loss_{loss:.4f}.ckpt").touch()

        # Get parent checkpoint
        parent = discovery.get_parent_checkpoint("tide", 42)
        assert parent is not None
        assert parent.val_loss == 0.0234

        # Find fine-tuned versions
        finetuned = discovery.find_finetuned_checkpoints(parent)
        assert len(finetuned) == 3

        # Check they're sorted by performance
        assert finetuned[0].val_loss == 0.0208
        assert finetuned[1].val_loss == 0.0210
        assert finetuned[2].val_loss == 0.0212