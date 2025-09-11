import logging
import re
from pathlib import Path
from statistics import mean, median, stdev

from .checkpoint import Checkpoint

logger = logging.getLogger(__name__)


class CheckpointDiscovery:
    """
    Primary interface for checkpoint discovery and querying.

    This class provides methods to:
    - Find specific checkpoints based on model, seed, and timestamp
    - Select median-performing models for fair paper reporting
    - Support fine-tuning workflows with parent/child checkpoint relationships
    - Compute statistics across multiple training runs
    """

    def __init__(self, base_dir: Path | str = Path("checkpoints")):
        """
        Initialize checkpoint discovery with base directory.

        Args:
            base_dir: Base directory containing checkpoints (default: "checkpoints")
        """
        self.base_dir = Path(base_dir) if isinstance(base_dir, str) else base_dir

    # ================== Core Query Methods ==================

    def get_checkpoint(
        self,
        model_name: str,
        seed: int,
        stage: str = "training",
        checkpoint_type: str = "best_val_loss",
        timestamp: str | None = None,
    ) -> Checkpoint | None:
        """
        Retrieve a specific checkpoint based on provided criteria.

        Args:
            model_name: Model identifier (e.g., "tide", "ealstm")
            seed: Random seed used for training
            stage: Either "training" or "finetuning"
            checkpoint_type: Either "best_val_loss" or "last"
            timestamp: Specific timestamp (uses most recent if None)

        Returns:
            Checkpoint object if found, None otherwise

        Example:
            >>> # Get most recent best checkpoint for tide, seed 42
            >>> ckpt = discovery.get_checkpoint("tide", 42)
            >>>
            >>> # Get last checkpoint from specific timestamp
            >>> ckpt = discovery.get_checkpoint("tide", 42,
            ...                                 timestamp="2024-11-20",
            ...                                 checkpoint_type="last")
        """
        # If no timestamp specified, find the most recent one
        if timestamp is None:
            timestamp = self.get_latest_timestamp(model_name, seed, stage)
            if timestamp is None:
                return None

        # Construct checkpoint directory path
        stage_dir = self.base_dir / stage
        model_dir = stage_dir / f"model_name={model_name}"
        run_dir = model_dir / f"run_{timestamp}_seed{seed}"
        checkpoints_dir = run_dir / "checkpoints"

        if not checkpoints_dir.exists():
            return None

        # Find checkpoint file based on type
        if checkpoint_type == "best_val_loss":
            # Look for best_val_loss_*.ckpt files (ignoring versions)
            pattern = "best_val_loss_*.ckpt"
            checkpoints = list(checkpoints_dir.glob(pattern))

            # Filter out versioned checkpoints if base version exists
            checkpoints = self._filter_checkpoint_versions(checkpoints)

            if not checkpoints:
                return None

            # Use the first (should be only one after filtering)
            checkpoint_path = checkpoints[0]

            # Extract validation loss from filename
            val_loss = self._extract_val_loss(checkpoint_path.name)

        elif checkpoint_type == "last":
            # Look for last.ckpt (ignoring versions)
            checkpoint_path = checkpoints_dir / "last.ckpt"
            if not checkpoint_path.exists():
                return None
            val_loss = None
        else:
            raise ValueError(f"Invalid checkpoint_type: {checkpoint_type}")

        # Extract version if present
        version = self._extract_version(checkpoint_path.name)

        return Checkpoint(
            path=checkpoint_path,
            model_name=model_name,
            timestamp=timestamp,
            seed=seed,
            checkpoint_type=checkpoint_type,
            val_loss=val_loss,
            stage=stage,
            version=version,
        )

    def get_best_checkpoint(self, model_name: str, seed: int, stage: str = "training") -> Checkpoint | None:
        """
        Convenience method to get the most recent best checkpoint.

        Args:
            model_name: Model identifier
            seed: Random seed
            stage: Stage to search in

        Returns:
            Best checkpoint if found, None otherwise

        Example:
            >>> best = discovery.get_best_checkpoint("tide", 42)
            >>> if best:
            >>>     model = ModelFactory.create_from_checkpoint("tide", best.path)
        """
        return self.get_checkpoint(model_name, seed, stage, "best_val_loss")

    def list_runs(self, model_name: str | None = None, seed: int | None = None, stage: str = "training") -> list[dict]:
        """
        List all available training runs with metadata.

        Args:
            model_name: Filter by model (None for all)
            seed: Filter by seed (None for all)
            stage: Stage to search in

        Returns:
            List of dictionaries with run information:
            - model_name: Name of the model
            - seed: Random seed used
            - timestamp: Run timestamp
            - has_best_checkpoint: Whether best checkpoint exists
            - has_last_checkpoint: Whether last checkpoint exists
            - best_val_loss: Validation loss if available

        Example:
            >>> # List all tide runs
            >>> runs = discovery.list_runs(model_name="tide")
            >>> for run in runs:
            >>>     print(f"{run['timestamp']}_seed{run['seed']}: "
            ...           f"loss={run['best_val_loss']:.4f}")
        """
        runs = []
        stage_dir = self.base_dir / stage

        if not stage_dir.exists():
            return runs

        # Get all model directories or specific one
        if model_name:
            model_dirs = [stage_dir / f"model_name={model_name}"]
            if not model_dirs[0].exists():
                return runs
        else:
            model_dirs = [d for d in stage_dir.iterdir() if d.is_dir() and d.name.startswith("model_name=")]

        for model_dir in model_dirs:
            # Extract model name from directory
            current_model = model_dir.name.replace("model_name=", "")

            # Skip if not the requested model
            if model_name and current_model != model_name:
                continue

            # Get all run directories
            run_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]

            for run_dir in run_dirs:
                # Parse run directory name
                match = re.match(r"run_(\d{4}-\d{2}-\d{2})_seed(\d+)", run_dir.name)
                if not match:
                    continue

                run_timestamp = match.group(1)
                run_seed = int(match.group(2))

                # Skip if not the requested seed
                if seed is not None and run_seed != seed:
                    continue

                # Check what checkpoints exist
                checkpoints_dir = run_dir / "checkpoints"
                has_best = False
                has_last = False
                best_val_loss = None

                if checkpoints_dir.exists():
                    # Check for best checkpoint
                    best_files = list(checkpoints_dir.glob("best_val_loss_*.ckpt"))
                    best_files = self._filter_checkpoint_versions(best_files)
                    if best_files:
                        has_best = True
                        best_val_loss = self._extract_val_loss(best_files[0].name)

                    # Check for last checkpoint
                    has_last = (checkpoints_dir / "last.ckpt").exists()

                runs.append(
                    {
                        "model_name": current_model,
                        "seed": run_seed,
                        "timestamp": run_timestamp,
                        "has_best_checkpoint": has_best,
                        "has_last_checkpoint": has_last,
                        "best_val_loss": best_val_loss,
                    }
                )

        # Sort by model name, then seed, then timestamp
        runs.sort(key=lambda x: (x["model_name"], x["seed"], x["timestamp"]))

        return runs

    def exists(
        self,
        model_name: str,
        seed: int,
        timestamp: str,
        stage: str = "training",
        checkpoint_type: str = "best_val_loss",
    ) -> bool:
        """
        Check if a specific checkpoint exists.

        Args:
            model_name: Model identifier
            seed: Random seed
            timestamp: Run timestamp
            stage: Stage to check
            checkpoint_type: Type of checkpoint

        Returns:
            True if checkpoint exists, False otherwise

        Example:
            >>> if not discovery.exists("tide", 42, "2024-11-20"):
            >>>     print("Checkpoint not found, training needed")
        """
        ckpt = self.get_checkpoint(model_name, seed, stage, checkpoint_type, timestamp)
        return ckpt is not None and ckpt.exists()

    def validate_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """
        Verify a checkpoint file exists and is potentially loadable.

        Args:
            checkpoint: Checkpoint to validate

        Returns:
            True if checkpoint exists and has reasonable size, False otherwise

        Example:
            >>> ckpt = discovery.get_best_checkpoint("tide", 42)
            >>> if not discovery.validate_checkpoint(ckpt):
            >>>     print("Warning: Checkpoint may be corrupted")
        """
        if not checkpoint.exists():
            return False

        # Check file size (checkpoint should be at least 1KB)
        size_mb = checkpoint.get_file_size_mb()
        if size_mb < 0.001:  # Less than 1KB
            logger.warning(f"Checkpoint file seems too small: {checkpoint.path} ({size_mb:.3f} MB)")
            return False

        return True

    # ================== Statistical Selection Methods ==================

    def get_median_checkpoint(
        self, model_name: str, seeds: list[int] | None = None, stage: str = "training", timestamp: str | None = None
    ) -> Checkpoint | None:
        """
        Get the median performing checkpoint across multiple seeds.

        Args:
            model_name: Model identifier
            seeds: Specific seeds to consider (None for all)
            stage: Stage to search in
            timestamp: Specific timestamp (uses most recent per seed if None)

        Returns:
            Median checkpoint with rank and percentile information, None if no checkpoints

        Example:
            >>> # For paper reporting - get median across all seeds
            >>> median = discovery.get_median_checkpoint("tide")
            >>> print(f"Using median model: seed {median.seed}, "
            ...       f"rank {median.rank}/{median.total_runs}")
            >>>
            >>> # Get median for specific seeds only
            >>> median = discovery.get_median_checkpoint("tide", seeds=[42, 43, 44])
        """
        rankings = self.get_checkpoint_rankings(model_name, seeds, stage, timestamp)

        if not rankings:
            return None

        # Find median checkpoint
        n = len(rankings)
        median_idx = (n - 1) // 2  # For odd n, this is the middle; for even, lower middle

        return rankings[median_idx]

    def get_checkpoint_rankings(
        self, model_name: str, seeds: list[int] | None = None, stage: str = "training", timestamp: str | None = None
    ) -> list[Checkpoint]:
        """
        Get all checkpoints ranked by performance.

        Args:
            model_name: Model identifier
            seeds: Seeds to include (None for all)
            stage: Stage to search in
            timestamp: Specific timestamp (uses most recent per seed if None)

        Returns:
            List of checkpoints sorted by val_loss (best first), with rank and percentile

        Example:
            >>> rankings = discovery.get_checkpoint_rankings("tide")
            >>> print("Performance distribution:")
            >>> for ckpt in rankings:
            >>>     print(f"  Rank {ckpt.rank}: Seed {ckpt.seed} = {ckpt.val_loss:.4f}")
        """
        # Get all runs for the model
        all_runs = self.list_runs(model_name=model_name, stage=stage)

        # Filter by seeds if specified
        if seeds is not None:
            all_runs = [r for r in all_runs if r["seed"] in seeds]

        # Filter by timestamp if specified
        if timestamp is not None:
            all_runs = [r for r in all_runs if r["timestamp"] == timestamp]
        else:
            # Keep only most recent run per seed
            latest_runs = {}
            for run in all_runs:
                seed = run["seed"]
                if seed not in latest_runs or run["timestamp"] > latest_runs[seed]["timestamp"]:
                    latest_runs[seed] = run
            all_runs = list(latest_runs.values())

        # Get checkpoint objects
        checkpoints = []
        for run in all_runs:
            ckpt = self.get_checkpoint(
                model_name=model_name,
                seed=run["seed"],
                stage=stage,
                checkpoint_type="best_val_loss",
                timestamp=run["timestamp"],
            )
            if ckpt and ckpt.val_loss is not None:
                checkpoints.append(ckpt)

        # Sort by validation loss (best first)
        checkpoints.sort()

        # Add ranking information
        total = len(checkpoints)
        for i, ckpt in enumerate(checkpoints):
            ckpt.rank = i + 1
            # Calculate percentile: 0% for best, 100% for worst
            # For 3 items: ranks 1,2,3 -> percentiles 0%, 50%, 100%
            ckpt.percentile = (i / (total - 1) * 100) if total > 1 else 50.0
            ckpt.total_runs = total

        return checkpoints

    def get_percentile_checkpoint(
        self,
        model_name: str,
        percentile: float,
        seeds: list[int] | None = None,
        stage: str = "training",
        timestamp: str | None = None,
    ) -> Checkpoint | None:
        """
        Get checkpoint at a specific percentile of performance.

        Args:
            model_name: Model identifier
            percentile: Percentile (0-100, where 0 = best, 50 = median, 100 = worst)
            seeds: Seeds to include (None for all)
            stage: Stage to search in
            timestamp: Specific timestamp

        Returns:
            Checkpoint at the specified percentile, None if no checkpoints

        Example:
            >>> # Get quartiles for robustness analysis
            >>> q1 = discovery.get_percentile_checkpoint("tide", 25)
            >>> median = discovery.get_percentile_checkpoint("tide", 50)
            >>> q3 = discovery.get_percentile_checkpoint("tide", 75)
        """
        if not 0 <= percentile <= 100:
            raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")

        rankings = self.get_checkpoint_rankings(model_name, seeds, stage, timestamp)

        if not rankings:
            return None

        # Calculate index for percentile
        n = len(rankings)
        if n == 1:
            return rankings[0]

        # Convert percentile to index (0-based)
        idx = int((percentile / 100) * (n - 1))
        idx = min(idx, n - 1)  # Ensure we don't go out of bounds

        return rankings[idx]

    def get_run_statistics(
        self, model_name: str, seeds: list[int] | None = None, stage: str = "training", timestamp: str | None = None
    ) -> dict:
        """
        Compute summary statistics across multiple seeds.

        Args:
            model_name: Model identifier
            seeds: Seeds to include (None for all)
            stage: Stage to search in
            timestamp: Specific timestamp

        Returns:
            Dictionary with statistics:
            - count: Number of runs
            - mean: Mean validation loss
            - std: Standard deviation
            - min: Minimum (best) loss
            - max: Maximum (worst) loss
            - median: Median loss
            - q1: First quartile (25th percentile)
            - q3: Third quartile (75th percentile)
            - best_seed: Seed with best performance
            - median_seed: Seed with median performance
            - worst_seed: Seed with worst performance

        Example:
            >>> stats = discovery.get_run_statistics("tide")
            >>> print(f"TiDE: {stats['mean']:.4f} ± {stats['std']:.4f}")
            >>> print(f"Best: {stats['min']:.4f} (seed {stats['best_seed']})")
        """
        rankings = self.get_checkpoint_rankings(model_name, seeds, stage, timestamp)

        if not rankings:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "median": None,
                "q1": None,
                "q3": None,
                "best_seed": None,
                "median_seed": None,
                "worst_seed": None,
            }

        losses = [ckpt.val_loss for ckpt in rankings]
        n = len(losses)

        # Get quartile checkpoints
        q1_ckpt = self.get_percentile_checkpoint(model_name, 25, seeds, stage, timestamp)
        median_ckpt = self.get_percentile_checkpoint(model_name, 50, seeds, stage, timestamp)
        q3_ckpt = self.get_percentile_checkpoint(model_name, 75, seeds, stage, timestamp)

        return {
            "count": n,
            "mean": mean(losses),
            "std": stdev(losses) if n > 1 else 0.0,
            "min": losses[0],  # Best (rankings are sorted)
            "max": losses[-1],  # Worst
            "median": median(losses),
            "q1": q1_ckpt.val_loss if q1_ckpt else None,
            "q3": q3_ckpt.val_loss if q3_ckpt else None,
            "best_seed": rankings[0].seed,
            "median_seed": median_ckpt.seed if median_ckpt else None,
            "worst_seed": rankings[-1].seed,
        }

    # ================== Fine-tuning Support Methods ==================

    def get_parent_checkpoint(self, model_name: str, seed: int, timestamp: str | None = None) -> Checkpoint | None:
        """
        Get the training checkpoint to use as parent for fine-tuning.

        Args:
            model_name: Model identifier
            seed: Random seed
            timestamp: Specific timestamp (uses most recent if None)

        Returns:
            Training checkpoint suitable for fine-tuning, None if not found

        Example:
            >>> # Find parent checkpoint for fine-tuning
            >>> parent = discovery.get_parent_checkpoint("tide", seed=42)
            >>> if not parent:
            >>>     raise ValueError("No training checkpoint found")
            >>> print(f"Fine-tuning from: {parent.path}")
        """
        return self.get_checkpoint(
            model_name=model_name, seed=seed, stage="training", checkpoint_type="best_val_loss", timestamp=timestamp
        )

    def find_finetuned_checkpoints(self, parent_checkpoint: Checkpoint) -> list[Checkpoint]:
        """
        Find all fine-tuned versions of a parent checkpoint.

        Args:
            parent_checkpoint: The parent training checkpoint

        Returns:
            List of fine-tuned checkpoints derived from the parent

        Example:
            >>> parent = discovery.get_parent_checkpoint("tide", 42)
            >>> finetuned = discovery.find_finetuned_checkpoints(parent)
            >>> for ft in finetuned:
            >>>     print(f"  {ft.timestamp}: {ft.val_loss:.4f}")
        """
        # Look for fine-tuned checkpoints with same model and seed
        finetuned = []

        # Get all fine-tuning runs for this model/seed
        runs = self.list_runs(model_name=parent_checkpoint.model_name, seed=parent_checkpoint.seed, stage="finetuning")

        for run in runs:
            ckpt = self.get_checkpoint(
                model_name=parent_checkpoint.model_name,
                seed=parent_checkpoint.seed,
                stage="finetuning",
                checkpoint_type="best_val_loss",
                timestamp=run["timestamp"],
            )
            if ckpt:
                finetuned.append(ckpt)

        # Sort by validation loss
        finetuned.sort()

        return finetuned

    # ================== Utility Methods ==================

    def get_checkpoints_for_model(self, model_name: str, stage: str = "training") -> dict[int, list[Checkpoint]]:
        """
        Get all checkpoints for a model, organized by seed.

        Args:
            model_name: Model identifier
            stage: Stage to search in

        Returns:
            Dictionary mapping seed to list of checkpoints

        Example:
            >>> all_checkpoints = discovery.get_checkpoints_for_model("tide")
            >>> for seed, checkpoints in all_checkpoints.items():
            >>>     print(f"Seed {seed}: {len(checkpoints)} checkpoints")
        """
        checkpoints_by_seed = {}

        # Get all runs for the model
        runs = self.list_runs(model_name=model_name, stage=stage)

        for run in runs:
            seed = run["seed"]

            if seed not in checkpoints_by_seed:
                checkpoints_by_seed[seed] = []

            # Try to get both best and last checkpoints
            for ckpt_type in ["best_val_loss", "last"]:
                ckpt = self.get_checkpoint(
                    model_name=model_name, seed=seed, stage=stage, checkpoint_type=ckpt_type, timestamp=run["timestamp"]
                )
                if ckpt:
                    checkpoints_by_seed[seed].append(ckpt)

        return checkpoints_by_seed

    def get_latest_timestamp(self, model_name: str, seed: int, stage: str = "training") -> str | None:
        """
        Find the most recent timestamp for a model/seed combination.

        Args:
            model_name: Model identifier
            seed: Random seed
            stage: Stage to search in

        Returns:
            Most recent timestamp string, None if no runs found

        Example:
            >>> latest = discovery.get_latest_timestamp("tide", 42)
            >>> print(f"Most recent run: {latest}")
        """
        runs = self.list_runs(model_name=model_name, seed=seed, stage=stage)

        if not runs:
            return None

        # Sort by timestamp (descending) and return the most recent
        runs.sort(key=lambda x: x["timestamp"], reverse=True)
        return runs[0]["timestamp"]

    # ================== Private Helper Methods ==================

    def _extract_val_loss(self, filename: str) -> float | None:
        """
        Extract validation loss from checkpoint filename.

        Args:
            filename: Checkpoint filename (e.g., "best_val_loss_0.0234.ckpt")

        Returns:
            Validation loss as float, None if not found
        """
        # Pattern for best_val_loss_X.XXXX.ckpt (with optional version suffix)
        match = re.match(r"best_val_loss_([\d.]+)(?:-v\d+)?\.ckpt", filename)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _extract_version(self, filename: str) -> int | None:
        """
        Extract version number from checkpoint filename.

        Args:
            filename: Checkpoint filename (e.g., "best_val_loss_0.0234-v2.ckpt")

        Returns:
            Version number if present, None otherwise
        """
        # Pattern for -vN suffix
        match = re.search(r"-v(\d+)\.ckpt", filename)
        if match:
            return int(match.group(1))
        return None

    def _filter_checkpoint_versions(self, checkpoint_paths: list[Path]) -> list[Path]:
        """
        Filter out versioned checkpoints if base version exists.

        When Lightning saves multiple versions (e.g., checkpoint.ckpt, checkpoint-v1.ckpt),
        we prefer the base version without suffix.

        Args:
            checkpoint_paths: List of checkpoint file paths

        Returns:
            Filtered list with only the preferred versions
        """
        if not checkpoint_paths:
            return checkpoint_paths

        # Group by base name (without version suffix)
        base_names = {}
        for path in checkpoint_paths:
            # Remove version suffix if present
            base_name = re.sub(r"-v\d+\.ckpt$", ".ckpt", path.name)

            if base_name not in base_names:
                base_names[base_name] = []
            base_names[base_name].append(path)

        # For each base name, prefer the one without version suffix
        filtered = []
        for base_name, paths in base_names.items():
            if len(paths) == 1:
                filtered.append(paths[0])
            else:
                # Find the one without version suffix
                base_path = None
                for path in paths:
                    if not re.search(r"-v\d+\.ckpt$", path.name):
                        base_path = path
                        break

                # If no base version, use the first versioned one
                filtered.append(base_path if base_path else paths[0])

        return filtered
