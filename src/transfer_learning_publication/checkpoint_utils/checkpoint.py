from dataclasses import dataclass
from pathlib import Path


@dataclass
class Checkpoint:
    """
    Represents a single checkpoint with its metadata.

    Attributes:
        path: Full path to the checkpoint file
        model_name: Name of the model (e.g., "tide", "ealstm")
        timestamp: Training run timestamp (YYYY-MM-DD format)
        seed: Random seed used for training
        checkpoint_type: Either "best_val_loss" or "last"
        val_loss: Validation loss extracted from filename (None for "last" checkpoints)
        stage: Either "training" or "finetuning"
        rank: Ranking position when compared with other seeds (1 = best)
        percentile: Percentile position (0-100, where 0 = best, 100 = worst)
        total_runs: Total number of runs being compared
        version: Version number if checkpoint has multiple versions (e.g., -v1, -v2)
    """

    path: Path
    model_name: str
    timestamp: str
    seed: int
    checkpoint_type: str
    val_loss: float | None = None
    stage: str = "training"
    rank: int | None = None
    percentile: float | None = None
    total_runs: int | None = None
    version: int | None = None

    def __post_init__(self):
        """Validate and convert path to Path object if needed."""
        if isinstance(self.path, str):
            self.path = Path(self.path)

        # Validate checkpoint_type
        if self.checkpoint_type not in ["best_val_loss", "last"]:
            raise ValueError(f"Invalid checkpoint_type: {self.checkpoint_type}. Must be 'best_val_loss' or 'last'")

        # Validate stage
        if self.stage not in ["training", "finetuning"]:
            raise ValueError(f"Invalid stage: {self.stage}. Must be 'training' or 'finetuning'")

    def exists(self) -> bool:
        """Check if the checkpoint file exists on disk."""
        return self.path.exists() and self.path.is_file()

    def get_file_size_mb(self) -> float:
        """Get the file size in megabytes."""
        if not self.exists():
            return 0.0
        return self.path.stat().st_size / (1024 * 1024)

    def __lt__(self, other: "Checkpoint") -> bool:
        """
        Compare checkpoints by validation loss for sorting.
        Checkpoints without val_loss are considered worse than those with.
        """
        if not isinstance(other, Checkpoint):
            return NotImplemented

        # Both have val_loss - lower is better
        if self.val_loss is not None and other.val_loss is not None:
            if self.val_loss != other.val_loss:
                return self.val_loss < other.val_loss
            # Tie-break by seed (lower seed wins)
            return self.seed < other.seed

        # Only self has val_loss - self is better
        if self.val_loss is not None:
            return True

        # Only other has val_loss - other is better
        if other.val_loss is not None:
            return False

        # Neither has val_loss - tie-break by seed
        return self.seed < other.seed

    def __eq__(self, other: object) -> bool:
        """Check equality based on path."""
        if not isinstance(other, Checkpoint):
            return NotImplemented
        return self.path == other.path

    def __hash__(self) -> int:
        """Hash based on path for use in sets/dicts."""
        return hash(self.path)

    def __str__(self) -> str:
        """String representation for logging."""
        loss_str = f", loss={self.val_loss:.4f}" if self.val_loss is not None else ""
        rank_str = f", rank={self.rank}/{self.total_runs}" if self.rank is not None else ""
        return f"Checkpoint({self.model_name}, seed={self.seed}, {self.checkpoint_type}{loss_str}{rank_str})"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"Checkpoint(path={self.path}, model_name='{self.model_name}', "
            f"timestamp='{self.timestamp}', seed={self.seed}, "
            f"checkpoint_type='{self.checkpoint_type}', val_loss={self.val_loss}, "
            f"stage='{self.stage}', rank={self.rank}, percentile={self.percentile}, "
            f"total_runs={self.total_runs}, version={self.version})"
        )
