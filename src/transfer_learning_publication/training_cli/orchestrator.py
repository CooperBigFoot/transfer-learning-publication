"""
Orchestrates the training of multiple models with multiple seeds.
Manages checkpointing, resumption, and progress tracking.
"""

import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

import yaml

from .trainer_factory import train_single_model

logger = logging.getLogger(__name__)


def run_experiment(
    experiment_path: Path, models: list[str] | None = None, n_runs: int = 1, start_seed: int = 42, fresh: bool = False
) -> dict:
    """
    Main orchestration function that:
    1. Loads experiment.yaml
    2. Determines which models to train (all or specified subset)
    3. Gets experiment timestamp (YYYY-MM-DD format)
    4. For each model and seed combination:
       - Checks if already completed (unless --fresh)
       - Creates checkpoint directories
       - Instantiates model and datamodule
       - Configures trainer with callbacks
       - Runs training
       - Handles failures gracefully
    5. Returns summary of results

    Args:
        experiment_path: Path to experiment configuration YAML file
        models: Optional list of specific models to train (None = all models)
        n_runs: Number of runs per model (different seeds)
        start_seed: Starting seed value
        fresh: If True, ignore existing checkpoints and restart

    Returns:
        Dictionary with experiment results including success/failure counts
    """
    start_time = time.time()

    # Load experiment configuration
    with open(experiment_path) as f:
        experiment_config = yaml.safe_load(f)

    if "models" not in experiment_config:
        raise ValueError("Experiment configuration must have a 'models' section")

    # Get models to train
    all_models = experiment_config["models"]
    if models:
        # Filter to requested models
        models_to_train = {name: path for name, path in all_models.items() if name in models}
        # Check for invalid model names
        invalid = set(models) - set(all_models.keys())
        if invalid:
            raise ValueError(f"Invalid model names: {invalid}. Available: {list(all_models.keys())}")
    else:
        models_to_train = all_models

    # Get timestamp for this experiment run
    timestamp = datetime.now().strftime("%Y-%m-%d")

    # Prepare seeds
    seeds = list(range(start_seed, start_seed + n_runs))

    # Results tracking
    results = {"total_time": None, "timestamp": timestamp, "models": {}}

    logger.info("=" * 60)
    logger.info(f"Starting experiment with {len(models_to_train)} models, {n_runs} runs each")
    logger.info(f"Models: {', '.join(models_to_train.keys())}")
    logger.info(f"Seeds: {seeds}")
    logger.info("=" * 60)

    # Train each model
    for model_name, config_path in models_to_train.items():
        logger.info(f"\nTraining model: {model_name}")
        logger.info("-" * 40)

        model_results = {"total": n_runs, "successful": 0, "failed_seeds": []}

        # Train with each seed
        for seed in seeds:
            # Check if already completed (unless fresh start requested)
            checkpoint_dir = create_checkpoint_path(model_name, timestamp, seed)

            if not fresh and is_run_complete(checkpoint_dir):
                logger.info(f"Skipping {model_name} seed={seed} (already completed)")
                model_results["successful"] += 1
                continue

            # Train the model
            try:
                success = train_single_model(
                    config_path=Path(config_path),
                    model_name=model_name,
                    seed=seed,
                    checkpoint_dir=checkpoint_dir,
                    tensorboard_dir=Path("tensorboard"),
                    max_epochs=experiment_config["trainer"].get("max_epochs")
                    if "trainer" in experiment_config
                    else None,
                )

                if success:
                    model_results["successful"] += 1
                else:
                    model_results["failed_seeds"].append(seed)
                    logger.info(f"✗ Failed: {model_name} seed={seed}")

            except Exception as e:
                model_results["failed_seeds"].append(seed)
                logger.info(f"✗ Failed: {model_name} seed={seed}")
                logger.error(f"Error training {model_name} seed={seed}: {e}")
                warnings.warn(f"Training failed for {model_name} seed={seed}: {e}", stacklevel=2)

        results["models"][model_name] = model_results

    # Calculate total time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    results["total_time"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return results


def create_checkpoint_path(model_name: str, timestamp: str, seed: int, base_dir: str = "checkpoints/training") -> Path:
    """
    Constructs hive-partitioned checkpoint path:
    checkpoints/training/model_name=tide/run_2024-11-20_seed42/

    Args:
        model_name: Name of the model
        timestamp: Timestamp string (YYYY-MM-DD format)
        seed: Random seed value
        base_dir: Base directory for checkpoints

    Returns:
        Path object for checkpoint directory
    """
    checkpoint_path = Path(base_dir) / f"model_name={model_name}" / f"run_{timestamp}_seed{seed}"
    # Note: Directory is NOT created here - that's done by the trainer
    return checkpoint_path


def is_run_complete(checkpoint_dir: Path) -> bool:
    """
    Check if a training run is complete by looking for best checkpoint.

    A run is considered complete if a best_val_loss_*.ckpt file exists.

    Args:
        checkpoint_dir: Directory to check for checkpoints

    Returns:
        True if run is complete, False otherwise
    """
    if not checkpoint_dir.exists():
        return False

    checkpoints_dir = checkpoint_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return False

    # Look for best_val_loss checkpoint
    best_checkpoints = list(checkpoints_dir.glob("best_val_loss_*.ckpt"))
    return len(best_checkpoints) > 0


def check_existing_runs(
    model_name: str, timestamp: str, seeds: list[int], base_dir: str = "checkpoints/training"
) -> list[int]:
    """
    Scans checkpoint directory to find completed runs.
    Returns list of seeds that still need to be trained.

    Args:
        model_name: Name of the model
        timestamp: Timestamp string
        seeds: List of all seeds to check
        base_dir: Base directory for checkpoints

    Returns:
        List of seeds that need training
    """
    remaining_seeds = []

    for seed in seeds:
        checkpoint_dir = create_checkpoint_path(model_name, timestamp, seed, base_dir)
        if not is_run_complete(checkpoint_dir):
            remaining_seeds.append(seed)

    return remaining_seeds
