"""
Orchestrates the fine-tuning of pre-trained models with reduced learning rates.
Manages checkpoint loading, learning rate modification, and progress tracking.
"""

import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

import lightning as pl
import yaml

from ..checkpoint_utils import CheckpointDiscovery
from ..data import LSHDataModule
from ..models import ModelFactory
from ..training_cli.trainer_factory import create_trainer

logger = logging.getLogger(__name__)


def run_finetuning(
    experiment_path: Path,
    models: list[str] | None = None,
    lr_reduction: float = 25.0,
    base_seed: int | None = None,
    n_runs: int = 1,
    start_seed: int = 42,
    max_epochs: int | None = None,
    fresh: bool = False,
) -> dict:
    """
    Main orchestration function for fine-tuning pre-trained models.

    This function:
    1. Loads experiment configuration
    2. Discovers base checkpoints to fine-tune from
    3. For each model and seed combination:
       - Loads the pre-trained checkpoint
       - Modifies the learning rate
       - Resumes training with same data
       - Saves to finetuning/ directory

    Args:
        experiment_path: Path to experiment configuration YAML file
        models: Optional list of specific models to fine-tune (None = all models)
        lr_reduction: Factor to reduce learning rate by (new_lr = old_lr / lr_reduction)
        base_seed: Optional specific seed checkpoint to use as base
        n_runs: Number of fine-tuning runs per model
        start_seed: Starting seed value for fine-tuning
        max_epochs: Maximum epochs for fine-tuning (None uses experiment config)
        fresh: If True, ignore existing fine-tuned checkpoints and restart

    Returns:
        Dictionary with fine-tuning results including success/failure counts
    """
    start_time = time.time()

    # Load experiment configuration
    with open(experiment_path) as f:
        experiment_config = yaml.safe_load(f)

    if "models" not in experiment_config:
        raise ValueError("Experiment configuration must have a 'models' section")

    # Get models to fine-tune
    all_models = experiment_config["models"]
    if models:
        # Filter to requested models
        models_to_finetune = {name: path for name, path in all_models.items() if name in models}
        # Check for invalid model names
        invalid = set(models) - set(all_models.keys())
        if invalid:
            raise ValueError(f"Invalid model names: {invalid}. Available: {list(all_models.keys())}")
    else:
        models_to_finetune = all_models

    # Get timestamp for this fine-tuning run
    timestamp = datetime.now().strftime("%Y-%m-%d")

    # Initialize checkpoint discovery
    discovery = CheckpointDiscovery()

    # Prepare seeds for fine-tuning
    seeds = list(range(start_seed, start_seed + n_runs))

    # Get max_epochs from config if not provided
    if max_epochs is None:
        max_epochs = experiment_config.get("trainer", {}).get("max_epochs", 100)

    # Results tracking
    results = {
        "total_time": None,
        "timestamp": timestamp,
        "lr_reduction": lr_reduction,
        "models": {},
    }

    logger.info("=" * 60)
    logger.info(f"Starting fine-tuning with {len(models_to_finetune)} models, {n_runs} runs each")
    logger.info(f"Models: {', '.join(models_to_finetune.keys())}")
    logger.info(f"LR reduction factor: {lr_reduction}")
    logger.info(f"Seeds: {seeds}")
    logger.info("=" * 60)

    # Fine-tune each model
    for model_name, config_path in models_to_finetune.items():
        logger.info(f"\nFine-tuning model: {model_name}")
        logger.info("-" * 40)

        model_results = {"total": n_runs, "successful": 0, "failed_seeds": []}

        # Find base checkpoint to fine-tune from
        if base_seed is not None:
            # Use specific seed checkpoint
            base_checkpoint = discovery.get_best_checkpoint(model_name, base_seed, stage="training")
            if base_checkpoint is None:
                logger.warning(f"No training checkpoint found for {model_name} seed={base_seed}, skipping")
                model_results["failed_seeds"] = seeds
                results["models"][model_name] = model_results
                continue
            logger.info(f"Using base checkpoint from seed {base_seed}: val_loss={base_checkpoint.val_loss:.4f}")
        else:
            # Use median checkpoint for fair comparison
            base_checkpoint = discovery.get_median_checkpoint(model_name, stage="training")
            if base_checkpoint is None:
                logger.warning(f"No training checkpoints found for {model_name}, skipping")
                model_results["failed_seeds"] = seeds
                results["models"][model_name] = model_results
                continue
            logger.info(
                f"Using median checkpoint (seed {base_checkpoint.seed}): val_loss={base_checkpoint.val_loss:.4f}"
            )

        # Fine-tune with each seed
        for seed in seeds:
            # Check if already completed (unless fresh start requested)
            checkpoint_dir = create_finetuning_checkpoint_path(model_name, timestamp, seed)

            if not fresh and is_finetuning_complete(checkpoint_dir):
                logger.info(f"Skipping {model_name} seed={seed} (already completed)")
                model_results["successful"] += 1
                continue

            # Fine-tune the model
            try:
                success = finetune_single_model(
                    base_checkpoint_path=base_checkpoint.path,
                    config_path=Path(config_path),
                    model_name=model_name,
                    seed=seed,
                    checkpoint_dir=checkpoint_dir,
                    lr_reduction=lr_reduction,
                    max_epochs=max_epochs,
                    base_val_loss=base_checkpoint.val_loss,
                )

                if success:
                    model_results["successful"] += 1
                else:
                    model_results["failed_seeds"].append(seed)
                    logger.info(f"✗ Failed: {model_name} seed={seed}")

            except Exception as e:
                model_results["failed_seeds"].append(seed)
                logger.info(f"✗ Failed: {model_name} seed={seed}")
                logger.error(f"Error fine-tuning {model_name} seed={seed}: {e}")
                warnings.warn(f"Fine-tuning failed for {model_name} seed={seed}: {e}", stacklevel=2)

        results["models"][model_name] = model_results

    # Calculate total time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    results["total_time"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return results


def finetune_single_model(
    base_checkpoint_path: Path,
    config_path: Path,
    model_name: str,
    seed: int,
    checkpoint_dir: Path,
    lr_reduction: float,
    max_epochs: int,
    base_val_loss: float | None = None,
) -> bool:
    """
    Fine-tune a single model from a base checkpoint.

    Args:
        base_checkpoint_path: Path to the base checkpoint to load
        config_path: Path to model configuration YAML file
        model_name: Name of the model (for logging)
        seed: Random seed for reproducibility
        checkpoint_dir: Directory to save fine-tuned checkpoints
        lr_reduction: Factor to reduce learning rate by
        max_epochs: Maximum epochs for fine-tuning
        base_val_loss: Validation loss of base checkpoint (for metadata)

    Returns:
        True if successful, False if fine-tuning failed
    """
    try:
        # Set seed BEFORE loading model to ensure reproducibility
        pl.seed_everything(seed, workers=True)

        # Load pre-trained model from checkpoint
        logger.info(f"Loading checkpoint: {base_checkpoint_path}")
        model = ModelFactory.create_from_checkpoint(model_name, base_checkpoint_path)

        # Store original learning rate
        original_lr = model.config.learning_rate

        # Modify learning rate
        new_lr = original_lr / lr_reduction
        model.config.learning_rate = new_lr

        logger.info(f"Modified learning rate: {original_lr:.6f} -> {new_lr:.6f} (÷{lr_reduction})")

        # Save fine-tuning metadata
        save_finetune_metadata(
            checkpoint_dir=checkpoint_dir,
            base_checkpoint_path=base_checkpoint_path,
            base_val_loss=base_val_loss,
            lr_reduction_factor=lr_reduction,
            original_lr=original_lr,
            finetuned_lr=new_lr,
            seed=seed,
            experiment_config=str(config_path),
        )

        # Create data module with same configuration
        datamodule = LSHDataModule(config_path)

        # Create trainer for fine-tuning
        # Note: This will use tensorboard/model_name_seed{seed}/ directory
        trainer = create_trainer(
            model_name=model_name,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            tensorboard_dir=Path("tensorboard"),
            max_epochs=max_epochs,
        )

        # Fine-tune the model
        trainer.fit(model, datamodule)

        return True

    except Exception as e:
        logger.error(f"Fine-tuning failed for {model_name} with seed={seed}: {e}")
        return False


def create_finetuning_checkpoint_path(
    model_name: str, timestamp: str, seed: int, base_dir: str = "checkpoints/finetuning"
) -> Path:
    """
    Constructs hive-partitioned checkpoint path for fine-tuning:
    checkpoints/finetuning/model_name=tide/run_2024-11-20_seed42/

    Args:
        model_name: Name of the model
        timestamp: Timestamp string (YYYY-MM-DD format)
        seed: Random seed value
        base_dir: Base directory for fine-tuning checkpoints

    Returns:
        Path object for checkpoint directory
    """
    checkpoint_path = Path(base_dir) / f"model_name={model_name}" / f"run_{timestamp}_seed{seed}"
    # Note: Directory is NOT created here - that's done by the trainer
    return checkpoint_path


def is_finetuning_complete(checkpoint_dir: Path) -> bool:
    """
    Check if a fine-tuning run is complete by looking for best checkpoint.

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


def save_finetune_metadata(
    checkpoint_dir: Path,
    base_checkpoint_path: Path,
    base_val_loss: float | None,
    lr_reduction_factor: float,
    original_lr: float,
    finetuned_lr: float,
    seed: int,
    experiment_config: str,
) -> None:
    """
    Save metadata about the fine-tuning process.

    Args:
        checkpoint_dir: Directory where fine-tuned checkpoint will be saved
        base_checkpoint_path: Path to the base checkpoint used
        base_val_loss: Validation loss of base checkpoint
        lr_reduction_factor: Factor by which LR was reduced
        original_lr: Original learning rate
        finetuned_lr: New learning rate after reduction
        seed: Random seed used
        experiment_config: Path to experiment configuration file
    """
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Prepare metadata
    metadata = {
        "base_checkpoint": str(base_checkpoint_path),
        "base_val_loss": base_val_loss,
        "lr_reduction_factor": lr_reduction_factor,
        "original_lr": original_lr,
        "finetuned_lr": finetuned_lr,
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "experiment_config": experiment_config,
    }

    # Save metadata to YAML file
    metadata_path = checkpoint_dir / "finetune_metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    logger.debug(f"Saved fine-tuning metadata to {metadata_path}")
