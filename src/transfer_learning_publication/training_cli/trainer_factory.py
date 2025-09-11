import logging
from pathlib import Path

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from ..callbacks.progress import MinimalProgressCallback
from ..data import LSHDataModule
from ..models import ModelFactory

logger = logging.getLogger(__name__)


def create_trainer(
    model_name: str,
    seed: int,
    checkpoint_dir: Path,
    tensorboard_dir: Path = Path("tensorboard"),
    max_epochs: int | None = None,
) -> pl.Trainer:
    """
    Creates trainer with:
    - ModelCheckpoint callback with custom naming:
      * filename pattern: "best_val_loss_{val_loss:.4f}"
      * Saves best model as: best_val_loss_0.0234.ckpt
      * Also saves last.ckpt
    - CSVLogger (saves metrics.csv to checkpoint_dir)
    - TensorBoardLogger (saves to tensorboard/{model_name}_seed{seed}/)
    - EarlyStopping (patience=10, monitor='val_loss')
    - Minimal progress output (just model name and epoch)
    - Deterministic training with specified seed

    Args:
        model_name: Name of the model being trained
        seed: Random seed for reproducibility
        checkpoint_dir: Directory to save checkpoints
        tensorboard_dir: Directory for TensorBoard logs
        max_epochs: Maximum number of epochs to train

    Returns:
        Configured PyTorch Lightning Trainer
    """
    # Create directories if needed
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir / "checkpoints",
            filename="best_val_loss_{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
            verbose=False,
        ),
        EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=False),
        MinimalProgressCallback(model_name=model_name, seed=seed),
    ]

    # Set up loggers
    loggers = [
        CSVLogger(save_dir=checkpoint_dir, name="", version=""),
        TensorBoardLogger(save_dir=tensorboard_dir, name=f"{model_name}_seed{seed}", version=""),
    ]

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs or 100,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        accelerator="auto",
        devices=1,
    )

    # Set seed for reproducibility
    pl.seed_everything(seed, workers=True)

    return trainer


def train_single_model(
    config_path: Path,
    model_name: str,
    seed: int,
    checkpoint_dir: Path,
    tensorboard_dir: Path,
    max_epochs: int | None = None,
) -> bool:
    """
    Trains a single model configuration.
    Uses ModelFactory and LSHDataModule from the main package.

    Args:
        config_path: Path to model configuration YAML file
        model_name: Name of the model (for logging)
        seed: Random seed for reproducibility
        checkpoint_dir: Directory to save checkpoints
        tensorboard_dir: Directory for TensorBoard logs
        max_epochs: Maximum number of epochs to train

    Returns:
        True if successful, False if training failed
    """
    try:
        # Create model from config
        model = ModelFactory.create_from_config(config_path)

        # Create data module from same config
        datamodule = LSHDataModule(config_path)

        # Create trainer
        trainer = create_trainer(
            model_name=model_name,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            tensorboard_dir=tensorboard_dir,
            max_epochs=max_epochs,
        )

        # Train the model
        trainer.fit(model, datamodule)

        return True

    except Exception as e:
        logger.error(f"Training failed for {model_name} with seed={seed}: {e}")
        return False
