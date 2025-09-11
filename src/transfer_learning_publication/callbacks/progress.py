"""
Provides minimal, informative progress updates during training.
"""

import logging
from datetime import datetime

from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


class MinimalProgressCallback(Callback):
    """
    Lightning callback that prints minimal progress.
    """

    def __init__(self, model_name: str, seed: int):
        """
        Initialize the progress callback.

        Args:
            model_name: Name of the model being trained
            seed: Random seed being used
        """
        super().__init__()
        self.model_name = model_name
        self.seed = seed
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        """Record training start time."""
        self.start_time = datetime.now()

    def on_train_epoch_start(self, trainer, pl_module):
        """Log minimal progress at epoch start."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current = trainer.current_epoch + 1
        total = trainer.max_epochs
        logger.info(f"[{timestamp}] Training: model_name={self.model_name}, seed={self.seed}, epoch {current}/{total}")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation loss after each validation epoch."""
        if trainer.callback_metrics.get("val_loss") is not None:
            val_loss = trainer.callback_metrics["val_loss"].item()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            epoch = trainer.current_epoch + 1
            logger.info(
                f"[{timestamp}] Validation: model_name={self.model_name}, "
                f"seed={self.seed}, epoch {epoch}, val_loss={val_loss:.4f}"
            )


def print_experiment_summary(results: dict):
    """
    Logs final summary of the experiment.

    Args:
        results: Dictionary containing experiment results with structure:
            {
                "total_time": str,
                "timestamp": str,
                "models": {
                    "model_name": {
                        "total": int,
                        "successful": int,
                        "failed_seeds": list
                    }
                }
            }
    """
    logger.info("\n" + "=" * 60)
    logger.info("Experiment Complete!")
    logger.info("=" * 60)

    # Print results for each model
    for model_name, model_results in results["models"].items():
        successful = model_results["successful"]
        total = model_results["total"]
        failed_seeds = model_results.get("failed_seeds", [])

        if successful == total:
            status = "✓"
            message = f"{model_name}: {successful}/{total} runs successful"
        else:
            status = "⚠"
            failed_list = ", ".join([f"seed{s}" for s in failed_seeds])
            message = f"{model_name}: {successful}/{total} runs successful (failed: {failed_list})"

        logger.info(f"{status} {message}")

    # Log summary information
    logger.info(f"\nTotal time: {results['total_time']}")
    logger.info(f"Experiment timestamp: {results['timestamp']}")
    logger.info("Checkpoints saved to: checkpoints/training/")
    logger.info("TensorBoard logs: tensorboard/")
    logger.info("\nView training curves with: tensorboard --logdir tensorboard/")
    logger.info("=" * 60)
