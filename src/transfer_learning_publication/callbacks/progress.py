"""
Provides minimal, informative progress updates during training.
"""

import logging
import sys
from datetime import datetime

from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


class MinimalProgressCallback(Callback):
    """
    Lightning callback that prints minimal progress on a single line.
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
        self.last_val_loss = None

    def on_train_start(self, trainer, pl_module):
        """Record training start time and print initial message."""
        self.start_time = datetime.now()
        # Force flush to ensure message appears before progress bar
        print(f"\nTraining {self.model_name} (seed={self.seed})", flush=True)

    def on_train_epoch_start(self, trainer, pl_module):
        """Update progress on single line."""
        current = trainer.current_epoch + 1
        total = trainer.max_epochs

        # Create progress bar
        progress = current / total
        bar_length = 30
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)

        # Build status message
        if self.last_val_loss is not None:
            status = f"\r  [{bar}] Epoch {current}/{total} | val_loss: {self.last_val_loss:.4f}"
        else:
            status = f"\r  [{bar}] Epoch {current}/{total}"

        # Write to stdout for immediate update
        sys.stdout.write(status)
        sys.stdout.flush()

    def on_validation_epoch_end(self, trainer, pl_module):
        """Update validation loss."""
        if trainer.callback_metrics.get("val_loss") is not None:
            self.last_val_loss = trainer.callback_metrics["val_loss"].item()

            # Update the progress line with new validation loss
            current = trainer.current_epoch + 1
            total = trainer.max_epochs
            progress = current / total
            bar_length = 30
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)

            status = f"\r  [{bar}] Epoch {current}/{total} | val_loss: {self.last_val_loss:.4f}"
            sys.stdout.write(status)
            sys.stdout.flush()

    def on_train_end(self, trainer, pl_module):
        """Print final newline and summary."""
        # Move to new line after progress bar
        print()  # Print newline to move past progress bar

        if self.last_val_loss is not None:
            print(f"✓ Completed {self.model_name} (seed={self.seed}): final val_loss={self.last_val_loss:.4f}")


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
