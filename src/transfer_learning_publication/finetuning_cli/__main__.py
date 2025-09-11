"""
Main entry point for the fine-tuning CLI.
Handles argument parsing and dispatches to fine-tuning orchestrator.

Usage:
    tl-finetune experiment.yaml
    tl-finetune experiment.yaml --models tide,ealstm --lr-reduction 25 --n-runs 3
    tl-finetune experiment.yaml --models tide --base-seed 42 --max-epochs 50
"""

import logging
from pathlib import Path

import click

from .finetune_orchestrator import run_finetuning


@click.command()
@click.argument("experiment_path", type=click.Path(exists=True))
@click.option("--models", help="Comma-separated list of models to fine-tune")
@click.option("--lr-reduction", default=25.0, help="Learning rate reduction factor (default: 25)")
@click.option("--base-seed", type=int, help="Specific seed checkpoint to use as base (optional)")
@click.option("--n-runs", default=1, help="Number of fine-tuning runs per model")
@click.option("--start-seed", default=42, help="Starting seed value for fine-tuning")
@click.option("--max-epochs", type=int, help="Maximum epochs for fine-tuning (optional)")
@click.option("--fresh", is_flag=True, help="Ignore existing fine-tuned checkpoints and restart")
def main(experiment_path, models, lr_reduction, base_seed, n_runs, start_seed, max_epochs, fresh):
    """Fine-tune models from experiment configuration with reduced learning rate."""
    # Configure logging to show INFO level messages
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",  # Simple format for clean output
    )

    # Parse models if provided
    if models:
        # Filter out empty strings from split
        model_list = [m.strip() for m in models.split(",") if m.strip()]
        # If all were empty, treat as None
        model_list = model_list if model_list else None
    else:
        model_list = None

    # Run the fine-tuning
    results = run_finetuning(
        experiment_path=Path(experiment_path),
        models=model_list,
        lr_reduction=lr_reduction,
        base_seed=base_seed,
        n_runs=n_runs,
        start_seed=start_seed,
        max_epochs=max_epochs,
        fresh=fresh,
    )

    # Print summary (reuse existing function with modified message)
    # The summary function will need slight modification to show finetuning path
    _print_finetuning_summary(results)


def _print_finetuning_summary(results: dict):
    """
    Print summary specifically for fine-tuning experiments.

    Args:
        results: Dictionary containing fine-tuning results
    """
    logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 60)
    logger.info("Fine-tuning Complete!")
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
    logger.info(f"Fine-tuning timestamp: {results['timestamp']}")
    logger.info(f"LR reduction factor: {results.get('lr_reduction', 'N/A')}")
    logger.info("Checkpoints saved to: checkpoints/finetuning/")
    logger.info("TensorBoard logs: tensorboard/")
    logger.info("\nView training curves with: tensorboard --logdir tensorboard/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
