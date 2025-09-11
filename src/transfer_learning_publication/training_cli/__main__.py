"""
Main entry point for the training CLI.
Handles argument parsing and dispatches to appropriate commands.

Usage:
    tl-train experiment.yaml
    tl-train experiment.yaml --models tide,ealstm --n-runs 10 --start-seed 42
"""

import logging
from pathlib import Path

import click

from ..callbacks.progress import print_experiment_summary
from .orchestrator import run_experiment


@click.command()
@click.argument("experiment_path", type=click.Path(exists=True))
@click.option("--models", help="Comma-separated list of models to train")
@click.option("--n-runs", default=1, help="Number of runs per model")
@click.option("--start-seed", default=42, help="Starting seed value")
@click.option("--fresh", is_flag=True, help="Ignore existing checkpoints and restart")
def main(experiment_path, models, n_runs, start_seed, fresh):
    """Train models from experiment configuration."""
    # Configure logging to show INFO level messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'  # Simple format for clean output
    )

    # Parse models if provided
    if models:
        # Filter out empty strings from split
        model_list = [m.strip() for m in models.split(",") if m.strip()]
        # If all were empty, treat as None
        model_list = model_list if model_list else None
    else:
        model_list = None

    # Run the experiment
    results = run_experiment(
        Path(experiment_path), models=model_list, n_runs=n_runs, start_seed=start_seed, fresh=fresh
    )

    # Print summary
    print_experiment_summary(results)


if __name__ == "__main__":
    main()
