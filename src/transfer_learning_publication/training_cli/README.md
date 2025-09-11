# Training CLI

A simple command-line interface for orchestrating model training experiments with automatic multi-seed runs and intelligent resumption capabilities.

## Installation

The CLI is automatically installed when you install the package:

```bash
uv sync
```

## Usage

### Basic Training

Train all models defined in an experiment configuration:

```bash
uv run tl-train experiments/baseline_experiment.yaml
```

### Train Specific Models

Train only specific models from the experiment:

```bash
uv run tl-train experiments/multi_model_experiment.yaml --models tide,ealstm
```

### Multi-Seed Training

Train models with multiple random seeds for reproducibility:

```bash
uv run tl-train experiments/baseline_experiment.yaml --n-runs 10 --start-seed 42
```

### Force Restart

Ignore existing checkpoints and restart training from scratch:

```bash
uv run tl-train experiments/baseline_experiment.yaml --fresh
```

### Resume Interrupted Training

Simply re-run the same command - the CLI automatically detects and skips completed runs:

```bash
# Initial run (interrupted)
uv run tl-train experiments/baseline_experiment.yaml --n-runs 10

# Resume (will skip completed runs)
uv run tl-train experiments/baseline_experiment.yaml --n-runs 10
```

## Experiment Configuration

Create an experiment YAML file that defines which models to train:

```yaml
# experiments/my_experiment.yaml
models:
  tide_baseline: configs/models/tide_config.yaml
  ealstm_baseline: configs/models/ealstm_config.yaml
  tsmixer: configs/models/tsmixer_config.yaml
```

Each model configuration file should contain both data and model settings. See `configs/models/` for examples.

## Output Structure

The CLI organizes outputs in a clear, queryable structure:

### Checkpoints

```
checkpoints/training/
├── model_name=tide/
│   ├── run_2024-11-20_seed42/
│   │   ├── checkpoints/
│   │   │   ├── best_val_loss_0.0234.ckpt
│   │   │   └── last.ckpt
│   │   └── metrics.csv
│   └── run_2024-11-20_seed43/
└── model_name=ealstm/
    └── run_2024-11-20_seed42/
```

### TensorBoard Logs

```
tensorboard/
├── tide_seed42/
├── tide_seed43/
└── ealstm_seed42/
```

View training curves:

```bash
tensorboard --logdir tensorboard/
```

## Features

- **Simple by default**: Just run `tl-train experiment.yaml`
- **Automatic resumption**: Re-run the same command to resume interrupted experiments
- **Multi-seed support**: Train with multiple seeds for reproducible research
- **Progress tracking**: Minimal, informative progress updates
- **Smart checkpointing**: Descriptive checkpoint names show validation loss at a glance
- **Hive-partitioned output**: Organized directory structure for easy querying

## Implementation Details

The CLI consists of four main components:

1. **`__main__.py`**: Entry point with Click CLI interface
2. **`orchestrator.py`**: Manages experiment execution and resumption logic
3. **`trainer_factory.py`**: Creates configured PyTorch Lightning trainers
4. **`progress.py`**: Provides minimal progress updates and summaries

The CLI leverages existing components:
- `ModelFactory.create_from_config()` for model instantiation
- `LSHDataModule` for data loading
- PyTorch Lightning for training orchestration