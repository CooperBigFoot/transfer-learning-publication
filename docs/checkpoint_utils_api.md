# Checkpoint Utils API Reference

The `checkpoint_utils` module provides tools for discovering, querying, and selecting model checkpoints from hive-partitioned directory structures.

## Classes

### `Checkpoint`

A dataclass representing a single checkpoint with metadata.

**Attributes:**

- `path: Path` - Full path to checkpoint file
- `model_name: str` - Model identifier (e.g., "tide", "ealstm")
- `timestamp: str` - Training run timestamp (YYYY-MM-DD format)
- `seed: int` - Random seed used
- `checkpoint_type: str` - Either "best_val_loss" or "last"
- `val_loss: float | None` - Validation loss from filename
- `stage: str` - Either "training" or "finetuning"
- `rank: int | None` - Ranking position when compared
- `percentile: float | None` - Percentile position (0-100)
- `total_runs: int | None` - Total runs being compared
- `version: int | None` - Version number if multiple versions exist

**Methods:**

- `exists() -> bool` - Check if checkpoint file exists on disk
- `get_file_size_mb() -> float` - Get file size in megabytes

### `CheckpointDiscovery`

Main interface for checkpoint discovery and querying.

## Core Query Methods

### `get_checkpoint(model_name, seed, stage="training", checkpoint_type="best_val_loss", timestamp=None)`

Retrieve a specific checkpoint based on criteria.

```python
ckpt = discovery.get_checkpoint("tide", 42, timestamp="2024-11-20")
```

### `get_best_checkpoint(model_name, seed, stage="training")`

Get the most recent best checkpoint.

```python
best = discovery.get_best_checkpoint("tide", 42)
```

### `list_runs(model_name=None, seed=None, stage="training")`

List all available training runs with metadata.

```python
runs = discovery.list_runs(model_name="tide")
```

### `exists(model_name, seed, timestamp, stage="training", checkpoint_type="best_val_loss")`

Check if a specific checkpoint exists.

```python
if discovery.exists("tide", 42, "2024-11-20"):
    print("Checkpoint found")
```

### `validate_checkpoint(checkpoint)`

Verify checkpoint file exists and is potentially loadable.

```python
if not discovery.validate_checkpoint(ckpt):
    print("Warning: Checkpoint may be corrupted")
```

## Statistical Selection Methods

### `get_median_checkpoint(model_name, seeds=None, stage="training", timestamp=None)`

Get the median performing checkpoint across multiple seeds.

```python
# For paper reporting
median = discovery.get_median_checkpoint("tide")
print(f"Median model: seed {median.seed}, loss {median.val_loss:.4f}")
```

### `get_checkpoint_rankings(model_name, seeds=None, stage="training", timestamp=None)`

Get all checkpoints ranked by performance (best first).

```python
rankings = discovery.get_checkpoint_rankings("tide")
for ckpt in rankings:
    print(f"Rank {ckpt.rank}: Seed {ckpt.seed} = {ckpt.val_loss:.4f}")
```

### `get_percentile_checkpoint(model_name, percentile, seeds=None, stage="training", timestamp=None)`

Get checkpoint at specific percentile (0=best, 100=worst).

```python
q1 = discovery.get_percentile_checkpoint("tide", 25)
median = discovery.get_percentile_checkpoint("tide", 50)
q3 = discovery.get_percentile_checkpoint("tide", 75)
```

### `get_run_statistics(model_name, seeds=None, stage="training", timestamp=None)`

Compute summary statistics across multiple seeds.

**Returns dict with:**

- `count`, `mean`, `std`, `min`, `max`, `median`
- `q1`, `q3` (quartiles)
- `best_seed`, `median_seed`, `worst_seed`

```python
stats = discovery.get_run_statistics("tide")
print(f"TiDE: {stats['mean']:.4f} ± {stats['std']:.4f}")
```

## Fine-tuning Support Methods

### `get_parent_checkpoint(model_name, seed, timestamp=None)`

Get training checkpoint to use as parent for fine-tuning.

```python
parent = discovery.get_parent_checkpoint("tide", 42)
```

### `find_finetuned_checkpoints(parent_checkpoint)`

Find all fine-tuned versions of a parent checkpoint.

```python
finetuned = discovery.find_finetuned_checkpoints(parent)
```

## Utility Methods

### `get_checkpoints_for_model(model_name, stage="training")`

Get all checkpoints organized by seed.

```python
checkpoints = discovery.get_checkpoints_for_model("tide")
# Returns: {42: [ckpt1, ckpt2], 43: [ckpt3, ckpt4], ...}
```

### `get_latest_timestamp(model_name, seed, stage="training")`

Find most recent timestamp for model/seed combination.

```python
latest = discovery.get_latest_timestamp("tide", 42)
```

## Usage Examples

### Paper Reporting Workflow

```python
from transfer_learning_publication.checkpoint_utils import CheckpointDiscovery
from transfer_learning_publication.models import ModelFactory

discovery = CheckpointDiscovery()

# Get median checkpoint for fair comparison
median = discovery.get_median_checkpoint("tide")
stats = discovery.get_run_statistics("tide")

print(f"Using median model (seed {median.seed})")
print(f"Performance: {stats['mean']:.4f} ± {stats['std']:.4f}")

# Load the median model
model = ModelFactory.create_from_checkpoint("tide", median.path)
```

### Finding Best Model

```python
# Get rankings to find best model
rankings = discovery.get_checkpoint_rankings("tide")
best = rankings[0]  # First is best

print(f"Best model: seed {best.seed}, loss {best.val_loss:.4f}")
model = ModelFactory.create_from_checkpoint("tide", best.path)
```

### Ensemble Creation

```python
# Get top 3 models for ensemble
rankings = discovery.get_checkpoint_rankings("tide")
top_3 = rankings[:3]

ensemble_models = []
for ckpt in top_3:
    model = ModelFactory.create_from_checkpoint("tide", ckpt.path)
    ensemble_models.append(model)
```

## Directory Structure

The module expects checkpoints in hive-partitioned format:

```
checkpoints/
├── training/
│   └── model_name=tide/
│       └── run_2024-11-20_seed42/
│           ├── checkpoints/
│           │   ├── best_val_loss_0.0234.ckpt
│           │   └── last.ckpt
│           └── metrics.csv
└── finetuning/
    └── [similar structure]
```
