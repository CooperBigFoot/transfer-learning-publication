import torch

from .batch import Batch
from .sample import Sample


def collate_samples(samples: list[Sample]) -> Batch:
    """
    Collate a list of samples into a batch.

    This function transforms individual samples into a batched format
    suitable for training. It follows the principle of trust - assuming
    all samples from the Dataset are valid and consistent.

    Args:
        samples: List of Sample objects to batch together

    Returns:
        Batch object with stacked tensors

    Raises:
        ValueError: If samples list is empty
        RuntimeError: If tensor stacking fails (indicates inconsistent sample shapes)
    """
    if not samples:
        raise ValueError("Cannot collate empty list of samples")

    try:
        X = torch.stack([sample.X for sample in samples])
        y = torch.stack([sample.y for sample in samples])
        static = torch.stack([sample.static for sample in samples])

        future = torch.stack([sample.future for sample in samples])

        group_identifiers = [sample.group_identifier for sample in samples]

        if samples[0].input_end_date is not None:
            # Convert list of ints to tensor
            input_end_dates = torch.tensor([sample.input_end_date for sample in samples], dtype=torch.long)
        else:
            input_end_dates = None

    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to stack samples into batch. This indicates inconsistent sample shapes from Dataset: {e}"
        ) from e

    return Batch(
        X=X, y=y, static=static, future=future, group_identifiers=group_identifiers, input_end_dates=input_end_dates
    )


def collate_fn(samples: list[Sample]) -> Batch:
    """
    Alias for collate_samples to match PyTorch DataLoader convention.

    PyTorch DataLoader expects a callable named 'collate_fn'.
    This provides that interface while keeping the descriptive name
    for the actual implementation.
    """
    return collate_samples(samples)
