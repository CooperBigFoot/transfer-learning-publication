from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for dataset behavior."""

    input_length: int
    output_length: int

    target_name: str
    forcing_features: list[str]
    static_features: list[str]
    future_features: list[str] = field(default_factory=list)  # Features known in future

    target_idx: int | None = None
    forcing_indices: list[int] | None = None
    future_indices: list[int] | None = None

    is_autoregressive: bool = True
    include_dates: bool = False

    # Metadata
    group_identifier_name: str = "entity_id"

    def __post_init__(self):
        """Validate configuration consistency."""
        if self.future_features:
            future_set = set(self.future_features)
            forcing_set = set(self.forcing_features)
            if not future_set.issubset(forcing_set):
                invalid = future_set - forcing_set
                raise ValueError(f"Future features must be subset of forcing features. Invalid: {invalid}")
