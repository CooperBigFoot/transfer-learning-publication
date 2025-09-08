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

    # Pre-computed indices for fast access
    target_idx: int | None = None  # Column index of target variable
    forcing_indices: list[int] | None = None  # Column indices of forcing features
    future_indices: list[int] | None = None  # Column indices of future features
    input_feature_indices: list[int] | None = None  # All input feature indices (computed by DataModule)

    is_autoregressive: bool = True
    include_dates: bool = False

    # Metadata
    group_identifier_name: str = "entity_id"

    def __post_init__(self):
        """Validate configuration consistency."""
        # Validate future features are subset of forcing features
        if self.future_features:
            future_set = set(self.future_features)
            forcing_set = set(self.forcing_features)
            if not future_set.issubset(forcing_set):
                invalid = future_set - forcing_set
                raise ValueError(f"Future features must be subset of forcing features. Invalid: {invalid}")

        # Validate indices if provided
        if self.future_indices and self.forcing_indices:
            future_set = set(self.future_indices)
            forcing_set = set(self.forcing_indices)
            if not future_set.issubset(forcing_set):
                raise ValueError(
                    f"Future indices {self.future_indices} must be subset of "
                    f"forcing indices {self.forcing_indices}"
                )

        # CRITICAL: Prevent data leakage - target must NEVER be in future features
        if self.is_autoregressive and self.target_idx is not None and self.future_indices and self.target_idx in self.future_indices:
            raise ValueError(
                f"Data leakage detected: Target index {self.target_idx} cannot be in "
                f"future_indices {self.future_indices}. Future target values are unknown "
                f"at prediction time. Remove target index from future_indices."
            )

        # Validate that target is not in forcing features for non-autoregressive models
        if not self.is_autoregressive and self.target_name in self.forcing_features:
            raise ValueError(
                f"Invalid configuration: Target '{self.target_name}' cannot be in forcing_features "
                f"when is_autoregressive=False. Non-autoregressive models should not use the target "
                f"as an input feature. Either set is_autoregressive=True or remove '{self.target_name}' "
                f"from forcing_features."
            )
