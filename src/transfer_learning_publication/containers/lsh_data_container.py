from dataclasses import dataclass

from .dataset_config import DatasetConfig
from .sequence_index import SequenceIndex
from .static_attribute import StaticAttributeCollection
from .time_series import TimeSeriesCollection


@dataclass(frozen=True)
class LSHDataContainer:
    """
    Immutable container holding all data needed by Dataset.

    This is a pure data structure with no methods - it simply
    binds together the data collections and configuration.
    """

    time_series: TimeSeriesCollection
    static_attributes: StaticAttributeCollection
    sequence_index: SequenceIndex
    config: DatasetConfig

    def __post_init__(self):
        """Validate container consistency."""
        # Ensure all collections reference same groups
        if len(self.time_series) != len(self.static_attributes):
            raise ValueError(
                f"Inconsistent group counts: time_series has {len(self.time_series)}, "
                f"static_attributes has {len(self.static_attributes)}"
            )

        # Validate sequence index matches data
        if self.sequence_index.n_groups != len(self.time_series):
            raise ValueError(
                f"Sequence index built for {self.sequence_index.n_groups} groups, "
                f"but time_series has {len(self.time_series)} groups"
            )

        # Validate config dimensions match index
        if self.sequence_index.input_length != self.config.input_length:
            raise ValueError(
                f"Config input_length ({self.config.input_length}) doesn't match "
                f"sequence index ({self.sequence_index.input_length})"
            )

        if self.sequence_index.output_length != self.config.output_length:
            raise ValueError(
                f"Config output_length ({self.config.output_length}) doesn't match "
                f"sequence index ({self.sequence_index.output_length})"
            )

        # Validate feature indices are within bounds
        n_features = self.time_series.get_n_features()

        if self.config.target_idx is not None and not 0 <= self.config.target_idx < n_features:
            raise ValueError(
                f"Target index {self.config.target_idx} out of bounds [0, {n_features})"
            )

        if self.config.forcing_indices:
            invalid = [i for i in self.config.forcing_indices if not 0 <= i < n_features]
            if invalid:
                raise ValueError(
                    f"Forcing indices {invalid} out of bounds [0, {n_features})"
                )

        if self.config.future_indices:
            invalid = [i for i in self.config.future_indices if not 0 <= i < n_features]
            if invalid:
                raise ValueError(
                    f"Future indices {invalid} out of bounds [0, {n_features})"
                )

        if self.config.input_feature_indices:
            invalid = [i for i in self.config.input_feature_indices if not 0 <= i < n_features]
            if invalid:
                raise ValueError(
                    f"Input feature indices {invalid} out of bounds [0, {n_features})"
                )

        # Validate static features match
        n_static = self.static_attributes.get_n_attributes()
        if len(self.config.static_features) != n_static:
            raise ValueError(
                f"Config has {len(self.config.static_features)} static features, "
                f"but static_attributes has {n_static}"
            )
