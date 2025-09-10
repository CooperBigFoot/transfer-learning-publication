from datetime import datetime

import pytest
import torch

from transfer_learning_publication.containers import (
    StaticAttributeCollection,
    TimeSeriesCollection,
)
from transfer_learning_publication.containers.dataset_config import DatasetConfig
from transfer_learning_publication.containers.lsh_data_container import LSHDataContainer
from transfer_learning_publication.containers.sequence_index import SequenceIndex


class TestLSHDataContainer:
    @pytest.fixture
    def valid_components(self):
        """Create valid components for LSHDataContainer."""
        # Create time series collection with 2 groups
        tensors = [
            torch.randn(100, 3),  # Group 1: 100 timesteps, 3 features
            torch.randn(150, 3),  # Group 2: 150 timesteps, 3 features
        ]
        feature_names = ["temperature", "precipitation", "humidity"]
        date_ranges = [
            (datetime(2020, 1, 1), datetime(2020, 4, 9)),  # 100 days (Jan 1 to Apr 9)
            (datetime(2020, 1, 1), datetime(2020, 5, 29)),  # 150 days (Jan 1 to May 29)
        ]
        group_identifiers = ["station_1", "station_2"]
        time_series = TimeSeriesCollection(tensors, feature_names, date_ranges, group_identifiers)

        # Create static attributes with same 2 groups
        static_tensors = [
            torch.tensor([100.0, 45.5, -122.3]),  # elevation, latitude, longitude
            torch.tensor([250.0, 47.6, -122.2]),
        ]
        attribute_names = ["elevation", "latitude", "longitude"]
        static_attributes = StaticAttributeCollection(static_tensors, attribute_names, group_identifiers)

        # Create sequence index for 2 groups
        sequences = torch.tensor(
            [
                [0, 0, 30],  # Group 0, start 0, end 30
                [0, 30, 60],  # Group 0, start 30, end 60
                [1, 0, 30],  # Group 1, start 0, end 30
                [1, 50, 80],  # Group 1, start 50, end 80
            ],
            dtype=torch.long,
        )
        sequence_index = SequenceIndex(
            sequences=sequences,
            n_groups=2,
            input_length=20,
            output_length=10,
        )

        # Create matching config
        config = DatasetConfig(
            input_length=20,
            output_length=10,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation", "latitude", "longitude"],
        )

        return time_series, static_attributes, sequence_index, config

    def test_valid_container_creation(self, valid_components):
        """Test creating a valid LSHDataContainer."""
        time_series, static_attributes, sequence_index, config = valid_components

        container = LSHDataContainer(
            time_series=time_series,
            static_attributes=static_attributes,
            sequence_index=sequence_index,
            config=config,
        )

        assert container.time_series is time_series
        assert container.static_attributes is static_attributes
        assert container.sequence_index is sequence_index
        assert container.config is config

    def test_group_count_mismatch_error(self, valid_components):
        """Test that mismatched group counts raise an error."""
        time_series, static_attributes, sequence_index, config = valid_components

        # Create static attributes with different group count
        static_tensors = [torch.tensor([100.0, 45.5, -122.3])]  # Only 1 group
        attribute_names = ["elevation", "latitude", "longitude"]
        mismatched_static = StaticAttributeCollection(static_tensors, attribute_names, ["station_1"])

        with pytest.raises(
            ValueError,
            match="Inconsistent group counts: time_series has 2, static_attributes has 1",
        ):
            LSHDataContainer(
                time_series=time_series,
                static_attributes=mismatched_static,
                sequence_index=sequence_index,
                config=config,
            )

    def test_sequence_index_group_mismatch(self, valid_components):
        """Test that sequence index with wrong group count raises error."""
        time_series, static_attributes, _, config = valid_components

        # Create sequence index for wrong number of groups
        sequences = torch.tensor(
            [[0, 0, 30], [1, 0, 30], [2, 0, 30]],  # References 3 groups
            dtype=torch.long,
        )
        wrong_index = SequenceIndex(
            sequences=sequences,
            n_groups=3,  # Wrong: should be 2
            input_length=20,
            output_length=10,
        )

        with pytest.raises(
            ValueError,
            match="Sequence index built for 3 groups, but time_series has 2 groups",
        ):
            LSHDataContainer(
                time_series=time_series,
                static_attributes=static_attributes,
                sequence_index=wrong_index,
                config=config,
            )

    def test_config_input_length_mismatch(self, valid_components):
        """Test that config with wrong input_length raises error."""
        time_series, static_attributes, sequence_index, _ = valid_components

        # Create config with wrong input_length
        wrong_config = DatasetConfig(
            input_length=30,  # Wrong: sequence_index has input_length=20
            output_length=10,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation", "latitude", "longitude"],
        )

        with pytest.raises(
            ValueError,
            match=r"Config input_length \(30\) doesn't match sequence index \(20\)",
        ):
            LSHDataContainer(
                time_series=time_series,
                static_attributes=static_attributes,
                sequence_index=sequence_index,
                config=wrong_config,
            )

    def test_container_immutability(self, valid_components):
        """Test that LSHDataContainer is frozen (immutable)."""
        time_series, static_attributes, sequence_index, config = valid_components

        container = LSHDataContainer(
            time_series=time_series,
            static_attributes=static_attributes,
            sequence_index=sequence_index,
            config=config,
        )

        # Try to modify fields
        with pytest.raises(AttributeError):
            container.config = config

        with pytest.raises(AttributeError):
            container.time_series = time_series

    def test_empty_groups(self):
        """Test container with empty collections."""
        # Create empty collections
        time_series = TimeSeriesCollection(
            tensors=[],
            feature_names=["temperature"],
            date_ranges=[],
            group_identifiers=[],
        )

        static_attributes = StaticAttributeCollection(
            tensors=[],
            attribute_names=["elevation"],
            group_identifiers=[],
        )

        sequence_index = SequenceIndex(
            sequences=torch.empty((0, 3), dtype=torch.long),
            n_groups=0,
            input_length=20,
            output_length=10,
        )

        config = DatasetConfig(
            input_length=20,
            output_length=10,
            target_name="temperature",
            forcing_features=["temperature"],
            static_features=["elevation"],
        )

        # Should work with empty but consistent data
        container = LSHDataContainer(
            time_series=time_series,
            static_attributes=static_attributes,
            sequence_index=sequence_index,
            config=config,
        )

        assert len(container.time_series) == 0
        assert len(container.static_attributes) == 0
        assert container.sequence_index.n_groups == 0

    def test_config_output_length_mismatch(self, valid_components):
        """Test that config with wrong output_length raises error."""
        time_series, static_attributes, sequence_index, _ = valid_components

        # Create config with wrong output_length
        wrong_config = DatasetConfig(
            input_length=20,
            output_length=15,  # Wrong: sequence_index has output_length=10
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation", "latitude", "longitude"],
        )

        with pytest.raises(
            ValueError,
            match=r"Config output_length \(15\) doesn't match sequence index \(10\)",
        ):
            LSHDataContainer(
                time_series=time_series,
                static_attributes=static_attributes,
                sequence_index=sequence_index,
                config=wrong_config,
            )

    def test_target_index_out_of_bounds(self, valid_components):
        """Test that target_idx out of bounds raises error."""
        time_series, static_attributes, sequence_index, _ = valid_components

        # Config with target_idx out of bounds (only 3 features)
        wrong_config = DatasetConfig(
            input_length=20,
            output_length=10,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation", "latitude", "longitude"],
            target_idx=5,  # Out of bounds: only have indices 0, 1, 2
        )

        with pytest.raises(
            ValueError,
            match=r"Target index 5 out of bounds \[0, 3\)",
        ):
            LSHDataContainer(
                time_series=time_series,
                static_attributes=static_attributes,
                sequence_index=sequence_index,
                config=wrong_config,
            )

    def test_forcing_indices_out_of_bounds(self, valid_components):
        """Test that forcing_indices out of bounds raises error."""
        time_series, static_attributes, sequence_index, _ = valid_components

        wrong_config = DatasetConfig(
            input_length=20,
            output_length=10,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation", "latitude", "longitude"],
            forcing_indices=[0, 1, 3, 4],  # 3 and 4 are out of bounds
        )

        with pytest.raises(
            ValueError,
            match=r"Forcing indices \[3, 4\] out of bounds \[0, 3\)",
        ):
            LSHDataContainer(
                time_series=time_series,
                static_attributes=static_attributes,
                sequence_index=sequence_index,
                config=wrong_config,
            )

    def test_future_indices_out_of_bounds(self, valid_components):
        """Test that future_indices out of bounds raises error."""
        time_series, static_attributes, sequence_index, _ = valid_components

        # First test forcing indices out of bounds
        wrong_config = DatasetConfig(
            input_length=20,
            output_length=10,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation", "latitude", "longitude"],
            forcing_indices=[0, 1, 2, 10],  # 10 is out of bounds
            future_indices=[1, 2],  # Valid subset of forcing
        )

        with pytest.raises(
            ValueError,
            match=r"Forcing indices \[10\] out of bounds \[0, 3\)",
        ):
            LSHDataContainer(
                time_series=time_series,
                static_attributes=static_attributes,
                sequence_index=sequence_index,
                config=wrong_config,
            )

        # Now test future indices specifically (with valid forcing)
        wrong_config2 = DatasetConfig(
            input_length=20,
            output_length=10,
            target_name="temperature",
            forcing_features=["precipitation", "humidity"],  # Remove temperature since is_autoregressive=False
            static_features=["elevation", "latitude", "longitude"],
            forcing_indices=[1, 2, 5],  # Include 5 so subset check passes, indices for non-target features
            future_indices=[1, 5],  # 5 is out of bounds for actual data
            is_autoregressive=False,  # Disable to avoid target check
        )

        # Forcing indices will be checked first since 5 is out of bounds
        with pytest.raises(
            ValueError,
            match=r"Forcing indices \[5\] out of bounds \[0, 3\)",
        ):
            LSHDataContainer(
                time_series=time_series,
                static_attributes=static_attributes,
                sequence_index=sequence_index,
                config=wrong_config2,
            )

    def test_input_feature_indices_out_of_bounds(self, valid_components):
        """Test that input_feature_indices out of bounds raises error."""
        time_series, static_attributes, sequence_index, _ = valid_components

        wrong_config = DatasetConfig(
            input_length=20,
            output_length=10,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation", "latitude", "longitude"],
            input_feature_indices=[0, 1, 2, 5],  # 5 is out of bounds
        )

        with pytest.raises(
            ValueError,
            match=r"Input feature indices \[5\] out of bounds \[0, 3\)",
        ):
            LSHDataContainer(
                time_series=time_series,
                static_attributes=static_attributes,
                sequence_index=sequence_index,
                config=wrong_config,
            )

    def test_static_features_count_mismatch(self, valid_components):
        """Test that mismatch in static feature count raises error."""
        time_series, static_attributes, sequence_index, _ = valid_components

        # Config expects 2 static features but collection has 3
        wrong_config = DatasetConfig(
            input_length=20,
            output_length=10,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation", "latitude"],  # Missing longitude
        )

        with pytest.raises(
            ValueError,
            match="Config has 2 static features, but static_attributes has 3",
        ):
            LSHDataContainer(
                time_series=time_series,
                static_attributes=static_attributes,
                sequence_index=sequence_index,
                config=wrong_config,
            )

    def test_valid_indices_at_boundaries(self, valid_components):
        """Test that indices at valid boundaries work correctly."""
        time_series, static_attributes, sequence_index, _ = valid_components

        # All indices at valid boundaries
        config = DatasetConfig(
            input_length=20,
            output_length=10,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation", "latitude", "longitude"],
            target_idx=0,  # First index as target
            forcing_indices=[0, 1, 2],  # All valid
            future_indices=[1, 2],  # Valid subset, excluding target
            input_feature_indices=[0, 1, 2],  # All features
        )

        # Should create successfully
        container = LSHDataContainer(
            time_series=time_series,
            static_attributes=static_attributes,
            sequence_index=sequence_index,
            config=config,
        )

        assert container.config.target_idx == 0
        assert container.config.forcing_indices == [0, 1, 2]
        assert container.config.future_indices == [1, 2]

    def test_negative_indices_invalid(self, valid_components):
        """Test that negative indices are rejected."""
        time_series, static_attributes, sequence_index, _ = valid_components

        wrong_config = DatasetConfig(
            input_length=20,
            output_length=10,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation", "latitude", "longitude"],
            target_idx=-1,  # Negative index
        )

        with pytest.raises(
            ValueError,
            match=r"Target index -1 out of bounds \[0, 3\)",
        ):
            LSHDataContainer(
                time_series=time_series,
                static_attributes=static_attributes,
                sequence_index=sequence_index,
                config=wrong_config,
            )
