import pytest

from transfer_learning_publication.containers.dataset_config import DatasetConfig


class TestDatasetConfig:
    def test_valid_config_creation(self):
        """Test creating a valid DatasetConfig with all basic fields."""
        config = DatasetConfig(
            input_length=30,
            output_length=7,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation", "latitude", "longitude"],
        )

        assert config.input_length == 30
        assert config.output_length == 7
        assert config.target_name == "temperature"
        assert config.forcing_features == ["temperature", "precipitation", "humidity"]
        assert config.static_features == ["elevation", "latitude", "longitude"]
        assert config.future_features == []
        assert config.is_autoregressive is True
        assert config.include_dates is False
        assert config.group_identifier_name == "entity_id"

    def test_config_with_future_features_valid(self):
        """Test that future features subset of forcing features works."""
        config = DatasetConfig(
            input_length=30,
            output_length=7,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity", "wind"],
            static_features=["elevation"],
            future_features=["precipitation", "wind"],
        )

        assert config.future_features == ["precipitation", "wind"]

    def test_config_with_future_features_invalid(self):
        """Test that future features not in forcing features raises error."""
        with pytest.raises(
            ValueError,
            match="Future features must be subset of forcing features. Invalid: {'solar_radiation'}",
        ):
            DatasetConfig(
                input_length=30,
                output_length=7,
                target_name="temperature",
                forcing_features=["temperature", "precipitation"],
                static_features=["elevation"],
                future_features=["precipitation", "solar_radiation"],
            )

    def test_config_with_indices(self):
        """Test creating config with optional index fields."""
        config = DatasetConfig(
            input_length=30,
            output_length=7,
            target_name="temperature",
            forcing_features=["temperature", "precipitation"],
            static_features=["elevation"],
            target_idx=0,
            forcing_indices=[0, 1],
            future_indices=[1],
        )

        assert config.target_idx == 0
        assert config.forcing_indices == [0, 1]
        assert config.future_indices == [1]

    def test_config_immutability(self):
        """Test that DatasetConfig is frozen (immutable)."""
        config = DatasetConfig(
            input_length=30,
            output_length=7,
            target_name="temperature",
            forcing_features=["temperature"],
            static_features=["elevation"],
        )

        with pytest.raises(AttributeError):
            config.input_length = 60

    def test_config_with_input_feature_indices(self):
        """Test creating config with input_feature_indices field."""
        config = DatasetConfig(
            input_length=30,
            output_length=7,
            target_name="temperature",
            forcing_features=["temperature", "precipitation"],
            static_features=["elevation"],
            input_feature_indices=[0, 1, 2],
        )

        assert config.input_feature_indices == [0, 1, 2]

    def test_future_indices_validation_valid(self):
        """Test that future indices subset of forcing indices works."""
        config = DatasetConfig(
            input_length=30,
            output_length=7,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation"],
            forcing_indices=[0, 1, 2],
            future_indices=[1, 2],
        )

        assert config.forcing_indices == [0, 1, 2]
        assert config.future_indices == [1, 2]

    def test_future_indices_validation_invalid(self):
        """Test that future indices not in forcing indices raises error."""
        with pytest.raises(
            ValueError,
            match="Future indices \\[1, 3\\] must be subset of forcing indices \\[0, 1, 2\\]",
        ):
            DatasetConfig(
                input_length=30,
                output_length=7,
                target_name="temperature",
                forcing_features=["temperature", "precipitation"],
                static_features=["elevation"],
                forcing_indices=[0, 1, 2],
                future_indices=[1, 3],
            )

    def test_indices_without_features(self):
        """Test that we can provide indices without corresponding feature names."""
        config = DatasetConfig(
            input_length=30,
            output_length=7,
            target_name="temperature",
            forcing_features=["temperature", "precipitation"],
            static_features=["elevation"],
            target_idx=0,
            forcing_indices=[0, 1],
            future_indices=[1],
            input_feature_indices=[0, 1],
        )

        assert config.target_idx == 0
        assert config.forcing_indices == [0, 1]
        assert config.future_indices == [1]
        assert config.input_feature_indices == [0, 1]

    def test_data_leakage_prevention_autoregressive(self):
        """Test that target index cannot be in future_indices for autoregressive models."""
        with pytest.raises(
            ValueError,
            match="Data leakage detected.*Target index 0 cannot be in future_indices",
        ):
            DatasetConfig(
                input_length=30,
                output_length=7,
                target_name="temperature",
                forcing_features=["temperature", "precipitation", "humidity"],
                static_features=["elevation"],
                target_idx=0,
                forcing_indices=[0, 1, 2],
                future_indices=[0, 1],  # INVALID: includes target index 0
                is_autoregressive=True,  # Autoregressive mode
            )

    def test_target_in_future_allowed_non_autoregressive(self):
        """Test that target in future_indices is allowed when not autoregressive."""
        # Should not raise an error
        config = DatasetConfig(
            input_length=30,
            output_length=7,
            target_name="temperature",
            forcing_features=["precipitation", "humidity"],  # Note: target NOT in forcing
            static_features=["elevation"],
            target_idx=0,
            forcing_indices=[0, 1],
            future_indices=[0, 1],  # Includes target, but OK since not autoregressive
            is_autoregressive=False,  # Non-autoregressive mode
        )
        assert config.target_idx == 0
        assert 0 in config.future_indices

    def test_target_not_in_forcing_when_non_autoregressive(self):
        """Test that target cannot be in forcing_features when is_autoregressive=False."""
        with pytest.raises(
            ValueError,
            match="Invalid configuration: Target 'streamflow' cannot be in forcing_features.*is_autoregressive=False",
        ):
            DatasetConfig(
                input_length=10,
                output_length=5,
                target_name="streamflow",
                forcing_features=["streamflow", "temp", "precip"],  # INVALID: target in forcing
                static_features=["area"],
                target_idx=0,
                forcing_indices=[0, 1, 2],
                future_indices=[1, 2],
                is_autoregressive=False,  # Non-autoregressive mode
            )
