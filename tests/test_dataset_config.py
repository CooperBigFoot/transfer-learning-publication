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