"""Tests for TSMixer configuration."""

import pytest

from transfer_learning_publication.models.tsmixer import TSMixerConfig


class TestTSMixerConfig:
    """Test suite for TSMixer configuration."""

    def test_basic_initialization(self):
        """Test basic config initialization with required parameters."""
        config = TSMixerConfig(
            input_len=30,
            output_len=7,
            input_size=5,
        )

        # Check standard parameters
        assert config.input_len == 30
        assert config.output_len == 7
        assert config.input_size == 5
        assert config.static_size == 0
        assert config.future_input_size == 4  # max(1, input_size - 1)
        assert config.learning_rate == 1e-3
        assert config.group_identifier == "gauge_id"
        assert config.use_rev_in is True

        # Check model-specific defaults
        assert config.hidden_size == 64
        assert config.dropout == 0.1
        assert config.static_embedding_size == 10
        assert config.num_mixing_layers == 5
        assert config.scheduler_patience == 2
        assert config.scheduler_factor == 0.5
        assert config.fusion_method == "add"

    def test_custom_parameters(self):
        """Test config with custom parameters."""
        config = TSMixerConfig(
            input_len=50,
            output_len=14,
            input_size=10,
            static_size=15,
            future_input_size=8,
            hidden_size=128,
            dropout=0.2,
            static_embedding_size=20,
            num_mixing_layers=3,
            learning_rate=1e-4,
            scheduler_patience=10,
            scheduler_factor=0.3,
            fusion_method="concat",
            use_rev_in=False,
        )

        assert config.input_len == 50
        assert config.output_len == 14
        assert config.input_size == 10
        assert config.static_size == 15
        assert config.future_input_size == 8
        assert config.hidden_size == 128
        assert config.dropout == 0.2
        assert config.static_embedding_size == 20
        assert config.num_mixing_layers == 3
        assert config.learning_rate == 1e-4
        assert config.scheduler_patience == 10
        assert config.scheduler_factor == 0.3
        assert config.fusion_method == "concat"
        assert config.use_rev_in is False

    def test_invalid_fusion_method(self):
        """Test that invalid fusion_method raises error."""
        with pytest.raises(ValueError, match="Invalid fusion_method"):
            TSMixerConfig(
                input_len=30,
                output_len=7,
                input_size=5,
                fusion_method="invalid",
            )

    def test_invalid_num_mixing_layers(self):
        """Test that num_mixing_layers must be at least 1."""
        with pytest.raises(ValueError, match="num_mixing_layers must be at least 1"):
            TSMixerConfig(
                input_len=30,
                output_len=7,
                input_size=5,
                num_mixing_layers=0,
            )

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "input_len": 40,
            "output_len": 10,
            "input_size": 7,
            "static_size": 5,
            "hidden_size": 96,
            "dropout": 0.15,
            "num_mixing_layers": 4,
            "static_embedding_size": 15,
        }

        config = TSMixerConfig.from_dict(config_dict)

        assert config.input_len == 40
        assert config.output_len == 10
        assert config.input_size == 7
        assert config.static_size == 5
        assert config.hidden_size == 96
        assert config.dropout == 0.15
        assert config.num_mixing_layers == 4
        assert config.static_embedding_size == 15

        # Check defaults are still applied
        assert config.scheduler_patience == 2
        assert config.scheduler_factor == 0.5
        assert config.fusion_method == "add"

    def test_model_params_list(self):
        """Test that MODEL_PARAMS list is correctly defined."""
        expected_params = [
            "static_embedding_size",
            "num_mixing_layers",
            "scheduler_patience",
            "scheduler_factor",
            "fusion_method",
            "hidden_size",
            "dropout",
        ]

        assert set(TSMixerConfig.MODEL_PARAMS) == set(expected_params)

    def test_future_input_size_default(self):
        """Test default future_input_size calculation."""
        # When input_size > 1
        config1 = TSMixerConfig(input_len=30, output_len=7, input_size=5)
        assert config1.future_input_size == 4  # max(1, input_size - 1)

        # When input_size == 1
        config2 = TSMixerConfig(input_len=30, output_len=7, input_size=1)
        assert config2.future_input_size == 1  # max(1, input_size - 1)

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TSMixerConfig(
            input_len=15,
            output_len=3,
            input_size=2,
            static_size=7,
            num_mixing_layers=2,
        )

        config_dict = config.to_dict()

        assert config_dict["input_len"] == 15
        assert config_dict["output_len"] == 3
        assert config_dict["input_size"] == 2
        assert config_dict["static_size"] == 7
        assert config_dict["num_mixing_layers"] == 2
        assert config_dict["learning_rate"] == 1e-3
        assert config_dict["use_rev_in"] is True
        assert "_" not in [k[0] for k in config_dict]  # No private attributes
