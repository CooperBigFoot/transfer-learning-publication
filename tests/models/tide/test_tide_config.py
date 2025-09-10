"""Tests for TiDE configuration."""

import pytest

from transfer_learning_publication.models.tide import TiDEConfig


class TestTiDEConfig:
    """Test suite for TiDE configuration."""

    def test_basic_initialization(self):
        """Test basic config initialization with required parameters."""
        config = TiDEConfig(
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
        assert config.hidden_size == 128
        assert config.dropout == 0.1
        assert config.num_encoder_layers == 1
        assert config.num_decoder_layers == 1
        assert config.decoder_output_size == 16
        assert config.temporal_decoder_hidden_size == 32
        assert config.past_feature_projection_size == 0
        assert config.future_forcing_projection_size == 0
        assert config.use_layer_norm is True
        assert config.scheduler_patience == 5
        assert config.scheduler_factor == 0.5

    def test_custom_parameters(self):
        """Test config with custom parameters."""
        config = TiDEConfig(
            input_len=50,
            output_len=14,
            input_size=10,
            static_size=15,
            future_input_size=8,
            hidden_size=256,
            dropout=0.2,
            num_encoder_layers=3,
            num_decoder_layers=2,
            decoder_output_size=32,
            temporal_decoder_hidden_size=64,
            past_feature_projection_size=16,
            future_forcing_projection_size=8,
            use_layer_norm=False,
            learning_rate=1e-4,
            scheduler_patience=10,
            scheduler_factor=0.3,
            use_rev_in=False,
        )

        assert config.input_len == 50
        assert config.output_len == 14
        assert config.input_size == 10
        assert config.static_size == 15
        assert config.future_input_size == 8
        assert config.hidden_size == 256
        assert config.dropout == 0.2
        assert config.num_encoder_layers == 3
        assert config.num_decoder_layers == 2
        assert config.decoder_output_size == 32
        assert config.temporal_decoder_hidden_size == 64
        assert config.past_feature_projection_size == 16
        assert config.future_forcing_projection_size == 8
        assert config.use_layer_norm is False
        assert config.learning_rate == 1e-4
        assert config.scheduler_patience == 10
        assert config.scheduler_factor == 0.3
        assert config.use_rev_in is False

    def test_invalid_encoder_layers(self):
        """Test that num_encoder_layers must be at least 1."""
        with pytest.raises(ValueError, match="num_encoder_layers must be at least 1"):
            TiDEConfig(
                input_len=30,
                output_len=7,
                input_size=5,
                num_encoder_layers=0,
            )

    def test_invalid_decoder_layers(self):
        """Test that num_decoder_layers must be at least 1."""
        with pytest.raises(ValueError, match="num_decoder_layers must be at least 1"):
            TiDEConfig(
                input_len=30,
                output_len=7,
                input_size=5,
                num_decoder_layers=0,
            )

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "input_len": 40,
            "output_len": 10,
            "input_size": 7,
            "static_size": 5,
            "hidden_size": 192,
            "dropout": 0.15,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
        }

        config = TiDEConfig.from_dict(config_dict)

        assert config.input_len == 40
        assert config.output_len == 10
        assert config.input_size == 7
        assert config.static_size == 5
        assert config.hidden_size == 192
        assert config.dropout == 0.15
        assert config.num_encoder_layers == 2
        assert config.num_decoder_layers == 2

        # Check defaults are still applied
        assert config.decoder_output_size == 16
        assert config.temporal_decoder_hidden_size == 32

    def test_model_params_list(self):
        """Test that MODEL_PARAMS list is correctly defined."""
        expected_params = [
            "num_encoder_layers",
            "num_decoder_layers",
            "decoder_output_size",
            "temporal_decoder_hidden_size",
            "past_feature_projection_size",
            "future_forcing_projection_size",
            "use_layer_norm",
            "hidden_size",
            "dropout",
        ]

        assert set(TiDEConfig.MODEL_PARAMS) == set(expected_params)

    def test_future_input_size_default(self):
        """Test default future_input_size calculation."""
        # When input_size > 1
        config1 = TiDEConfig(input_len=30, output_len=7, input_size=5)
        assert config1.future_input_size == 4  # input_size - 1

        # When input_size == 1
        config2 = TiDEConfig(input_len=30, output_len=7, input_size=1)
        assert config2.future_input_size == 1  # max(1, input_size - 1)
