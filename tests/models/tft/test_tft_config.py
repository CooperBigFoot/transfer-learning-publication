"""Tests for TFT configuration."""

import pytest

from transfer_learning_publication.models.tft import TFTConfig


class TestTFTConfig:
    """Test suite for TFT configuration."""

    def test_basic_initialization(self):
        """Test basic config initialization with required parameters."""
        config = TFTConfig(
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
        assert config.lstm_layers == 1
        assert config.num_attention_heads == 4
        assert config.dropout == 0.1
        assert config.hidden_continuous_size == 64  # defaults to hidden_size
        assert config.attn_dropout == 0.0
        assert config.add_relative_index is True
        assert config.scheduler_patience == 5
        assert config.scheduler_factor == 0.5
        assert config.quantiles == [0.5]
        assert config.context_length_ratio == 1.0
        assert config.encoder_layers == 1

    def test_custom_parameters(self):
        """Test config with custom parameters."""
        config = TFTConfig(
            input_len=50,
            output_len=14,
            input_size=10,
            static_size=15,
            future_input_size=8,
            hidden_size=128,
            lstm_layers=2,
            num_attention_heads=8,
            dropout=0.2,
            hidden_continuous_size=96,
            attn_dropout=0.1,
            add_relative_index=False,
            learning_rate=5e-4,
            group_identifier="basin_id",
            scheduler_patience=10,
            scheduler_factor=0.7,
            quantiles=[0.1, 0.5, 0.9],
            context_length_ratio=0.5,
            encoder_layers=3,
            use_rev_in=False,
        )

        assert config.input_len == 50
        assert config.output_len == 14
        assert config.input_size == 10
        assert config.static_size == 15
        assert config.future_input_size == 8
        assert config.hidden_size == 128
        assert config.lstm_layers == 2
        assert config.num_attention_heads == 8
        assert config.dropout == 0.2
        assert config.hidden_continuous_size == 96
        assert config.attn_dropout == 0.1
        assert config.add_relative_index is False
        assert config.learning_rate == 5e-4
        assert config.group_identifier == "basin_id"
        assert config.scheduler_patience == 10
        assert config.scheduler_factor == 0.7
        assert config.quantiles == [0.1, 0.5, 0.9]
        assert config.context_length_ratio == 0.5
        assert config.encoder_layers == 3
        assert config.use_rev_in is False

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "input_len": 40,
            "output_len": 10,
            "input_size": 7,
            "hidden_size": 96,
            "lstm_layers": 3,
            "num_attention_heads": 6,
        }

        config = TFTConfig.from_dict(config_dict)

        assert config.input_len == 40
        assert config.output_len == 10
        assert config.input_size == 7
        assert config.hidden_size == 96
        assert config.lstm_layers == 3
        assert config.num_attention_heads == 6
        # Check defaults are still applied
        assert config.dropout == 0.1
        assert config.hidden_continuous_size == 96  # defaults to hidden_size

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TFTConfig(
            input_len=25,
            output_len=5,
            input_size=3,
            hidden_size=48,
        )

        config_dict = config.to_dict()

        assert config_dict["input_len"] == 25
        assert config_dict["output_len"] == 5
        assert config_dict["input_size"] == 3
        assert config_dict["hidden_size"] == 48
        assert config_dict["lstm_layers"] == 1
        assert config_dict["num_attention_heads"] == 4
        assert "learning_rate" in config_dict
        assert "_" not in [k[0] for k in config_dict.keys()]  # No private attributes

    def test_update(self):
        """Test updating config parameters."""
        config = TFTConfig(
            input_len=30,
            output_len=7,
            input_size=5,
        )

        config.update(hidden_size=256, lstm_layers=4, learning_rate=1e-2)

        assert config.hidden_size == 256
        assert config.lstm_layers == 4
        assert config.learning_rate == 1e-2
        assert config.input_len == 30  # unchanged

    def test_invalid_lstm_layers(self):
        """Test that invalid lstm_layers raises error."""
        with pytest.raises(ValueError, match="lstm_layers must be at least 1"):
            TFTConfig(
                input_len=30,
                output_len=7,
                input_size=5,
                lstm_layers=0,
            )

    def test_invalid_attention_heads(self):
        """Test that invalid num_attention_heads raises error."""
        with pytest.raises(ValueError, match="num_attention_heads must be at least 1"):
            TFTConfig(
                input_len=30,
                output_len=7,
                input_size=5,
                num_attention_heads=0,
            )

    def test_hidden_continuous_size_default(self):
        """Test that hidden_continuous_size defaults to hidden_size."""
        config = TFTConfig(
            input_len=30,
            output_len=7,
            input_size=5,
            hidden_size=256,
        )

        assert config.hidden_continuous_size == 256

        # Test with explicit value
        config2 = TFTConfig(
            input_len=30,
            output_len=7,
            input_size=5,
            hidden_size=256,
            hidden_continuous_size=128,
        )

        assert config2.hidden_continuous_size == 128

    def test_quantiles_default(self):
        """Test that quantiles defaults to [0.5]."""
        config = TFTConfig(
            input_len=30,
            output_len=7,
            input_size=5,
        )

        assert config.quantiles == [0.5]

    def test_model_params_list(self):
        """Test that MODEL_PARAMS contains all expected parameters."""
        expected_params = [
            "hidden_size",
            "lstm_layers",
            "num_attention_heads",
            "dropout",
            "hidden_continuous_size",
            "attn_dropout",
            "add_relative_index",
            "scheduler_patience",
            "scheduler_factor",
            "quantiles",
            "context_length_ratio",
            "encoder_layers",
        ]

        assert set(TFTConfig.MODEL_PARAMS) == set(expected_params)

    def test_unknown_parameter_raises_error(self):
        """Test that unknown parameters raise ValueError."""
        with pytest.raises(ValueError, match="Unknown parameter 'invalid_param'"):
            TFTConfig(
                input_len=30,
                output_len=7,
                input_size=5,
                invalid_param=42,
            )