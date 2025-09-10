"""Tests for EA-LSTM configuration."""

import pytest

from transfer_learning_publication.models.ealstm import EALSTMConfig


class TestEALSTMConfig:
    """Test suite for EA-LSTM configuration."""

    def test_basic_initialization(self):
        """Test basic config initialization with required parameters."""
        config = EALSTMConfig(
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
        assert config.dropout == 0.0
        assert config.num_layers == 1
        assert config.bias is True
        assert config.scheduler_patience == 5
        assert config.scheduler_factor == 0.5

        # Check bidirectional defaults
        assert config.future_hidden_size == 64  # defaults to hidden_size
        assert config.future_layers == 1  # defaults to num_layers
        assert config.bidirectional_fusion == "concat"
        assert config.bidirectional is True

    def test_custom_parameters(self):
        """Test config with custom parameters."""
        config = EALSTMConfig(
            input_len=50,
            output_len=14,
            input_size=10,
            static_size=15,
            future_input_size=8,
            hidden_size=128,
            dropout=0.2,
            num_layers=3,
            bias=False,
            learning_rate=1e-4,
            scheduler_patience=10,
            scheduler_factor=0.3,
            use_rev_in=False,
            future_hidden_size=96,
            future_layers=2,
            bidirectional_fusion="add",
            bidirectional=False,
        )

        assert config.input_len == 50
        assert config.output_len == 14
        assert config.input_size == 10
        assert config.static_size == 15
        assert config.future_input_size == 8
        assert config.hidden_size == 128
        assert config.dropout == 0.2
        assert config.num_layers == 3
        assert config.bias is False
        assert config.learning_rate == 1e-4
        assert config.scheduler_patience == 10
        assert config.scheduler_factor == 0.3
        assert config.use_rev_in is False
        assert config.future_hidden_size == 96
        assert config.future_layers == 2
        assert config.bidirectional_fusion == "add"
        assert config.bidirectional is False

    def test_invalid_num_layers(self):
        """Test that num_layers must be at least 1."""
        with pytest.raises(ValueError, match="num_layers must be at least 1"):
            EALSTMConfig(
                input_len=30,
                output_len=7,
                input_size=5,
                num_layers=0,
            )

    def test_invalid_future_layers(self):
        """Test that future_layers must be at least 1."""
        with pytest.raises(ValueError, match="future_layers must be at least 1"):
            EALSTMConfig(
                input_len=30,
                output_len=7,
                input_size=5,
                future_layers=0,
            )

    def test_invalid_bidirectional_fusion(self):
        """Test that bidirectional_fusion must be valid."""
        with pytest.raises(ValueError, match="bidirectional_fusion must be one of"):
            EALSTMConfig(
                input_len=30,
                output_len=7,
                input_size=5,
                bidirectional_fusion="invalid",
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
            "num_layers": 2,
            "bidirectional": False,
        }

        config = EALSTMConfig.from_dict(config_dict)

        assert config.input_len == 40
        assert config.output_len == 10
        assert config.input_size == 7
        assert config.static_size == 5
        assert config.hidden_size == 96
        assert config.dropout == 0.15
        assert config.num_layers == 2
        assert config.bidirectional is False

        # Check defaults are still applied
        assert config.bias is True
        assert config.scheduler_patience == 5
        assert config.scheduler_factor == 0.5
        assert config.future_hidden_size == 96  # defaults to hidden_size
        assert config.future_layers == 2  # defaults to num_layers

    def test_model_params_list(self):
        """Test that MODEL_PARAMS list is correctly defined."""
        expected_params = [
            "num_layers",
            "bias",
            "dropout",
            "hidden_size",
            "future_hidden_size",
            "future_layers",
            "bidirectional_fusion",
            "bidirectional",
        ]

        assert set(EALSTMConfig.MODEL_PARAMS) == set(expected_params)

    def test_future_input_size_default(self):
        """Test default future_input_size calculation."""
        # When input_size > 1
        config1 = EALSTMConfig(input_len=30, output_len=7, input_size=5)
        assert config1.future_input_size == 4  # max(1, input_size - 1)

        # When input_size == 1
        config2 = EALSTMConfig(input_len=30, output_len=7, input_size=1)
        assert config2.future_input_size == 1  # max(1, input_size - 1)

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = EALSTMConfig(
            input_len=15,
            output_len=3,
            input_size=2,
            static_size=7,
            num_layers=2,
        )

        config_dict = config.to_dict()

        assert config_dict["input_len"] == 15
        assert config_dict["output_len"] == 3
        assert config_dict["input_size"] == 2
        assert config_dict["static_size"] == 7
        assert config_dict["num_layers"] == 2
        assert config_dict["learning_rate"] == 1e-3
        assert config_dict["use_rev_in"] is True
        assert "_" not in [k[0] for k in config_dict]  # No private attributes

    def test_bidirectional_fusion_options(self):
        """Test all valid bidirectional fusion options."""
        for fusion in ["concat", "add", "average"]:
            config = EALSTMConfig(
                input_len=30,
                output_len=7,
                input_size=5,
                bidirectional_fusion=fusion,
            )
            assert config.bidirectional_fusion == fusion
