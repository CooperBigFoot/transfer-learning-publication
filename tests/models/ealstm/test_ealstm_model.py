"""Tests for EA-LSTM model implementation."""

import pytest
import torch

from transfer_learning_publication.models.ealstm import (
    EALSTM,
    BiEALSTM,
    EALSTMCell,
    EALSTMConfig,
)


class TestEALSTMCell:
    """Test suite for EA-LSTM cell."""

    @pytest.fixture
    def cell_config(self):
        """Create a simple EA-LSTM cell for testing."""
        return {
            "input_size": 3,
            "hidden_size": 16,
            "static_size": 5,
            "bias": True,
        }

    def test_cell_initialization(self, cell_config):
        """Test EA-LSTM cell initialization."""
        cell = EALSTMCell(**cell_config)

        assert cell.input_size == 3
        assert cell.hidden_size == 16
        assert cell.static_size == 5
        assert cell.bias is True

        # Check that required layers exist
        assert hasattr(cell, "weight_sh")
        assert hasattr(cell, "dynamic_to_gates")
        assert hasattr(cell, "hidden_to_gates")

    def test_cell_forward_with_static(self, cell_config):
        """Test EA-LSTM cell forward pass with static features."""
        cell = EALSTMCell(**cell_config)
        batch_size = 4

        dynamic_x = torch.randn(batch_size, cell_config["input_size"])
        static_x = torch.randn(batch_size, cell_config["static_size"])

        h, c = cell(dynamic_x, static_x)

        assert h.shape == (batch_size, cell_config["hidden_size"])
        assert c.shape == (batch_size, cell_config["hidden_size"])

    def test_cell_forward_without_static(self):
        """Test EA-LSTM cell forward pass without static features."""
        cell = EALSTMCell(
            input_size=3,
            hidden_size=16,
            static_size=0,
            bias=True,
        )
        batch_size = 4

        dynamic_x = torch.randn(batch_size, 3)

        h, c = cell(dynamic_x, None)

        assert h.shape == (batch_size, 16)
        assert c.shape == (batch_size, 16)

    def test_cell_with_initial_hidden_state(self, cell_config):
        """Test EA-LSTM cell with provided initial hidden state."""
        cell = EALSTMCell(**cell_config)
        batch_size = 4

        dynamic_x = torch.randn(batch_size, cell_config["input_size"])
        static_x = torch.randn(batch_size, cell_config["static_size"])
        h_0 = torch.randn(batch_size, cell_config["hidden_size"])
        c_0 = torch.randn(batch_size, cell_config["hidden_size"])

        h, c = cell(dynamic_x, static_x, (h_0, c_0))

        assert h.shape == (batch_size, cell_config["hidden_size"])
        assert c.shape == (batch_size, cell_config["hidden_size"])


class TestEALSTM:
    """Test suite for EA-LSTM model."""

    @pytest.fixture
    def config(self):
        """Create a sample EA-LSTM config."""
        return EALSTMConfig(
            input_len=10,
            output_len=5,
            input_size=3,
            static_size=4,
            hidden_size=32,
            num_layers=2,
            dropout=0.1,
        )

    @pytest.fixture
    def config_no_static(self):
        """Create EA-LSTM config without static features."""
        return EALSTMConfig(
            input_len=10,
            output_len=5,
            input_size=3,
            static_size=0,
            hidden_size=32,
        )

    def test_model_initialization(self, config):
        """Test EA-LSTM model initialization."""
        model = EALSTM(config)

        assert len(model.ealstm_cells) == config.num_layers
        assert model.config == config
        assert hasattr(model, "projection")
        assert hasattr(model, "future_forcing_layer")

    def test_forward_pass_with_static(self, config):
        """Test forward pass with static features."""
        model = EALSTM(config)
        batch_size = 4

        x = torch.randn(batch_size, config.input_len, config.input_size)
        static = torch.randn(batch_size, config.static_size)

        output = model(x, static)

        assert output.shape == (batch_size, config.output_len, 1)

    def test_forward_pass_without_static(self, config_no_static):
        """Test forward pass without static features."""
        model = EALSTM(config_no_static)
        batch_size = 4

        x = torch.randn(batch_size, config_no_static.input_len, config_no_static.input_size)

        output = model(x, None)

        assert output.shape == (batch_size, config_no_static.output_len, 1)

    def test_forward_with_future_forcing(self, config):
        """Test forward pass with future forcing data."""
        model = EALSTM(config)
        batch_size = 4

        x = torch.randn(batch_size, config.input_len, config.input_size)
        static = torch.randn(batch_size, config.static_size)
        future = torch.randn(batch_size, config.output_len, config.future_input_size)

        output = model(x, static, future)

        assert output.shape == (batch_size, config.output_len, 1)

    def test_return_hidden_state(self, config):
        """Test returning hidden state instead of predictions."""
        model = EALSTM(config)
        batch_size = 4

        x = torch.randn(batch_size, config.input_len, config.input_size)
        static = torch.randn(batch_size, config.static_size)

        hidden = model(x, static, return_hidden=True)

        assert hidden.shape == (batch_size, config.hidden_size)

    def test_missing_static_features_error(self, config):
        """Test that missing required static features raises error."""
        model = EALSTM(config)
        batch_size = 4

        x = torch.randn(batch_size, config.input_len, config.input_size)

        with pytest.raises(ValueError, match="Model expects static features"):
            model(x, None)

    def test_single_layer_model(self):
        """Test EA-LSTM with single layer."""
        config = EALSTMConfig(
            input_len=10,
            output_len=5,
            input_size=3,
            static_size=2,
            hidden_size=16,
            num_layers=1,
        )
        model = EALSTM(config)
        batch_size = 2

        x = torch.randn(batch_size, config.input_len, config.input_size)
        static = torch.randn(batch_size, config.static_size)

        output = model(x, static)

        assert output.shape == (batch_size, config.output_len, 1)
        assert len(model.ealstm_cells) == 1
        assert model.hidden_to_input_projections is None  # No projections for single layer


class TestBiEALSTM:
    """Test suite for Bidirectional EA-LSTM model."""

    @pytest.fixture
    def config(self):
        """Create a sample Bidirectional EA-LSTM config."""
        return EALSTMConfig(
            input_len=10,
            output_len=5,
            input_size=3,
            static_size=4,
            future_input_size=2,
            hidden_size=32,
            num_layers=2,
            bidirectional=True,
            bidirectional_fusion="concat",
        )

    def test_model_initialization(self, config):
        """Test Bidirectional EA-LSTM model initialization."""
        model = BiEALSTM(config)

        assert hasattr(model, "past_ealstm")
        assert hasattr(model, "future_ealstm")
        assert hasattr(model, "projection")
        assert model.fusion_method == "concat"

    def test_forward_pass_with_future(self, config):
        """Test forward pass with future data."""
        model = BiEALSTM(config)
        batch_size = 4

        x = torch.randn(batch_size, config.input_len, config.input_size)
        static = torch.randn(batch_size, config.static_size)
        future = torch.randn(batch_size, config.output_len, config.future_input_size)

        output = model(x, static, future)

        assert output.shape == (batch_size, config.output_len, 1)

    def test_forward_pass_without_future(self, config):
        """Test forward pass without future data (fallback to standard EA-LSTM)."""
        model = BiEALSTM(config)
        batch_size = 4

        x = torch.randn(batch_size, config.input_len, config.input_size)
        static = torch.randn(batch_size, config.static_size)

        output = model(x, static, None)

        assert output.shape == (batch_size, config.output_len, 1)

    def test_fusion_methods(self):
        """Test different fusion methods for bidirectional model."""
        for fusion in ["concat", "add", "average"]:
            config = EALSTMConfig(
                input_len=10,
                output_len=5,
                input_size=3,
                static_size=4,
                future_input_size=2,
                hidden_size=32,
                bidirectional=True,
                bidirectional_fusion=fusion,
            )
            model = BiEALSTM(config)
            batch_size = 2

            x = torch.randn(batch_size, config.input_len, config.input_size)
            static = torch.randn(batch_size, config.static_size)
            future = torch.randn(batch_size, config.output_len, config.future_input_size)

            output = model(x, static, future)

            assert output.shape == (batch_size, config.output_len, 1)

    def test_missing_static_features_error(self, config):
        """Test that missing required static features raises error."""
        model = BiEALSTM(config)
        batch_size = 4

        x = torch.randn(batch_size, config.input_len, config.input_size)

        with pytest.raises(ValueError, match="Model expects static features"):
            model(x, None)

    def test_gradient_flow(self, config):
        """Test that gradients flow through the model."""
        model = BiEALSTM(config)
        batch_size = 2

        x = torch.randn(batch_size, config.input_len, config.input_size, requires_grad=True)
        static = torch.randn(batch_size, config.static_size)
        future = torch.randn(batch_size, config.output_len, config.future_input_size)

        output = model(x, static, future)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
