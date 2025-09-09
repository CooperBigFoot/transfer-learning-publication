"""Tests for EA-LSTM PyTorch Lightning module."""

import pytest
import torch

from transfer_learning_publication.contracts import Batch
from transfer_learning_publication.models.ealstm import (
    EALSTM,
    BiEALSTM,
    EALSTMConfig,
)
from transfer_learning_publication.models.ealstm.lightning import LitEALSTM


class TestLitEALSTM:
    """Test suite for LitEALSTM Lightning module."""

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
            use_rev_in=False,  # Disable for simpler testing
            bidirectional=False,  # Use standard EALSTM for testing
        )

    @pytest.fixture
    def config_bidirectional(self):
        """Create a bidirectional EA-LSTM config."""
        return EALSTMConfig(
            input_len=10,
            output_len=5,
            input_size=3,
            static_size=4,
            future_input_size=2,
            hidden_size=32,
            bidirectional=True,
            use_rev_in=False,
        )

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        batch_size = 4
        return Batch(
            X=torch.randn(batch_size, 10, 3),
            y=torch.randn(batch_size, 5),
            static=torch.randn(batch_size, 4),
            future=torch.randn(batch_size, 5, 2),
            group_identifiers=["gauge_1", "gauge_2", "gauge_3", "gauge_4"],
            input_end_dates=torch.tensor([100, 200, 300, 400]),
        )

    def test_initialization_with_config(self, config):
        """Test initialization with EALSTMConfig object."""
        model = LitEALSTM(config)

        assert isinstance(model.model, EALSTM)
        assert model.config == config
        assert hasattr(model, "criterion")
        assert hasattr(model, "test_outputs")

    def test_initialization_with_dict(self):
        """Test initialization with dictionary config."""
        config_dict = {
            "input_len": 15,
            "output_len": 7,
            "input_size": 4,
            "static_size": 5,
            "hidden_size": 64,
            "num_layers": 3,
            "bidirectional": False,  # Explicitly set to test standard EALSTM
        }

        model = LitEALSTM(config_dict)

        assert isinstance(model.model, EALSTM)
        assert model.config.input_len == 15
        assert model.config.output_len == 7
        assert model.config.input_size == 4
        assert model.config.static_size == 5
        assert model.config.hidden_size == 64
        assert model.config.num_layers == 3

    def test_bidirectional_model_creation(self, config_bidirectional):
        """Test that bidirectional config creates BiEALSTM model."""
        model = LitEALSTM(config_bidirectional)

        assert isinstance(model.model, BiEALSTM)
        assert model.config.bidirectional is True

    def test_forward_pass(self, config, sample_batch):
        """Test forward pass through the model."""
        model = LitEALSTM(config)

        output = model(sample_batch.X, sample_batch.static, sample_batch.future)

        assert output.shape == (4, 5, 1)

    def test_training_step(self, config, sample_batch):
        """Test training step."""
        model = LitEALSTM(config)

        loss = model.training_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss
        assert loss.requires_grad

    def test_validation_step(self, config, sample_batch):
        """Test validation step."""
        model = LitEALSTM(config)

        loss = model.validation_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss

    def test_test_step(self, config, sample_batch):
        """Test test step."""
        model = LitEALSTM(config)

        output = model.test_step(sample_batch, batch_idx=0)

        assert isinstance(output, dict)
        assert "predictions" in output
        assert "observations" in output
        assert "group_identifiers" in output
        assert "input_end_dates" in output

        assert output["predictions"].shape == (4, 5)
        assert output["observations"].shape == (4, 5)
        assert len(output["group_identifiers"]) == 4
        assert output["input_end_dates"].shape == (4,)

    def test_configure_optimizers(self, config):
        """Test optimizer configuration."""
        model = LitEALSTM(config)

        optimizer_config = model.configure_optimizers()

        # Should return dict with optimizer and lr_scheduler
        assert isinstance(optimizer_config, dict)
        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config

        optimizer = optimizer_config["optimizer"]
        assert isinstance(optimizer, torch.optim.Adam)

        scheduler_config = optimizer_config["lr_scheduler"]
        assert isinstance(scheduler_config, dict)
        assert "scheduler" in scheduler_config
        assert isinstance(scheduler_config["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_with_rev_in(self):
        """Test model with RevIN normalization enabled."""
        config = EALSTMConfig(
            input_len=10,
            output_len=5,
            input_size=3,
            static_size=4,
            use_rev_in=True,
        )

        model = LitEALSTM(config)

        assert model.rev_in is not None
        assert model.rev_in.num_features == 1

    def test_without_static_features(self):
        """Test model without static features."""
        config = EALSTMConfig(
            input_len=10,
            output_len=5,
            input_size=3,
            static_size=0,
            use_rev_in=False,
        )

        model = LitEALSTM(config)
        batch_size = 4

        x = torch.randn(batch_size, 10, 3)
        output = model(x, None, None)

        assert output.shape == (batch_size, 5, 1)

    def test_hyperparameters_saved(self, config):
        """Test that hyperparameters are properly saved."""
        model = LitEALSTM(config)

        hparams = model.hparams

        assert "input_len" in hparams
        assert "output_len" in hparams
        assert "input_size" in hparams
        assert "static_size" in hparams
        assert "hidden_size" in hparams
        assert "num_layers" in hparams
        assert "learning_rate" in hparams
        assert "bidirectional" in hparams

    def test_gradient_flow_training(self, config, sample_batch):
        """Test that gradients flow properly during training."""
        model = LitEALSTM(config)

        # Enable gradient computation
        model.train()

        # Forward pass
        loss = model.training_step(sample_batch, batch_idx=0)

        # Check that loss requires gradient
        assert loss.requires_grad

        # Backward pass
        loss.backward()

        # Check that at least some model parameters have gradients
        has_gradients = False
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                break
        assert has_gradients
