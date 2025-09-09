"""Integration tests for TiDE Lightning module."""

import torch

from transfer_learning_publication.contracts.batch import Batch
from transfer_learning_publication.models.tide import LitTiDE, TiDEConfig


class TestLitTiDE:
    """Test suite for LitTiDE Lightning module."""

    def test_initialization_with_config(self):
        """Test initialization with TiDEConfig object."""
        config = TiDEConfig(
            input_len=30,
            output_len=7,
            input_size=5,
        )

        lit_model = LitTiDE(config)

        assert lit_model.config == config
        assert lit_model.model is not None
        assert hasattr(lit_model, "model")

    def test_initialization_with_dict(self):
        """Test initialization with dictionary config."""
        config_dict = {
            "input_len": 30,
            "output_len": 7,
            "input_size": 5,
            "hidden_size": 64,
            "dropout": 0.2,
        }

        lit_model = LitTiDE(config_dict)

        assert lit_model.config.input_len == 30
        assert lit_model.config.output_len == 7
        assert lit_model.config.input_size == 5
        assert lit_model.config.hidden_size == 64
        assert lit_model.config.dropout == 0.2

    def test_forward_delegation(self):
        """Test that forward method delegates to the underlying model."""
        config = TiDEConfig(
            input_len=30,
            output_len=7,
            input_size=3,
            static_size=4,
            future_input_size=2,
        )

        lit_model = LitTiDE(config)

        batch_size = 8
        x = torch.randn(batch_size, config.input_len, config.input_size)
        static = torch.randn(batch_size, config.static_size)
        future = torch.randn(batch_size, config.output_len, config.future_input_size)

        # Test forward with all inputs
        output = lit_model(x, static=static, future=future)
        assert output.shape == (batch_size, config.output_len, 1)

        # Test forward with only x
        output = lit_model(x)
        assert output.shape == (batch_size, config.output_len, 1)

    def test_training_step(self):
        """Test training step functionality."""
        config = TiDEConfig(
            input_len=10,
            output_len=5,
            input_size=3,
        )

        lit_model = LitTiDE(config)

        batch_size = 4
        batch = Batch(
            X=torch.randn(batch_size, config.input_len, config.input_size),
            y=torch.randn(batch_size, config.output_len),
            static=torch.zeros(batch_size, 0),  # Empty static features
            future=torch.zeros(batch_size, config.output_len, 0),  # Empty future features
            group_identifiers=["gauge_1"] * batch_size,
        )

        # Run training step
        loss = lit_model.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar
        assert loss.requires_grad

    def test_validation_step(self):
        """Test validation step functionality."""
        config = TiDEConfig(
            input_len=10,
            output_len=5,
            input_size=3,
            static_size=2,
        )

        lit_model = LitTiDE(config)

        batch_size = 4
        batch = Batch(
            X=torch.randn(batch_size, config.input_len, config.input_size),
            y=torch.randn(batch_size, config.output_len),
            static=torch.randn(batch_size, config.static_size),
            future=torch.zeros(batch_size, config.output_len, 0),  # Empty future features
            group_identifiers=["gauge_1", "gauge_2", "gauge_1", "gauge_2"],
        )

        # Run validation step
        loss = lit_model.validation_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar

    def test_test_step(self):
        """Test test step functionality."""
        config = TiDEConfig(
            input_len=10,
            output_len=5,
            input_size=3,
            future_input_size=2,
        )

        lit_model = LitTiDE(config)

        batch_size = 4
        batch = Batch(
            X=torch.randn(batch_size, config.input_len, config.input_size),
            y=torch.randn(batch_size, config.output_len),
            static=torch.zeros(batch_size, 0),  # Empty static features
            future=torch.randn(batch_size, config.output_len, config.future_input_size),
            group_identifiers=["gauge_1", "gauge_2", "gauge_3", "gauge_4"],
        )

        # Run test step
        outputs = lit_model.test_step(batch, batch_idx=0)

        # Check that outputs are collected
        assert isinstance(outputs, dict)
        assert "predictions" in outputs
        assert "observations" in outputs  # BaseLitModel uses "observations" not "targets"
        assert "group_identifiers" in outputs

        # Check shapes
        assert outputs["predictions"].shape == (batch_size, config.output_len)
        assert outputs["observations"].shape == (batch_size, config.output_len)
        assert len(outputs["group_identifiers"]) == batch_size

    def test_optimizer_configuration(self):
        """Test optimizer configuration."""
        config = TiDEConfig(
            input_len=10,
            output_len=5,
            input_size=3,
            learning_rate=1e-3,
        )

        lit_model = LitTiDE(config)

        optimizer_config = lit_model.configure_optimizers()

        # Check optimizer
        assert "optimizer" in optimizer_config
        optimizer = optimizer_config["optimizer"]
        assert optimizer.__class__.__name__ == "Adam"
        assert optimizer.param_groups[0]["lr"] == 1e-3

        # Check scheduler if present
        if "lr_scheduler" in optimizer_config:
            scheduler_config = optimizer_config["lr_scheduler"]
            assert "scheduler" in scheduler_config
            assert scheduler_config["monitor"] == "val_loss"

    def test_with_revin_normalization(self):
        """Test model with RevIN normalization enabled."""
        config = TiDEConfig(
            input_len=10,
            output_len=5,
            input_size=3,
            use_rev_in=True,
        )

        lit_model = LitTiDE(config)

        # Check that RevIN layer is created (it's called rev_in in BaseLitModel)
        assert hasattr(lit_model, "rev_in")
        assert lit_model.config.use_rev_in is True

        batch_size = 4
        x = torch.randn(batch_size, config.input_len, config.input_size)

        # Forward pass should work with RevIN
        output = lit_model(x)
        assert output.shape == (batch_size, config.output_len, 1)

    def test_without_revin_normalization(self):
        """Test model with RevIN normalization disabled."""
        config = TiDEConfig(
            input_len=10,
            output_len=5,
            input_size=3,
            use_rev_in=False,
        )

        lit_model = LitTiDE(config)

        # Check that RevIN is disabled
        assert lit_model.config.use_rev_in is False

        batch_size = 4
        x = torch.randn(batch_size, config.input_len, config.input_size)

        # Forward pass should work without RevIN
        output = lit_model(x)
        assert output.shape == (batch_size, config.output_len, 1)

    def test_batch_independence(self):
        """Test that batch samples are processed independently."""
        config = TiDEConfig(
            input_len=10,
            output_len=5,
            input_size=3,
        )

        lit_model = LitTiDE(config)
        lit_model.eval()

        # Create two different inputs
        x1 = torch.randn(1, config.input_len, config.input_size)
        x2 = torch.randn(1, config.input_len, config.input_size)

        # Process separately
        with torch.no_grad():
            y1 = lit_model(x1)
            y2 = lit_model(x2)

            # Process together
            x_batch = torch.cat([x1, x2], dim=0)
            y_batch = lit_model(x_batch)

        # Results should be the same whether processed separately or together
        assert torch.allclose(y1, y_batch[0:1], atol=1e-6)
        assert torch.allclose(y2, y_batch[1:2], atol=1e-6)
