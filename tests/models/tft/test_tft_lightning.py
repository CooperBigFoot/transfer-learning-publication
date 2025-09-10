"""Tests for TFT Lightning module."""

import lightning as pl
import pytest
import torch

from transfer_learning_publication.contracts import Batch, ForecastOutput
from transfer_learning_publication.models.tft import TemporalFusionTransformer, TFTConfig
from transfer_learning_publication.models.tft.lightning import LitTFT


class TestLitTFT:
    """Test suite for LitTFT Lightning module."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        batch_size = 4
        input_len = 30
        output_len = 7
        input_size = 5
        static_size = 10
        future_size = 4

        return Batch(
            X=torch.randn(batch_size, input_len, input_size),
            y=torch.randn(batch_size, output_len),
            static=torch.randn(batch_size, static_size),
            future=torch.randn(batch_size, output_len, future_size),
            group_identifiers=["gauge_1", "gauge_2", "gauge_3", "gauge_4"],
            input_end_dates=torch.tensor([100, 200, 300, 400]),
        )

    @pytest.fixture
    def tft_config(self):
        """Create a TFT config for testing."""
        return TFTConfig(
            input_len=30,
            output_len=7,
            input_size=5,
            static_size=10,
            future_input_size=4,
            hidden_size=32,
            lstm_layers=1,
            num_attention_heads=2,
            dropout=0.1,
            use_rev_in=False,  # Disable for simpler testing
        )

    @pytest.fixture
    def tft_model(self, tft_config):
        """Create a LitTFT model for testing."""
        return LitTFT(tft_config)

    def test_initialization_with_config(self, tft_config):
        """Test model initialization with TFTConfig."""
        model = LitTFT(tft_config)

        assert isinstance(model.model, TemporalFusionTransformer)
        assert model.config.input_len == 30
        assert model.config.output_len == 7
        assert model.config.hidden_size == 32
        assert model.rev_in is None  # use_rev_in=False

    def test_initialization_with_dict(self):
        """Test model initialization with dict config."""
        config_dict = {
            "input_len": 25,
            "output_len": 5,
            "input_size": 4,
            "static_size": 8,
            "hidden_size": 16,
            "lstm_layers": 1,
            "num_attention_heads": 2,
        }

        model = LitTFT(config_dict)

        assert isinstance(model.model, TemporalFusionTransformer)
        assert model.config.input_len == 25
        assert model.config.output_len == 5
        assert model.config.input_size == 4
        assert model.config.hidden_size == 16

    def test_forward_pass(self, tft_model, sample_batch):
        """Test forward pass through the model."""
        output = tft_model(sample_batch.X, sample_batch.static, sample_batch.future)

        assert output.shape == (4, 7, 1)  # [batch_size, output_len, 1]

    def test_training_step(self, tft_model, sample_batch):
        """Test training step."""
        loss = tft_model.training_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss
        assert loss.requires_grad

    def test_validation_step(self, tft_model, sample_batch):
        """Test validation step."""
        loss = tft_model.validation_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss

    def test_test_step(self, tft_model, sample_batch):
        """Test test step."""
        output = tft_model.test_step(sample_batch, batch_idx=0)

        assert isinstance(output, dict)
        assert "predictions" in output
        assert "observations" in output
        assert "group_identifiers" in output
        assert "input_end_dates" in output

        assert output["predictions"].shape == (4, 7)  # Squeezed last dimension
        assert output["observations"].shape == (4, 7)
        assert len(output["group_identifiers"]) == 4
        assert output["input_end_dates"].shape == (4,)

    def test_test_epoch_end(self, tft_model, sample_batch):
        """Test consolidation of test outputs."""
        # Simulate multiple test steps
        tft_model.on_test_epoch_start()

        for i in range(3):
            tft_model.test_step(sample_batch, batch_idx=i)

        forecast_output = tft_model.on_test_epoch_end()

        assert isinstance(forecast_output, ForecastOutput)
        assert forecast_output.predictions.shape == (12, 7)  # 3 batches * 4 samples
        assert forecast_output.observations.shape == (12, 7)
        assert len(forecast_output.group_identifiers) == 12
        assert forecast_output.input_end_dates.shape == (12,)

        # Check that test_outputs is cleared
        assert tft_model.test_outputs == []

    def test_configure_optimizers(self, tft_model):
        """Test optimizer configuration."""
        optimizer_config = tft_model.configure_optimizers()

        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config

        optimizer = optimizer_config["optimizer"]
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == tft_model.config.learning_rate

        scheduler_config = optimizer_config["lr_scheduler"]
        assert "scheduler" in scheduler_config
        assert scheduler_config["monitor"] == "val_loss"

    def test_with_rev_in(self):
        """Test model with RevIN normalization enabled."""
        config = TFTConfig(
            input_len=20,
            output_len=5,
            input_size=3,
            hidden_size=16,
            use_rev_in=True,
        )
        model = LitTFT(config)

        assert model.rev_in is not None
        assert model.rev_in.num_features == 1

        # Test forward pass with RevIN
        batch_size = 2
        x = torch.randn(batch_size, config.input_len, config.input_size)
        output = model(x, None, None)

        assert output.shape == (batch_size, config.output_len, 1)

    def test_without_static_features(self):
        """Test model without static features."""
        config = TFTConfig(
            input_len=20,
            output_len=5,
            input_size=3,
            static_size=0,
            hidden_size=16,
            use_rev_in=False,  # Disable RevIN for simpler testing
        )
        model = LitTFT(config)

        batch_size = 2
        # Use empty tensor for static features
        batch = Batch(
            X=torch.randn(batch_size, config.input_len, config.input_size),
            y=torch.randn(batch_size, config.output_len),
            static=torch.zeros(batch_size, 0),  # Empty tensor for no static features
            future=torch.randn(batch_size, config.output_len, config.future_input_size),
            group_identifiers=["gauge_1", "gauge_2"],
            input_end_dates=torch.tensor([100, 200]),
        )

        # Test forward pass without static (passing None to the model)
        output = model(batch.X, None, batch.future)
        assert output.shape == (batch_size, config.output_len, 1)

    def test_without_future_features(self):
        """Test model without future features."""
        config = TFTConfig(
            input_len=20,
            output_len=5,
            input_size=3,
            static_size=5,
            future_input_size=0,
            hidden_size=16,
            use_rev_in=False,  # Disable RevIN for simpler testing
        )
        model = LitTFT(config)

        batch_size = 2
        # Use empty tensor for future features
        batch = Batch(
            X=torch.randn(batch_size, config.input_len, config.input_size),
            y=torch.randn(batch_size, config.output_len),
            static=torch.randn(batch_size, config.static_size),
            future=torch.zeros(batch_size, config.output_len, 0),  # Empty tensor for no future features
            group_identifiers=["gauge_1", "gauge_2"],
            input_end_dates=torch.tensor([100, 200]),
        )

        # Test forward pass without future (passing None to the model)
        output = model(batch.X, batch.static, None)
        assert output.shape == (batch_size, config.output_len, 1)

    def test_hyperparameters_saved(self, tft_model):
        """Test that hyperparameters are saved."""
        hparams = tft_model.hparams

        assert "input_len" in hparams
        assert "output_len" in hparams
        assert "input_size" in hparams
        assert "hidden_size" in hparams
        assert "lstm_layers" in hparams
        assert "num_attention_heads" in hparams
        assert "learning_rate" in hparams

    def test_integration_with_trainer(self, tft_config, sample_batch):
        """Test integration with PyTorch Lightning Trainer."""
        model = LitTFT(tft_config)

        # Create a minimal trainer for testing
        trainer = pl.Trainer(
            max_epochs=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            accelerator="cpu",
            num_sanity_val_steps=0,  # Skip validation sanity check
        )

        # Create simple DataLoaders
        train_dataloader = torch.utils.data.DataLoader(
            [sample_batch],
            batch_size=None,
            batch_sampler=None,
        )

        val_dataloader = torch.utils.data.DataLoader(
            [sample_batch],
            batch_size=None,
            batch_sampler=None,
        )

        # Test that training doesn't crash
        trainer.fit(model, train_dataloader, val_dataloader)

        # Model should have been trained (even if minimally)
        assert trainer.current_epoch == 1
