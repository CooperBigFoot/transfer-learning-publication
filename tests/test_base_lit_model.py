"""Tests for BaseLitModel."""

import pytest
import torch

from transfer_learning_publication.contracts import Batch, ForecastOutput
from transfer_learning_publication.models.base import BaseLitModel
from transfer_learning_publication.models.dummy import NaiveLastValueConfig
from transfer_learning_publication.models.dummy.lightning import LitNaiveLastValue


class TestBaseLitModel:
    """Test suite for BaseLitModel functionality."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        batch_size = 4
        input_len = 10
        output_len = 5
        input_size = 3
        static_size = 2
        future_size = 2

        return Batch(
            X=torch.randn(batch_size, input_len, input_size),
            y=torch.randn(batch_size, output_len),
            static=torch.randn(batch_size, static_size),
            future=torch.randn(batch_size, output_len, future_size),
            group_identifiers=["gauge_1", "gauge_2", "gauge_3", "gauge_4"],
            input_end_dates=torch.tensor([100, 200, 300, 400]),
        )

    @pytest.fixture
    def naive_model(self):
        """Create a NaiveLastValue model for testing."""
        config = NaiveLastValueConfig(
            input_len=10,
            output_len=5,
            input_size=3,
            static_size=2,
            future_input_size=2,
            use_rev_in=False,
        )
        return LitNaiveLastValue(config)

    def test_initialization(self, naive_model):
        """Test model initialization."""
        assert isinstance(naive_model, BaseLitModel)
        assert isinstance(naive_model.config, NaiveLastValueConfig)
        assert naive_model.rev_in is None  # use_rev_in=False
        assert isinstance(naive_model.criterion, torch.nn.MSELoss)
        assert naive_model.test_outputs == []
        assert naive_model._forecast_output is None

    def test_initialization_with_rev_in(self):
        """Test model initialization with RevIN enabled."""
        config = NaiveLastValueConfig(
            input_len=10,
            output_len=5,
            input_size=3,
            use_rev_in=True,
        )
        model = LitNaiveLastValue(config)

        assert model.rev_in is not None
        assert model.rev_in.num_features == 1

    def test_forward_pass(self, naive_model, sample_batch):
        """Test forward pass through the model."""
        output = naive_model(sample_batch.X, sample_batch.static, sample_batch.future)

        assert output.shape == (4, 5, 1)  # [batch_size, output_len, 1]

    def test_training_step(self, naive_model, sample_batch):
        """Test training step."""
        loss = naive_model.training_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss
        # Note: loss.requires_grad may be False due to no gradients in naive model

    def test_validation_step(self, naive_model, sample_batch):
        """Test validation step."""
        loss = naive_model.validation_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss

    def test_test_step(self, naive_model, sample_batch):
        """Test test step."""
        output = naive_model.test_step(sample_batch, batch_idx=0)

        assert isinstance(output, dict)
        assert "predictions" in output
        assert "observations" in output
        assert "group_identifiers" in output
        assert "input_end_dates" in output

        assert output["predictions"].shape == (4, 5)  # Squeezed last dimension
        assert output["observations"].shape == (4, 5)
        assert len(output["group_identifiers"]) == 4
        assert output["input_end_dates"].shape == (4,)

    def test_test_epoch_end(self, naive_model, sample_batch):
        """Test consolidation of test outputs."""
        # Simulate multiple test steps
        naive_model.on_test_epoch_start()

        for i in range(3):
            naive_model.test_step(sample_batch, batch_idx=i)

        forecast_output = naive_model.on_test_epoch_end()

        assert isinstance(forecast_output, ForecastOutput)
        assert forecast_output.predictions.shape == (12, 5)  # 3 batches * 4 samples
        assert forecast_output.observations.shape == (12, 5)
        assert len(forecast_output.group_identifiers) == 12
        assert forecast_output.input_end_dates.shape == (12,)

        # Check that test_outputs is cleared
        assert naive_model.test_outputs == []

        # Check that forecast_output is stored internally
        assert naive_model._forecast_output is not None
        assert naive_model._forecast_output is forecast_output

    def test_test_epoch_end_no_outputs(self, naive_model):
        """Test error when no test outputs collected."""
        naive_model.on_test_epoch_start()

        with pytest.raises(RuntimeError, match="No test outputs collected"):
            naive_model.on_test_epoch_end()

    def test_configure_optimizers(self, naive_model):
        """Test optimizer configuration."""
        optimizer_config = naive_model.configure_optimizers()

        # NaiveLastValue overrides configure_optimizers to return just optimizer
        assert isinstance(optimizer_config, torch.optim.Adam)

    def test_batch_validation(self, naive_model):
        """Test that invalid batch types are rejected."""
        invalid_batch = {"X": torch.randn(2, 10, 3)}  # Dict instead of Batch

        with pytest.raises(TypeError, match="Expected Batch instance"):
            naive_model.training_step(invalid_batch, batch_idx=0)

    def test_dict_config_initialization(self):
        """Test model initialization with dict config."""
        config_dict = {
            "input_len": 15,
            "output_len": 7,
            "input_size": 4,
            "static_size": 3,
            "learning_rate": 1e-4,
        }

        model = LitNaiveLastValue(config_dict)

        assert model.config.input_len == 15
        assert model.config.output_len == 7
        assert model.config.input_size == 4
        # NaiveLastValue uses the passed learning_rate since it's in the dict
        assert model.config.learning_rate == 1e-4

    def test_hyperparameters_saved(self, naive_model):
        """Test that hyperparameters are saved."""
        hparams = naive_model.hparams

        assert "input_len" in hparams
        assert "output_len" in hparams
        assert "input_size" in hparams
        assert "learning_rate" in hparams

    def test_forecast_output_property(self, naive_model, sample_batch):
        """Test forecast_output property access."""
        # Should raise error before testing
        with pytest.raises(RuntimeError, match="No forecast output available"):
            _ = naive_model.forecast_output

        # Run test epoch
        naive_model.on_test_epoch_start()
        for i in range(2):
            naive_model.test_step(sample_batch, batch_idx=i)
        result = naive_model.on_test_epoch_end()

        # Now property should work
        stored_output = naive_model.forecast_output
        assert stored_output is result
        assert isinstance(stored_output, ForecastOutput)
        assert stored_output.predictions.shape == (8, 5)  # 2 batches * 4 samples

    def test_forecast_output_reset_on_new_test(self, naive_model, sample_batch):
        """Test that forecast_output is reset on new test epoch."""
        # Run first test
        naive_model.on_test_epoch_start()
        naive_model.test_step(sample_batch, batch_idx=0)
        first_output = naive_model.on_test_epoch_end()

        # Start new test - should reset
        naive_model.on_test_epoch_start()
        assert naive_model._forecast_output is None

        # Run second test
        naive_model.test_step(sample_batch, batch_idx=0)
        second_output = naive_model.on_test_epoch_end()

        # Should have new output
        assert naive_model.forecast_output is second_output
        assert naive_model.forecast_output is not first_output

    def test_rev_in_normalization(self):
        """Test RevIN normalization and denormalization."""
        config = NaiveLastValueConfig(
            input_len=10,
            output_len=5,
            input_size=3,
            use_rev_in=True,
        )
        model = LitNaiveLastValue(config)

        # Create input with known values
        batch_size = 2
        input_len = 10
        input_size = 3

        # Create input where target (first feature) has mean=5, std=2
        x = torch.zeros(batch_size, input_len, input_size)
        x[:, :, 0] = torch.randn(batch_size, input_len) * 2 + 5

        # Apply normalization
        x_normalized = model._apply_rev_in_normalization(x)

        # Check that only first feature was normalized
        assert x_normalized.shape == x.shape
        assert not torch.equal(x_normalized[:, :, 0], x[:, :, 0])  # Changed
        if input_size > 1:
            assert torch.equal(x_normalized[:, :, 1:], x[:, :, 1:])  # Unchanged

        # Test denormalization
        y_hat = torch.randn(batch_size, 5, 1)
        y_denorm = model._apply_rev_in_denormalization(y_hat)

        assert y_denorm.shape == y_hat.shape
