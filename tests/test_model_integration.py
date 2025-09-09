"""Integration tests for the model framework."""

import pytest
import torch

from transfer_learning_publication.contracts import Batch, ForecastOutput
from transfer_learning_publication.models.dummy import (
    NaiveLastValue,
    NaiveLastValueConfig,
    LitNaiveLastValue,
)


class TestNaiveLastValueIntegration:
    """Integration tests for NaiveLastValue model."""
    
    @pytest.fixture
    def config(self):
        """Create model configuration."""
        return NaiveLastValueConfig(
            input_len=30,
            output_len=7,
            input_size=5,
            static_size=10,
            future_input_size=3,
            use_rev_in=False,
        )
        
    @pytest.fixture
    def batch(self, config):
        """Create a realistic batch."""
        batch_size = 8
        
        return Batch(
            X=torch.randn(batch_size, config.input_len, config.input_size),
            y=torch.randn(batch_size, config.output_len),
            static=torch.randn(batch_size, config.static_size),
            future=torch.randn(batch_size, config.output_len, config.future_input_size),
            group_identifiers=[f"gauge_{i:03d}" for i in range(batch_size)],
            input_end_dates=torch.arange(batch_size) * 100,
        )
        
    def test_core_model_forward(self, config, batch):
        """Test forward pass through core model."""
        model = NaiveLastValue(config)
        
        output = model(batch.X, batch.static, batch.future)
        
        # Check output shape
        assert output.shape == (batch.batch_size, config.output_len, 1)
        
        # Check that output repeats last value
        last_values = batch.X[:, -1, 0:1].unsqueeze(1)  # [batch, 1, 1]
        expected = last_values.repeat(1, config.output_len, 1)
        
        assert torch.allclose(output, expected)
        
    def test_lightning_model_forward(self, config, batch):
        """Test forward pass through Lightning model."""
        model = LitNaiveLastValue(config)
        
        output = model(batch.X, batch.static, batch.future)
        
        assert output.shape == (batch.batch_size, config.output_len, 1)
        
    def test_training_step_integration(self, config, batch):
        """Test complete training step."""
        model = LitNaiveLastValue(config)
        
        # Run training step
        loss = model.training_step(batch, batch_idx=0)
        
        # Verify loss
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0  # Should have some loss
        
    def test_validation_step_integration(self, config, batch):
        """Test complete validation step."""
        model = LitNaiveLastValue(config)
        
        # Run validation step
        loss = model.validation_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        
    def test_test_step_integration(self, config, batch):
        """Test complete test step."""
        model = LitNaiveLastValue(config)
        
        # Initialize test outputs
        model.on_test_epoch_start()
        
        # Run test step
        output = model.test_step(batch, batch_idx=0)
        
        # Verify output structure
        assert "predictions" in output
        assert "observations" in output
        assert "group_identifiers" in output
        assert "input_end_dates" in output
        
        # Verify shapes
        assert output["predictions"].shape == (batch.batch_size, config.output_len)
        assert output["observations"].shape == (batch.batch_size, config.output_len)
        
    def test_full_test_epoch(self, config):
        """Test full test epoch with multiple batches."""
        model = LitNaiveLastValue(config)
        
        # Create multiple batches
        batches = []
        for _ in range(3):
            batch = Batch(
                X=torch.randn(4, config.input_len, config.input_size),
                y=torch.randn(4, config.output_len),
                static=torch.randn(4, config.static_size),
                future=torch.randn(4, config.output_len, config.future_input_size),
                group_identifiers=[f"gauge_{i}" for i in range(4)],
                input_end_dates=torch.arange(4),
            )
            batches.append(batch)
            
        # Run test epoch
        model.on_test_epoch_start()
        
        for i, batch in enumerate(batches):
            model.test_step(batch, batch_idx=i)
            
        # Get consolidated output
        forecast_output = model.on_test_epoch_end()
        
        # Verify ForecastOutput
        assert isinstance(forecast_output, ForecastOutput)
        assert forecast_output.predictions.shape == (12, config.output_len)  # 3 batches * 4 samples
        assert forecast_output.observations.shape == (12, config.output_len)
        assert len(forecast_output.group_identifiers) == 12
        
    def test_with_rev_in_enabled(self):
        """Test model with RevIN normalization enabled."""
        config = NaiveLastValueConfig(
            input_len=20,
            output_len=5,
            input_size=3,
            use_rev_in=True,
        )
        
        model = LitNaiveLastValue(config)
        
        batch = Batch(
            X=torch.randn(4, 20, 3) * 10 + 50,  # Large values
            y=torch.randn(4, 5),
            static=torch.zeros(4, 0),
            future=torch.zeros(4, 5, 0),
            group_identifiers=["a", "b", "c", "d"],
        )
        
        # Should work with RevIN
        output = model(batch.X)
        assert output.shape == (4, 5, 1)
        
        # Training should work
        loss = model.training_step(batch, 0)
        assert loss.item() > 0
        
    def test_dict_config_integration(self):
        """Test model creation with dictionary config."""
        config_dict = {
            "input_len": 25,
            "output_len": 10,
            "input_size": 7,
            "static_size": 5,
            "learning_rate": 1e-6,
        }
        
        model = LitNaiveLastValue(config_dict)
        
        # Verify config was properly converted
        assert model.config.input_len == 25
        assert model.config.output_len == 10
        assert model.config.input_size == 7
        
        # Test forward pass
        batch = Batch(
            X=torch.randn(2, 25, 7),
            y=torch.randn(2, 10),
            static=torch.randn(2, 5),
            future=torch.zeros(2, 10, 0),
            group_identifiers=["a", "b"],
        )
        
        output = model(batch.X, batch.static, batch.future)
        assert output.shape == (2, 10, 1)
        
    def test_optimizer_configuration(self, config):
        """Test that optimizer is properly configured."""
        model = LitNaiveLastValue(config)
        
        optimizer_config = model.configure_optimizers()
        
        # Get optimizer
        optimizer = optimizer_config
        if isinstance(optimizer_config, dict):
            optimizer = optimizer_config["optimizer"]
            
        # Verify parameters are registered
        param_count = sum(1 for _ in model.parameters())
        assert param_count > 0
        
        # Verify optimizer has model parameters
        assert len(optimizer.param_groups) > 0
        assert len(optimizer.param_groups[0]["params"]) > 0
        
    def test_loss_calculation(self, config, batch):
        """Test that loss is calculated correctly."""
        model = LitNaiveLastValue(config)
        
        # Get predictions
        predictions = model(batch.X, batch.static, batch.future)
        
        # Calculate expected loss manually
        criterion = torch.nn.MSELoss()
        expected_loss = criterion(predictions, batch.y.unsqueeze(-1))
        
        # Get actual loss from training step
        actual_loss = model.training_step(batch, 0)
        
        assert torch.allclose(actual_loss, expected_loss, atol=1e-5)