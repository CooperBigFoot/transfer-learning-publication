"""Tests for ForecastOutput contract."""

import pytest
import torch

from transfer_learning_publication.contracts import ForecastOutput


class TestForecastOutput:
    """Test suite for ForecastOutput contract."""
    
    def test_valid_forecast_output(self):
        """Test creating a valid ForecastOutput."""
        n_samples = 10
        output_len = 7
        
        predictions = torch.randn(n_samples, output_len)
        observations = torch.randn(n_samples, output_len)
        group_identifiers = [f"gauge_{i}" for i in range(n_samples)]
        
        output = ForecastOutput(
            predictions=predictions,
            observations=observations,
            group_identifiers=group_identifiers,
        )
        
        assert output.predictions.shape == (n_samples, output_len)
        assert output.observations.shape == (n_samples, output_len)
        assert len(output.group_identifiers) == n_samples
        assert output.input_end_dates is None
        
    def test_forecast_output_with_dates(self):
        """Test ForecastOutput with input_end_dates."""
        n_samples = 5
        output_len = 3
        
        predictions = torch.randn(n_samples, output_len)
        observations = torch.randn(n_samples, output_len)
        group_identifiers = [f"basin_{i}" for i in range(n_samples)]
        input_end_dates = torch.tensor([100, 200, 300, 400, 500])
        
        output = ForecastOutput(
            predictions=predictions,
            observations=observations,
            group_identifiers=group_identifiers,
            input_end_dates=input_end_dates,
        )
        
        assert output.input_end_dates is not None
        assert output.input_end_dates.shape == (n_samples,)
        
    def test_immutability(self):
        """Test that ForecastOutput is immutable (frozen dataclass)."""
        n_samples = 3
        output_len = 2
        
        output = ForecastOutput(
            predictions=torch.zeros(n_samples, output_len),
            observations=torch.ones(n_samples, output_len),
            group_identifiers=["a", "b", "c"],
        )
        
        # Should not be able to modify attributes
        with pytest.raises(AttributeError):
            output.predictions = torch.randn(n_samples, output_len)
            
    def test_mismatched_predictions_observations(self):
        """Test error when predictions and observations have different sizes."""
        predictions = torch.randn(10, 5)
        observations = torch.randn(8, 5)  # Different number of samples
        group_identifiers = ["gauge"] * 10
        
        with pytest.raises(ValueError, match="Predictions .* and observations .* must have same number of samples"):
            ForecastOutput(
                predictions=predictions,
                observations=observations,
                group_identifiers=group_identifiers,
            )
            
    def test_mismatched_group_identifiers(self):
        """Test error when group_identifiers length doesn't match samples."""
        n_samples = 10
        output_len = 5
        
        predictions = torch.randn(n_samples, output_len)
        observations = torch.randn(n_samples, output_len)
        group_identifiers = ["gauge"] * 8  # Wrong length
        
        with pytest.raises(ValueError, match="Number of predictions .* must match number of group identifiers"):
            ForecastOutput(
                predictions=predictions,
                observations=observations,
                group_identifiers=group_identifiers,
            )
            
    def test_mismatched_input_end_dates(self):
        """Test error when input_end_dates size doesn't match samples."""
        n_samples = 10
        output_len = 5
        
        predictions = torch.randn(n_samples, output_len)
        observations = torch.randn(n_samples, output_len)
        group_identifiers = ["gauge"] * n_samples
        input_end_dates = torch.tensor([1, 2, 3])  # Wrong length
        
        with pytest.raises(ValueError, match="Input end dates .* must match number of predictions"):
            ForecastOutput(
                predictions=predictions,
                observations=observations,
                group_identifiers=group_identifiers,
                input_end_dates=input_end_dates,
            )
            
    def test_to_dict(self):
        """Test converting ForecastOutput to dictionary."""
        n_samples = 4
        output_len = 3
        
        predictions = torch.randn(n_samples, output_len)
        observations = torch.randn(n_samples, output_len)
        group_identifiers = [f"gauge_{i}" for i in range(n_samples)]
        input_end_dates = torch.tensor([10, 20, 30, 40])
        
        output = ForecastOutput(
            predictions=predictions,
            observations=observations,
            group_identifiers=group_identifiers,
            input_end_dates=input_end_dates,
        )
        
        output_dict = output.to_dict()
        
        assert "predictions" in output_dict
        assert "observations" in output_dict
        assert "group_identifiers" in output_dict
        assert "input_end_dates" in output_dict
        
        assert torch.equal(output_dict["predictions"], predictions)
        assert torch.equal(output_dict["observations"], observations)
        assert output_dict["group_identifiers"] == group_identifiers
        assert torch.equal(output_dict["input_end_dates"], input_end_dates)
        
    def test_empty_group_identifiers(self):
        """Test error with empty group_identifiers list."""
        predictions = torch.randn(0, 5)
        observations = torch.randn(0, 5)
        group_identifiers = []
        
        # Should work with all empty
        output = ForecastOutput(
            predictions=predictions,
            observations=observations,
            group_identifiers=group_identifiers,
        )
        
        assert output.predictions.shape == (0, 5)
        assert output.observations.shape == (0, 5)
        assert len(output.group_identifiers) == 0