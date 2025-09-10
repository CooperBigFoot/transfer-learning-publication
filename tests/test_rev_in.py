"""Tests for RevIN (Reversible Instance Normalization) layer."""

import pytest
import torch

from transfer_learning_publication.models.layers import RevIN


class TestRevIN:
    """Test suite for RevIN layer."""

    def test_initialization(self):
        """Test RevIN initialization."""
        rev_in = RevIN(num_features=1, eps=1e-5, affine=True)

        assert rev_in.num_features == 1
        assert rev_in.eps == 1e-5
        assert rev_in.affine is True
        assert rev_in.mean is None
        assert rev_in.stdev is None

        # Check affine parameters initialized
        assert hasattr(rev_in, "affine_weight")
        assert hasattr(rev_in, "affine_bias")
        assert rev_in.affine_weight.shape == (1,)
        assert rev_in.affine_bias.shape == (1,)

    def test_initialization_no_affine(self):
        """Test RevIN initialization without affine transformation."""
        rev_in = RevIN(num_features=1, affine=False)

        assert rev_in.affine is False
        assert not hasattr(rev_in, "affine_weight")
        assert not hasattr(rev_in, "affine_bias")

    def test_normalization(self):
        """Test normalization process."""
        rev_in = RevIN(num_features=1, affine=False)

        # Create input with known statistics
        batch_size = 2
        seq_len = 10
        features = 1

        # Create data with mean=5, std=2
        x = torch.randn(batch_size, seq_len, features) * 2 + 5

        # Apply normalization
        x_norm = rev_in(x, mode="norm")

        # Check output shape
        assert x_norm.shape == x.shape

        # Check that statistics are stored
        assert rev_in.mean is not None
        assert rev_in.stdev is not None
        assert rev_in.mean.shape == (batch_size, 1, features)
        assert rev_in.stdev.shape == (batch_size, 1, features)

        # Check normalization effect (should have near-zero mean and unit variance per instance)
        instance_means = x_norm.mean(dim=1, keepdim=True)
        instance_stds = x_norm.std(dim=1, keepdim=True, unbiased=False)

        assert torch.allclose(instance_means, torch.zeros_like(instance_means), atol=1e-5)
        assert torch.allclose(instance_stds, torch.ones_like(instance_stds), atol=1e-5)

    def test_denormalization(self):
        """Test denormalization process."""
        rev_in = RevIN(num_features=1, affine=False)

        batch_size = 3
        seq_len = 15
        features = 1

        # Original data
        x = torch.randn(batch_size, seq_len, features) * 3 + 10

        # Normalize
        x_norm = rev_in(x, mode="norm")

        # Denormalize
        x_denorm = rev_in(x_norm, mode="denorm")

        # Should recover original data
        assert torch.allclose(x_denorm, x, atol=1e-5)

    def test_affine_transformation(self):
        """Test affine transformation in normalization."""
        rev_in = RevIN(num_features=1, affine=True)

        # Set known affine parameters
        with torch.no_grad():
            rev_in.affine_weight.fill_(2.0)
            rev_in.affine_bias.fill_(3.0)

        batch_size = 2
        seq_len = 10
        features = 1

        x = torch.randn(batch_size, seq_len, features)

        # Apply normalization with affine
        x_norm = rev_in(x, mode="norm")

        # The output should be: (normalized * 2) + 3
        x_norm_no_affine = (x - x.mean(dim=1, keepdim=True)) / torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + rev_in.eps
        )
        expected = x_norm_no_affine * 2.0 + 3.0

        assert torch.allclose(x_norm, expected, atol=1e-4)

    def test_multi_feature(self):
        """Test RevIN with multiple features."""
        num_features = 3
        rev_in = RevIN(num_features=num_features, affine=True)

        batch_size = 2
        seq_len = 10

        x = torch.randn(batch_size, seq_len, num_features)

        # Normalize
        x_norm = rev_in(x, mode="norm")

        assert x_norm.shape == x.shape
        assert rev_in.mean.shape == (batch_size, 1, num_features)
        assert rev_in.stdev.shape == (batch_size, 1, num_features)

        # Denormalize
        x_denorm = rev_in(x_norm, mode="denorm")

        assert torch.allclose(x_denorm, x, atol=1e-5)

    def test_invalid_mode(self):
        """Test error with invalid mode."""
        rev_in = RevIN(num_features=1)
        x = torch.randn(2, 10, 1)

        with pytest.raises(NotImplementedError, match="Mode 'invalid' not implemented"):
            rev_in(x, mode="invalid")

    def test_denorm_without_norm(self):
        """Test error when denormalizing without prior normalization."""
        rev_in = RevIN(num_features=1)
        x = torch.randn(2, 10, 1)

        with pytest.raises(RuntimeError, match="Cannot denormalize: statistics not computed"):
            rev_in(x, mode="denorm")

    def test_reset_statistics(self):
        """Test resetting stored statistics."""
        rev_in = RevIN(num_features=1)
        x = torch.randn(2, 10, 1)

        # Normalize to store statistics
        rev_in(x, mode="norm")

        assert rev_in.mean is not None
        assert rev_in.stdev is not None

        # Reset
        rev_in.reset_statistics()

        assert rev_in.mean is None
        assert rev_in.stdev is None

    def test_different_batch_sizes(self):
        """Test RevIN with different batch sizes."""
        rev_in = RevIN(num_features=1)

        # First batch
        x1 = torch.randn(3, 10, 1)
        x1_norm = rev_in(x1, mode="norm")

        assert x1_norm.shape == (3, 10, 1)
        assert rev_in.mean.shape == (3, 1, 1)

        # Second batch with different size
        x2 = torch.randn(5, 10, 1)
        x2_norm = rev_in(x2, mode="norm")

        assert x2_norm.shape == (5, 10, 1)
        assert rev_in.mean.shape == (5, 1, 1)  # Updated to new batch size

    def test_zero_variance_handling(self):
        """Test handling of zero variance (constant sequences)."""
        rev_in = RevIN(num_features=1, eps=1e-5)

        batch_size = 2
        seq_len = 10

        # Create constant sequences
        x = torch.ones(batch_size, seq_len, 1) * 5.0

        # Should handle without division by zero
        x_norm = rev_in(x, mode="norm")

        # With constant input, normalized output should be zero (due to eps in denominator)
        assert torch.allclose(x_norm, torch.zeros_like(x_norm), atol=1e-4)

    def test_gradient_flow(self):
        """Test that gradients flow through RevIN."""
        rev_in = RevIN(num_features=1, affine=True)

        x = torch.randn(2, 10, 1, requires_grad=True)

        # Forward
        x_norm = rev_in(x, mode="norm")

        # Create loss and backward
        loss = x_norm.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert rev_in.affine_weight.grad is not None
        assert rev_in.affine_bias.grad is not None
