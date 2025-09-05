import pytest
import torch

from transfer_learning_publication.contracts import Sample


class TestSample:
    """Test Sample dataclass functionality."""

    @pytest.fixture
    def valid_tensors(self):
        """Create valid tensors for testing."""
        X = torch.randn(10, 5)  # (input_length, n_input_features)
        y = torch.randn(5)  # (output_length,)
        static = torch.randn(3)  # (n_static_features,)
        future = torch.randn(5, 2)  # (output_length, n_future_features)
        return X, y, static, future

    @pytest.fixture
    def valid_tensors_2d_y(self):
        """Create valid tensors with 2D y."""
        X = torch.randn(10, 5)
        y = torch.randn(5, 1)  # 2D target
        static = torch.randn(3)
        future = torch.randn(5, 2)
        return X, y, static, future

    def test_valid_sample_creation(self, valid_tensors):
        """Test creating a valid sample."""
        X, y, static, future = valid_tensors
        sample = Sample(
            X=X,
            y=y,
            static=static,
            future=future,
            group_identifier="gauge_001",
            input_end_date=1234567890,
        )

        assert torch.equal(sample.X, X)
        assert torch.equal(sample.y, y)
        assert torch.equal(sample.static, static)
        assert torch.equal(sample.future, future)
        assert sample.group_identifier == "gauge_001"
        assert sample.input_end_date == 1234567890

    def test_valid_sample_2d_y(self, valid_tensors_2d_y):
        """Test creating a sample with 2D y tensor."""
        X, y, static, future = valid_tensors_2d_y
        sample = Sample(
            X=X,
            y=y,
            static=static,
            future=future,
            group_identifier="gauge_002",
            input_end_date=None,
        )

        assert sample.y.shape == (5, 1)
        assert sample.input_end_date is None

    def test_sample_with_empty_future(self, valid_tensors):
        """Test creating a sample with empty future tensor."""
        X, y, static, _ = valid_tensors
        future = torch.empty(5, 0)  # Empty tensor with 0 features
        sample = Sample(
            X=X,
            y=y,
            static=static,
            future=future,
            group_identifier="gauge_003",
            input_end_date=None,
        )

        assert sample.future.shape == (5, 0)
        assert sample.future.numel() == 0

    def test_sample_immutability(self, valid_tensors):
        """Test that Sample is immutable (frozen dataclass)."""
        X, y, static, future = valid_tensors
        sample = Sample(
            X=X,
            y=y,
            static=static,
            future=future,
            group_identifier="gauge_001",
        )

        with pytest.raises(AttributeError):
            sample.group_identifier = "new_id"

    def test_invalid_X_type(self, valid_tensors):
        """Test validation of X tensor type."""
        _, y, static, future = valid_tensors
        with pytest.raises(TypeError, match="X must be torch.Tensor"):
            Sample(
                X=[1, 2, 3],  # Not a tensor
                y=y,
                static=static,
                future=future,
                group_identifier="gauge_001",
            )

    def test_invalid_y_type(self, valid_tensors):
        """Test validation of y tensor type."""
        X, _, static, future = valid_tensors
        with pytest.raises(TypeError, match="y must be torch.Tensor"):
            Sample(
                X=X,
                y=[1, 2, 3],  # Not a tensor
                static=static,
                future=future,
                group_identifier="gauge_001",
            )

    def test_invalid_static_type(self, valid_tensors):
        """Test validation of static tensor type."""
        X, y, _, future = valid_tensors
        with pytest.raises(TypeError, match="static must be torch.Tensor"):
            Sample(
                X=X,
                y=y,
                static=[1, 2, 3],  # Not a tensor
                future=future,
                group_identifier="gauge_001",
            )

    def test_invalid_future_type(self):
        """Test validation of future tensor type."""
        X = torch.randn(10, 5)
        y = torch.randn(5)
        static = torch.randn(3)
        
        with pytest.raises(TypeError, match="future must be torch.Tensor"):
            Sample(
                X=X,
                y=y,
                static=static,
                future=[1, 2, 3],  # Not a tensor
                group_identifier="gauge_001",
            )

    def test_invalid_X_dimensions(self, valid_tensors):
        """Test validation of X tensor dimensions."""
        _, y, static, future = valid_tensors
        X_1d = torch.randn(10)  # 1D instead of 2D
        
        with pytest.raises(ValueError, match="X must be 2D"):
            Sample(
                X=X_1d,
                y=y,
                static=static,
                future=future,
                group_identifier="gauge_001",
            )

    def test_invalid_y_dimensions(self, valid_tensors):
        """Test validation of y tensor dimensions."""
        X, _, static, future = valid_tensors
        y_3d = torch.randn(5, 1, 1)  # 3D instead of 1D or 2D
        
        with pytest.raises(ValueError, match="y must be 1D or 2D"):
            Sample(
                X=X,
                y=y_3d,
                static=static,
                future=future,
                group_identifier="gauge_001",
            )

    def test_invalid_static_dimensions(self, valid_tensors):
        """Test validation of static tensor dimensions."""
        X, y, _, future = valid_tensors
        static_2d = torch.randn(3, 1)  # 2D instead of 1D
        
        with pytest.raises(ValueError, match="static must be 1D"):
            Sample(
                X=X,
                y=y,
                static=static_2d,
                future=future,
                group_identifier="gauge_001",
            )

    def test_invalid_future_dimensions(self):
        """Test validation of future tensor dimensions."""
        X = torch.randn(10, 5)
        y = torch.randn(5)
        static = torch.randn(3)
        future_1d = torch.randn(5)  # 1D instead of 2D
        
        with pytest.raises(ValueError, match="future must be 2D"):
            Sample(
                X=X,
                y=y,
                static=static,
                future=future_1d,
                group_identifier="gauge_001",
            )

    def test_future_output_length_mismatch(self):
        """Test validation of future tensor length matching output length."""
        X = torch.randn(10, 5)
        y = torch.randn(5)
        static = torch.randn(3)
        future = torch.randn(3, 2)  # Wrong length (3 instead of 5)
        
        with pytest.raises(ValueError, match="future length .* must match output length"):
            Sample(
                X=X,
                y=y,
                static=static,
                future=future,
                group_identifier="gauge_001",
            )

    def test_empty_tensors(self):
        """Test sample with empty feature dimensions."""
        X = torch.randn(10, 0)  # No features
        y = torch.randn(5)
        static = torch.randn(0)  # No static features
        future = torch.empty(5, 0)  # Empty future tensor
        
        sample = Sample(
            X=X,
            y=y,
            static=static,
            future=future,
            group_identifier="gauge_001",
        )
        
        assert sample.X.shape == (10, 0)
        assert sample.static.shape == (0,)
        assert sample.future.shape == (5, 0)
        assert sample.future.numel() == 0

    def test_zero_length_sequences(self):
        """Test sample with zero-length sequences."""
        X = torch.randn(0, 5)  # Zero input length
        y = torch.randn(0)  # Zero output length
        static = torch.randn(3)
        future = torch.randn(0, 2)  # Matching zero output length
        
        sample = Sample(
            X=X,
            y=y,
            static=static,
            future=future,
            group_identifier="gauge_001",
        )
        
        assert sample.X.shape == (0, 5)
        assert sample.y.shape == (0,)
        assert sample.future.shape == (0, 2)