import pytest
import torch

from transfer_learning_publication.contracts import Batch


class TestBatch:
    """Test Batch dataclass functionality."""

    @pytest.fixture
    def valid_batch_tensors(self):
        """Create valid batch tensors for testing."""
        batch_size = 4
        input_length = 10
        output_length = 5
        n_input_features = 6
        n_static_features = 3
        n_future_features = 2

        X = torch.randn(batch_size, input_length, n_input_features)
        y = torch.randn(batch_size, output_length)
        static = torch.randn(batch_size, n_static_features)
        future = torch.randn(batch_size, output_length, n_future_features)
        group_identifiers = ["gauge_001", "gauge_002", "gauge_003", "gauge_004"]
        input_end_dates = torch.tensor([1234567890, 1234567891, 1234567892, 1234567893], dtype=torch.long)

        return X, y, static, future, group_identifiers, input_end_dates

    @pytest.fixture
    def valid_batch_2d_y(self):
        """Create valid batch tensors with 2D y."""
        batch_size = 3
        X = torch.randn(batch_size, 10, 5)
        y = torch.randn(batch_size, 5, 1)  # 3D target
        static = torch.randn(batch_size, 3)
        future = torch.randn(batch_size, 5, 2)
        group_identifiers = ["g1", "g2", "g3"]
        return X, y, static, future, group_identifiers

    def test_valid_batch_creation(self, valid_batch_tensors):
        """Test creating a valid batch."""
        X, y, static, future, group_identifiers, input_end_dates = valid_batch_tensors
        batch = Batch(
            X=X,
            y=y,
            static=static,
            future=future,
            group_identifiers=group_identifiers,
            input_end_dates=input_end_dates,
        )

        assert torch.equal(batch.X, X)
        assert torch.equal(batch.y, y)
        assert torch.equal(batch.static, static)
        assert torch.equal(batch.future, future)
        assert batch.group_identifiers == group_identifiers
        assert torch.equal(batch.input_end_dates, input_end_dates)

    def test_batch_with_empty_future(self, valid_batch_tensors):
        """Test creating a batch with empty future tensor."""
        X, y, static, _, group_identifiers, _ = valid_batch_tensors
        batch_size = X.shape[0]
        output_length = y.shape[1]
        future = torch.empty(batch_size, output_length, 0)  # Empty future tensor

        batch = Batch(
            X=X,
            y=y,
            static=static,
            future=future,
            group_identifiers=group_identifiers,
            input_end_dates=None,
        )

        assert batch.future.shape == (batch_size, output_length, 0)
        assert batch.n_future_features == 0
        assert batch.input_end_dates is None

    def test_batch_properties(self, valid_batch_tensors):
        """Test batch property methods."""
        X, y, static, future, group_identifiers, input_end_dates = valid_batch_tensors
        batch = Batch(
            X=X,
            y=y,
            static=static,
            future=future,
            group_identifiers=group_identifiers,
            input_end_dates=input_end_dates,
        )

        assert batch.batch_size == 4
        assert batch.input_length == 10
        assert batch.output_length == 5
        assert batch.n_input_features == 6
        assert batch.n_static_features == 3
        assert batch.n_future_features == 2

    def test_batch_properties_with_empty_future(self, valid_batch_tensors):
        """Test batch properties when future has zero features."""
        X, y, static, _, group_identifiers, _ = valid_batch_tensors
        future = torch.empty(4, 5, 0)  # Empty future tensor
        batch = Batch(
            X=X,
            y=y,
            static=static,
            future=future,
            group_identifiers=group_identifiers,
        )

        assert batch.n_future_features == 0
        assert batch.future.shape == (4, 5, 0)

    def test_batch_properties_with_zero_features(self, valid_batch_tensors):
        """Test batch properties when future has zero features."""
        X, y, static, _, group_identifiers, _ = valid_batch_tensors
        future_empty = torch.empty(4, 5, 0)  # Zero future features
        batch = Batch(
            X=X,
            y=y,
            static=static,
            future=future_empty,
            group_identifiers=group_identifiers,
        )

        assert batch.n_future_features == 0
        assert batch.future.numel() == 0

    def test_batch_immutability(self, valid_batch_tensors):
        """Test that Batch is immutable (frozen dataclass)."""
        X, y, static, future, group_identifiers, _ = valid_batch_tensors
        batch = Batch(
            X=X,
            y=y,
            static=static,
            future=future,
            group_identifiers=group_identifiers,
        )

        with pytest.raises(AttributeError):
            batch.X = torch.randn(4, 10, 6)

    def test_to_device_method(self, valid_batch_tensors):
        """Test moving batch to different device."""
        X, y, static, future, group_identifiers, input_end_dates = valid_batch_tensors
        batch = Batch(
            X=X,
            y=y,
            static=static,
            future=future,
            group_identifiers=group_identifiers,
            input_end_dates=input_end_dates,
        )

        # Test moving to CPU (should work even if already on CPU)
        device = torch.device("cpu")
        batch_cpu = batch.to(device)

        assert batch_cpu.X.device.type == "cpu"
        assert batch_cpu.y.device.type == "cpu"
        assert batch_cpu.static.device.type == "cpu"
        assert batch_cpu.future.device.type == "cpu"
        assert batch_cpu.input_end_dates.device.type == "cpu"
        assert batch_cpu.group_identifiers == group_identifiers  # Strings stay as is

        # Test that it returns a new batch
        assert batch_cpu is not batch

    def test_to_device_with_none_input_dates(self, valid_batch_tensors):
        """Test moving batch with None input_end_dates to device."""
        X, y, static, future, group_identifiers, _ = valid_batch_tensors
        batch = Batch(
            X=X,
            y=y,
            static=static,
            future=future,
            group_identifiers=group_identifiers,
            input_end_dates=None,  # Only input_end_dates can be None
        )

        device = torch.device("cpu")
        batch_cpu = batch.to(device)

        assert batch_cpu.X.device.type == "cpu"
        assert batch_cpu.future.device.type == "cpu"
        assert batch_cpu.input_end_dates is None

    def test_as_dict_method(self, valid_batch_tensors):
        """Test converting batch to dictionary."""
        X, y, static, future, group_identifiers, input_end_dates = valid_batch_tensors
        batch = Batch(
            X=X,
            y=y,
            static=static,
            future=future,
            group_identifiers=group_identifiers,
            input_end_dates=input_end_dates,
        )

        batch_dict = batch.as_dict()

        assert isinstance(batch_dict, dict)
        assert torch.equal(batch_dict["X"], X)
        assert torch.equal(batch_dict["y"], y)
        assert torch.equal(batch_dict["static"], static)
        assert torch.equal(batch_dict["future"], future)
        assert batch_dict["group_identifiers"] == group_identifiers
        assert torch.equal(batch_dict["input_end_dates"], input_end_dates)

    def test_invalid_y_batch_size(self, valid_batch_tensors):
        """Test validation of y batch size consistency."""
        X, _, static, future, group_identifiers, _ = valid_batch_tensors
        y_wrong_size = torch.randn(3, 5)  # Wrong batch size

        with pytest.raises(ValueError, match="y batch size .* doesn't match X"):
            Batch(
                X=X,
                y=y_wrong_size,
                static=static,
                future=future,
                group_identifiers=group_identifiers,
            )

    def test_invalid_static_batch_size(self, valid_batch_tensors):
        """Test validation of static batch size consistency."""
        X, y, _, future, group_identifiers, _ = valid_batch_tensors
        static_wrong_size = torch.randn(3, 3)  # Wrong batch size

        with pytest.raises(ValueError, match="static batch size .* doesn't match X"):
            Batch(
                X=X,
                y=y,
                static=static_wrong_size,
                future=future,
                group_identifiers=group_identifiers,
            )

    def test_invalid_future_batch_size(self, valid_batch_tensors):
        """Test validation of future batch size consistency."""
        X, y, static, _, group_identifiers, _ = valid_batch_tensors
        future_wrong_size = torch.randn(3, 5, 2)  # Wrong batch size

        with pytest.raises(ValueError, match="future batch size .* doesn't match X"):
            Batch(
                X=X,
                y=y,
                static=static,
                future=future_wrong_size,
                group_identifiers=group_identifiers,
            )

    def test_invalid_group_identifiers_length(self, valid_batch_tensors):
        """Test validation of group_identifiers length."""
        X, y, static, future, _, _ = valid_batch_tensors
        wrong_identifiers = ["g1", "g2", "g3"]  # Wrong length

        with pytest.raises(ValueError, match="group_identifiers length .* doesn't match batch size"):
            Batch(
                X=X,
                y=y,
                static=static,
                future=future,
                group_identifiers=wrong_identifiers,
            )

    def test_invalid_input_end_dates_size(self, valid_batch_tensors):
        """Test validation of input_end_dates size."""
        X, y, static, future, group_identifiers, _ = valid_batch_tensors
        wrong_dates = torch.tensor([1, 2, 3], dtype=torch.long)  # Wrong size

        with pytest.raises(ValueError, match="input_end_dates size .* doesn't match batch size"):
            Batch(
                X=X,
                y=y,
                static=static,
                future=future,
                group_identifiers=group_identifiers,
                input_end_dates=wrong_dates,
            )

    def test_invalid_X_dimensions(self, valid_batch_tensors):
        """Test validation of X tensor dimensions."""
        _, y, static, future, group_identifiers, _ = valid_batch_tensors
        X_2d = torch.randn(4, 10)  # 2D instead of 3D

        with pytest.raises(ValueError, match="X must be 3D"):
            Batch(
                X=X_2d,
                y=y,
                static=static,
                future=future,
                group_identifiers=group_identifiers,
            )

    def test_invalid_y_dimensions(self, valid_batch_tensors):
        """Test validation of y tensor dimensions."""
        X, _, static, future, group_identifiers, _ = valid_batch_tensors
        y_1d = torch.randn(4)  # 1D instead of 2D or 3D

        with pytest.raises(ValueError, match="y must be 2D or 3D"):
            Batch(
                X=X,
                y=y_1d,
                static=static,
                future=future,
                group_identifiers=group_identifiers,
            )

    def test_invalid_static_dimensions(self, valid_batch_tensors):
        """Test validation of static tensor dimensions."""
        X, y, _, future, group_identifiers, _ = valid_batch_tensors
        static_3d = torch.randn(4, 3, 1)  # 3D instead of 2D

        with pytest.raises(ValueError, match="static must be 2D"):
            Batch(
                X=X,
                y=y,
                static=static_3d,
                future=future,
                group_identifiers=group_identifiers,
            )

    def test_invalid_future_dimensions(self, valid_batch_tensors):
        """Test validation of future tensor dimensions."""
        X, y, static, _, group_identifiers, _ = valid_batch_tensors
        future_2d = torch.randn(4, 5)  # 2D instead of 3D

        with pytest.raises(ValueError, match="future must be 3D"):
            Batch(
                X=X,
                y=y,
                static=static,
                future=future_2d,
                group_identifiers=group_identifiers,
            )

    def test_empty_batch(self):
        """Test batch with zero samples."""
        X = torch.randn(0, 10, 5)
        y = torch.randn(0, 5)
        static = torch.randn(0, 3)
        future = torch.randn(0, 5, 2)
        group_identifiers = []

        batch = Batch(
            X=X,
            y=y,
            static=static,
            future=future,
            group_identifiers=group_identifiers,
        )

        assert batch.batch_size == 0
        assert batch.input_length == 10
        assert batch.output_length == 5

    def test_single_sample_batch(self):
        """Test batch with single sample."""
        X = torch.randn(1, 10, 5)
        y = torch.randn(1, 5)
        static = torch.randn(1, 3)
        future = torch.randn(1, 5, 2)
        group_identifiers = ["gauge_001"]

        batch = Batch(
            X=X,
            y=y,
            static=static,
            future=future,
            group_identifiers=group_identifiers,
        )

        assert batch.batch_size == 1
        assert len(batch.group_identifiers) == 1
