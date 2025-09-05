import pytest
import torch

from transfer_learning_publication.contracts import Batch, Sample, collate_fn, collate_samples


class TestCollate:
    """Test collation functions."""

    @pytest.fixture
    def create_sample(self):
        """Factory fixture to create samples."""

        def _create_sample(
            input_length=10,
            output_length=5,
            n_input_features=3,
            n_static_features=2,
            n_future_features=1,
            group_id="gauge_001",
            input_end_date=None,
            with_future=True,
        ):
            X = torch.randn(input_length, n_input_features)
            y = torch.randn(output_length)
            static = torch.randn(n_static_features)
            
            if with_future and n_future_features > 0:
                future = torch.randn(output_length, n_future_features)
            else:
                future = torch.empty(output_length, 0)  # Always use empty tensor, never None
                
            return Sample(
                X=X,
                y=y,
                static=static,
                future=future,
                group_identifier=group_id,
                input_end_date=input_end_date,
            )

        return _create_sample

    @pytest.fixture
    def consistent_samples(self, create_sample):
        """Create a list of consistent samples."""
        samples = [
            create_sample(group_id=f"gauge_{i:03d}", input_end_date=1234567890 + i)
            for i in range(4)
        ]
        return samples

    @pytest.fixture
    def samples_without_future(self, create_sample):
        """Create samples without future tensors."""
        samples = [
            create_sample(group_id=f"gauge_{i:03d}", with_future=False)
            for i in range(3)
        ]
        return samples

    @pytest.fixture
    def samples_with_empty_future(self, create_sample):
        """Create samples with empty future tensors."""
        samples = []
        for i in range(3):
            sample = create_sample(
                group_id=f"gauge_{i:03d}",
                n_future_features=0,  # Empty future features
            )
            # Manually set to empty tensor with correct shape
            sample = Sample(
                X=sample.X,
                y=sample.y,
                static=sample.static,
                future=torch.empty(5, 0),  # (output_length, 0)
                group_identifier=sample.group_identifier,
                input_end_date=sample.input_end_date,
            )
            samples.append(sample)
        return samples

    def test_collate_samples_success(self, consistent_samples):
        """Test successful collation of consistent samples."""
        batch = collate_samples(consistent_samples)

        assert isinstance(batch, Batch)
        assert batch.batch_size == 4
        assert batch.X.shape == (4, 10, 3)
        assert batch.y.shape == (4, 5)
        assert batch.static.shape == (4, 2)
        assert batch.future.shape == (4, 5, 1)
        assert len(batch.group_identifiers) == 4
        assert batch.group_identifiers == ["gauge_000", "gauge_001", "gauge_002", "gauge_003"]
        assert batch.input_end_dates.shape == (4,)
        assert batch.input_end_dates[0] == 1234567890
        assert batch.input_end_dates[3] == 1234567893

    def test_collate_fn_alias(self, consistent_samples):
        """Test that collate_fn is an alias for collate_samples."""
        batch1 = collate_samples(consistent_samples)
        batch2 = collate_fn(consistent_samples)

        assert torch.equal(batch1.X, batch2.X)
        assert torch.equal(batch1.y, batch2.y)
        assert torch.equal(batch1.static, batch2.static)
        assert torch.equal(batch1.future, batch2.future)
        assert batch1.group_identifiers == batch2.group_identifiers

    def test_collate_empty_list(self):
        """Test collating an empty list of samples."""
        with pytest.raises(ValueError, match="Cannot collate empty list of samples"):
            collate_samples([])

    def test_collate_single_sample(self, create_sample):
        """Test collating a single sample."""
        sample = create_sample()
        batch = collate_samples([sample])

        assert batch.batch_size == 1
        assert batch.X.shape == (1, 10, 3)
        assert batch.y.shape == (1, 5)
        assert batch.static.shape == (1, 2)
        assert batch.future.shape == (1, 5, 1)

    def test_collate_samples_without_future(self, samples_without_future):
        """Test collating samples with empty future tensors."""
        # No workaround needed - samples already have empty tensors
        batch = collate_samples(samples_without_future)

        assert batch.future.shape == (3, 5, 0)  # Empty future features
        assert batch.n_future_features == 0

    def test_collate_samples_with_empty_future(self, samples_with_empty_future):
        """Test collating samples with empty future tensors."""
        batch = collate_samples(samples_with_empty_future)

        assert batch.future.shape == (3, 5, 0)
        assert batch.n_future_features == 0

    def test_collate_samples_without_dates(self, create_sample):
        """Test collating samples without input_end_date."""
        samples = [
            create_sample(group_id=f"gauge_{i:03d}", input_end_date=None)
            for i in range(3)
        ]
        batch = collate_samples(samples)

        assert batch.input_end_dates is None

    def test_collate_inconsistent_shapes(self, create_sample):
        """Test collating samples with inconsistent shapes."""
        sample1 = create_sample(input_length=10)
        sample2 = create_sample(input_length=15)  # Different input length

        with pytest.raises(RuntimeError, match="Failed to stack samples into batch"):
            collate_samples([sample1, sample2])

    def test_collate_inconsistent_features(self, create_sample):
        """Test collating samples with inconsistent feature dimensions."""
        sample1 = create_sample(n_input_features=3)
        sample2 = create_sample(n_input_features=5)  # Different feature count

        with pytest.raises(RuntimeError, match="Failed to stack samples into batch"):
            collate_samples([sample1, sample2])

    def test_collate_inconsistent_output_length(self, create_sample):
        """Test collating samples with inconsistent output lengths."""
        sample1 = create_sample(output_length=5)
        sample2 = create_sample(output_length=7)  # Different output length

        with pytest.raises(RuntimeError, match="Failed to stack samples into batch"):
            collate_samples([sample1, sample2])

    def test_collate_mixed_future_empty_and_nonempty(self, create_sample):
        """Test collating samples with mixed empty and non-empty future tensors."""
        sample1 = create_sample(with_future=True, n_future_features=2)
        sample2 = create_sample(with_future=False)  # Will have empty tensor

        # This should fail because future tensors have different shapes
        with pytest.raises(RuntimeError, match="Failed to stack samples into batch"):
            collate_samples([sample1, sample2])

    def test_collate_2d_y(self, create_sample):
        """Test collating samples with 2D y tensors."""
        samples = []
        for i in range(3):
            base_sample = create_sample(group_id=f"gauge_{i:03d}")
            # Create sample with 2D y
            sample = Sample(
                X=base_sample.X,
                y=base_sample.y.unsqueeze(-1),  # Make 2D
                static=base_sample.static,
                future=base_sample.future,
                group_identifier=base_sample.group_identifier,
                input_end_date=base_sample.input_end_date,
            )
            samples.append(sample)

        batch = collate_samples(samples)
        assert batch.y.shape == (3, 5, 1)

    def test_collate_large_batch(self, create_sample):
        """Test collating a large batch of samples."""
        samples = [
            create_sample(group_id=f"gauge_{i:04d}")
            for i in range(100)
        ]
        batch = collate_samples(samples)

        assert batch.batch_size == 100
        assert len(batch.group_identifiers) == 100

    def test_collate_preserves_tensor_values(self, create_sample):
        """Test that collation preserves exact tensor values."""
        # Create samples with known values
        samples = []
        for i in range(3):
            X = torch.ones(10, 3) * i
            y = torch.ones(5) * (i + 10)
            static = torch.ones(2) * (i + 20)
            future = torch.ones(5, 1) * (i + 30)
            
            sample = Sample(
                X=X,
                y=y,
                static=static,
                future=future,
                group_identifier=f"gauge_{i:03d}",
                input_end_date=1000 + i,
            )
            samples.append(sample)

        batch = collate_samples(samples)

        # Check that values are preserved
        for i in range(3):
            assert torch.allclose(batch.X[i], torch.ones(10, 3) * i)
            assert torch.allclose(batch.y[i], torch.ones(5) * (i + 10))
            assert torch.allclose(batch.static[i], torch.ones(2) * (i + 20))
            assert torch.allclose(batch.future[i], torch.ones(5, 1) * (i + 30))
            assert batch.input_end_dates[i] == 1000 + i

    def test_collate_zero_length_sequences(self):
        """Test collating samples with zero-length sequences."""
        samples = []
        for i in range(2):
            sample = Sample(
                X=torch.randn(0, 3),  # Zero input length
                y=torch.randn(0),  # Zero output length
                static=torch.randn(2),
                future=torch.randn(0, 1),  # Matching zero output
                group_identifier=f"gauge_{i:03d}",
            )
            samples.append(sample)

        batch = collate_samples(samples)
        assert batch.X.shape == (2, 0, 3)
        assert batch.y.shape == (2, 0)
        assert batch.future.shape == (2, 0, 1)

    def test_empty_future_end_to_end(self):
        """Test that empty future tensors work through entire pipeline without special handling."""
        # Create samples with empty future tensors
        samples = []
        for i in range(3):
            sample = Sample(
                X=torch.randn(10, 4),
                y=torch.randn(5),
                static=torch.randn(2),
                future=torch.empty(5, 0),  # Empty future tensor
                group_identifier=f"gauge_{i:03d}",
                input_end_date=1000 + i,
            )
            samples.append(sample)
        
        # Collate should work without special handling
        batch = collate_samples(samples)
        
        # Verify batch properties
        assert batch.future.shape == (3, 5, 0)
        assert batch.n_future_features == 0
        assert batch.future.numel() == 0
        
        # Test device transfer works
        device = torch.device("cpu")
        batch_moved = batch.to(device)
        assert batch_moved.future.shape == (3, 5, 0)
        assert batch_moved.future.device.type == "cpu"
        
        # Test as_dict works
        batch_dict = batch.as_dict()
        assert batch_dict["future"].shape == (3, 5, 0)
        
        # Verify other tensors are unaffected
        assert batch.X.shape == (3, 10, 4)
        assert batch.y.shape == (3, 5)
        assert batch.static.shape == (3, 2)
        assert len(batch.group_identifiers) == 3
        assert batch.input_end_dates.tolist() == [1000, 1001, 1002]