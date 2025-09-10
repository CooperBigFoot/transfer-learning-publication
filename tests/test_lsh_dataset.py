import logging
from datetime import datetime
from unittest.mock import MagicMock, Mock

import pytest
import torch

from transfer_learning_publication.containers import (
    DatasetConfig,
    LSHDataContainer,
    SequenceIndex,
    StaticAttributeCollection,
    TimeSeriesCollection,
)
from transfer_learning_publication.contracts.sample import Sample
from transfer_learning_publication.data import LSHDataset


class TestLSHDataset:
    """Test suite for LSHDataset class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock DatasetConfig."""
        return DatasetConfig(
            input_length=10,
            output_length=5,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation", "latitude"],
            target_idx=0,
            forcing_indices=[0, 1, 2],
            future_indices=[1, 2],
            input_feature_indices=[0, 1, 2],
            is_autoregressive=True,
            include_dates=False,
        )

    @pytest.fixture
    def mock_time_series(self):
        """Create a mock TimeSeriesCollection."""
        mock = MagicMock(spec=TimeSeriesCollection)
        # Create sample tensors for 3 groups
        mock._tensors = [
            torch.randn(100, 3),  # Group 0: 100 timesteps, 3 features
            torch.randn(150, 3),  # Group 1: 150 timesteps, 3 features
            torch.randn(200, 3),  # Group 2: 200 timesteps, 3 features
        ]
        mock.group_identifiers = ["gauge_001", "gauge_002", "gauge_003"]
        mock.__len__ = Mock(return_value=3)
        mock.index_to_date = Mock(return_value=datetime(2023, 1, 15))
        return mock

    @pytest.fixture
    def mock_static_attributes(self):
        """Create a mock StaticAttributeCollection."""
        mock = MagicMock(spec=StaticAttributeCollection)
        # Create static tensors for 3 groups
        mock._tensors = [
            torch.tensor([100.0, 45.5]),  # Group 0: elevation, latitude
            torch.tensor([250.0, 46.2]),  # Group 1
            torch.tensor([500.0, 47.1]),  # Group 2
        ]
        mock.__len__ = Mock(return_value=3)
        return mock

    @pytest.fixture
    def mock_sequence_index(self):
        """Create a mock SequenceIndex."""
        mock = MagicMock(spec=SequenceIndex)
        mock.n_groups = 3
        mock.input_length = 10
        mock.output_length = 5
        mock.__len__ = Mock(return_value=10)  # 10 valid sequences total

        # Define sequence info for different indices
        def get_sequence_info(idx):
            # Simple mapping for test predictability
            sequences = [
                (0, 0, 15),  # Group 0, start 0, end 15
                (0, 10, 25),  # Group 0, start 10, end 25
                (0, 20, 35),  # Group 0, start 20, end 35
                (1, 0, 15),  # Group 1, start 0, end 15
                (1, 15, 30),  # Group 1, start 15, end 30
                (1, 30, 45),  # Group 1, start 30, end 45
                (2, 0, 15),  # Group 2, start 0, end 15
                (2, 20, 35),  # Group 2, start 20, end 35
                (2, 40, 55),  # Group 2, start 40, end 55
                (2, 60, 75),  # Group 2, start 60, end 75
            ]
            if idx < 0 or idx >= len(sequences):
                raise IndexError(f"Index {idx} out of range")
            return sequences[idx]

        mock.get_sequence_info = Mock(side_effect=get_sequence_info)
        return mock

    @pytest.fixture
    def mock_container(self, mock_config, mock_time_series, mock_static_attributes, mock_sequence_index):
        """Create a mock LSHDataContainer."""
        container = MagicMock(spec=LSHDataContainer)
        container.config = mock_config
        container.time_series = mock_time_series
        container.static_attributes = mock_static_attributes
        container.sequence_index = mock_sequence_index
        return container

    def test_init_caching(self, mock_container):
        """Test that initialization properly caches frequently accessed attributes."""
        dataset = LSHDataset(mock_container)

        # Check cached attributes
        assert dataset._n_sequences == 10
        assert dataset._config is mock_container.config
        assert dataset._sequence_index is mock_container.sequence_index
        assert dataset._time_tensors is mock_container.time_series._tensors
        assert dataset._static_tensors is mock_container.static_attributes._tensors
        assert dataset._group_ids == mock_container.time_series.group_identifiers
        assert dataset._input_length == 10
        assert dataset._output_length == 5

    def test_len(self, mock_container):
        """Test __len__ returns correct number of sequences."""
        dataset = LSHDataset(mock_container)
        assert len(dataset) == 10

    def test_getitem_basic(self, mock_container):
        """Test basic __getitem__ functionality."""
        dataset = LSHDataset(mock_container)

        # Get first sample
        sample = dataset[0]

        # Verify it's a Sample instance
        assert isinstance(sample, Sample)

        # Verify tensors have correct shapes
        assert sample.X.shape == (10, 3)  # input_length x input_features
        assert sample.y.shape == (5,)  # output_length
        assert sample.static.shape == (2,)  # n_static_features
        assert sample.future.shape == (5, 2)  # output_length x n_future_features

        # Verify metadata
        assert sample.group_identifier == "gauge_001"
        assert sample.input_end_date is None  # dates disabled

    def test_getitem_with_input_feature_indices(self, mock_time_series, mock_static_attributes, mock_sequence_index):
        """Test __getitem__ when input_feature_indices is specified."""
        config = DatasetConfig(
            input_length=10,
            output_length=5,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation", "latitude"],
            target_idx=0,
            forcing_indices=[0, 1, 2],
            future_indices=[1, 2],
            input_feature_indices=[0, 2],  # Select features 0 and 2
            is_autoregressive=True,
            include_dates=False,
        )
        container = MagicMock(spec=LSHDataContainer)
        container.config = config
        container.time_series = mock_time_series
        container.static_attributes = mock_static_attributes
        container.sequence_index = mock_sequence_index

        dataset = LSHDataset(container)
        sample = dataset[3]

        # Should only have selected features
        assert sample.X.shape == (10, 2)  # input_length x selected_features
        assert sample.group_identifier == "gauge_002"

    def test_getitem_without_input_feature_indices(self, mock_time_series, mock_static_attributes, mock_sequence_index):
        """Test __getitem__ when input_feature_indices is None."""
        config = DatasetConfig(
            input_length=10,
            output_length=5,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation", "latitude"],
            target_idx=0,
            forcing_indices=[0, 1, 2],
            future_indices=[1, 2],
            input_feature_indices=None,  # No specific indices
            is_autoregressive=True,
            include_dates=False,
        )
        container = MagicMock(spec=LSHDataContainer)
        container.config = config
        container.time_series = mock_time_series
        container.static_attributes = mock_static_attributes
        container.sequence_index = mock_sequence_index

        dataset = LSHDataset(container)
        sample = dataset[6]

        # Should have all features
        assert sample.X.shape == (10, 3)  # input_length x all_features
        assert sample.group_identifier == "gauge_003"

    def test_getitem_with_future_indices(self, mock_container):
        """Test __getitem__ with future features."""
        dataset = LSHDataset(mock_container)

        sample = dataset[1]

        # Future should have selected indices
        assert sample.future.shape == (5, 2)  # output_length x n_future_features

    def test_getitem_without_future_indices(self, mock_time_series, mock_static_attributes, mock_sequence_index):
        """Test __getitem__ when no future features."""
        config = DatasetConfig(
            input_length=10,
            output_length=5,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation", "latitude"],
            target_idx=0,
            forcing_indices=[0, 1, 2],
            future_indices=None,  # No future indices
            input_feature_indices=[0, 1, 2],
            is_autoregressive=True,
            include_dates=False,
        )
        container = MagicMock(spec=LSHDataContainer)
        container.config = config
        container.time_series = mock_time_series
        container.static_attributes = mock_static_attributes
        container.sequence_index = mock_sequence_index

        dataset = LSHDataset(container)
        sample = dataset[2]

        # Future should be empty tensor with correct shape
        assert sample.future.shape == (5, 0)  # output_length x 0

    def test_getitem_with_dates(self, mock_time_series, mock_static_attributes, mock_sequence_index):
        """Test __getitem__ with date tracking enabled."""
        config = DatasetConfig(
            input_length=10,
            output_length=5,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation", "latitude"],
            target_idx=0,
            forcing_indices=[0, 1, 2],
            future_indices=[1, 2],
            input_feature_indices=[0, 1, 2],
            is_autoregressive=True,
            include_dates=True,  # Enable date tracking
        )
        container = MagicMock(spec=LSHDataContainer)
        container.config = config
        container.time_series = mock_time_series
        container.static_attributes = mock_static_attributes
        container.sequence_index = mock_sequence_index

        dataset = LSHDataset(container)
        sample = dataset[4]

        # Should have computed input_end_date
        assert sample.input_end_date is not None
        assert isinstance(sample.input_end_date, int)
        # Verify it's a reasonable timestamp (milliseconds)
        assert sample.input_end_date == int(datetime(2023, 1, 15).timestamp() * 1000)

    def test_getitem_date_computation_error(
        self, mock_time_series, mock_static_attributes, mock_sequence_index, caplog
    ):
        """Test __getitem__ handles date computation errors gracefully."""
        config = DatasetConfig(
            input_length=10,
            output_length=5,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation", "latitude"],
            target_idx=0,
            forcing_indices=[0, 1, 2],
            future_indices=[1, 2],
            input_feature_indices=[0, 1, 2],
            is_autoregressive=True,
            include_dates=True,  # Enable date tracking
        )
        container = MagicMock(spec=LSHDataContainer)
        container.config = config
        container.time_series = mock_time_series
        container.static_attributes = mock_static_attributes
        container.sequence_index = mock_sequence_index

        # Make index_to_date raise an error
        mock_time_series.index_to_date.side_effect = ValueError("Date error")

        dataset = LSHDataset(container)

        with caplog.at_level(logging.WARNING):
            sample = dataset[0]

        # Should log warning and set date to None
        assert sample.input_end_date is None
        assert "Failed to compute input_end_date" in caplog.text

    def test_getitem_date_type(self, mock_time_series, mock_static_attributes, mock_sequence_index):
        """Test __getitem__ handles date type as date (not datetime)."""
        from datetime import date

        config = DatasetConfig(
            input_length=10,
            output_length=5,
            target_name="temperature",
            forcing_features=["temperature", "precipitation", "humidity"],
            static_features=["elevation", "latitude"],
            target_idx=0,
            forcing_indices=[0, 1, 2],
            future_indices=[1, 2],
            input_feature_indices=[0, 1, 2],
            is_autoregressive=True,
            include_dates=True,  # Enable date tracking
        )
        container = MagicMock(spec=LSHDataContainer)
        container.config = config
        container.time_series = mock_time_series
        container.static_attributes = mock_static_attributes
        container.sequence_index = mock_sequence_index

        # Return a date instead of datetime
        mock_time_series.index_to_date.return_value = date(2023, 1, 15)

        dataset = LSHDataset(container)
        sample = dataset[0]

        # Should convert date to datetime and then to timestamp
        expected_timestamp = int(datetime.combine(date(2023, 1, 15), datetime.min.time()).timestamp() * 1000)
        assert sample.input_end_date == expected_timestamp

    def test_getitem_index_out_of_bounds(self, mock_container):
        """Test __getitem__ raises IndexError for invalid index."""
        dataset = LSHDataset(mock_container)

        with pytest.raises(IndexError):
            _ = dataset[100]  # Out of bounds

        with pytest.raises(IndexError):
            _ = dataset[-11]  # Negative out of bounds

    def test_getitem_multiple_sequences(self, mock_container):
        """Test getting multiple different sequences."""
        dataset = LSHDataset(mock_container)

        # Get sequences from different groups
        sample0 = dataset[0]  # Group 0
        sample3 = dataset[3]  # Group 1
        sample6 = dataset[6]  # Group 2

        # Verify different group identifiers
        assert sample0.group_identifier == "gauge_001"
        assert sample3.group_identifier == "gauge_002"
        assert sample6.group_identifier == "gauge_003"

        # All should have same dimensions but different data
        assert sample0.X.shape == sample3.X.shape == sample6.X.shape
        assert not torch.equal(sample0.X, sample3.X)
        assert not torch.equal(sample0.static, sample3.static)

    def test_dataset_logging(self, mock_container, caplog):
        """Test that dataset logs initialization info."""
        with caplog.at_level(logging.INFO):
            LSHDataset(mock_container)

        assert "LSHDataset initialized" in caplog.text
        assert "10 sequences" in caplog.text
        assert "input_length=10" in caplog.text
        assert "output_length=5" in caplog.text

    def test_direct_tensor_access_performance(self, mock_container):
        """Test that dataset uses direct tensor access for performance."""
        dataset = LSHDataset(mock_container)

        # Verify direct tensor references are cached
        assert dataset._time_tensors is mock_container.time_series._tensors
        assert dataset._static_tensors is mock_container.static_attributes._tensors

        # Get a sample - should use cached tensors directly
        sample = dataset[0]

        # Verify sample was created (direct access works)
        assert isinstance(sample, Sample)
        assert sample.X is not None
        assert sample.y is not None

    def test_target_extraction(self, mock_container):
        """Test that target is correctly extracted using target_idx."""
        dataset = LSHDataset(mock_container)

        sample = dataset[0]

        # Target should be from column 0 (target_idx=0)
        assert sample.y.shape == (5,)  # output_length, single target

    def test_sequence_boundaries(self, mock_container):
        """Test that sequences respect input/output boundaries."""
        dataset = LSHDataset(mock_container)

        # Test a specific sequence
        sample = dataset[7]  # Group 2, start 20, end 35

        # Verify shapes match expected boundaries
        assert sample.X.shape[0] == 10  # input_length
        assert sample.y.shape[0] == 5  # output_length
        assert sample.future.shape[0] == 5  # output_length

    def test_container_reference(self, mock_container):
        """Test that dataset maintains reference to container."""
        dataset = LSHDataset(mock_container)

        assert dataset.container is mock_container

    def test_target_never_in_future_features(self):
        """
        CRITICAL TEST: Ensure target is NEVER included in future features.

        This test prevents data leakage by ensuring that even if someone
        misconfigures the system to include the target index in future_indices,
        the configuration will be rejected with a clear error message.

        In autoregressive forecasting:
        - Past target values (in X) are OK - we know yesterday's streamflow
        - Future target values (in future) are NOT OK - we don't know tomorrow's streamflow
        """
        # Create a MISCONFIGURATION where target is incorrectly in future_indices
        with pytest.raises(ValueError, match="Data leakage detected.*Target index 0 cannot be in future_indices"):
            DatasetConfig(
                input_length=10,
                output_length=5,
                target_name="temperature",
                forcing_features=["temperature", "precipitation", "humidity"],
                static_features=["elevation", "latitude"],
                target_idx=0,  # First column is the target
                forcing_indices=[0, 1, 2],
                future_indices=[0, 1, 2],  # INCORRECT: includes target index 0!
                input_feature_indices=[0, 1, 2],  # OK: includes target for autoregressive
                is_autoregressive=True,  # Target history should be in X
                include_dates=False,
            )

    def test_target_in_future_allowed_when_not_autoregressive(self):
        """Test that target in future_indices is allowed when not autoregressive."""
        # When is_autoregressive=False, the model doesn't use past target values
        # In this case, having target_idx in future_indices might be intentional
        # (though unusual), so we don't block it
        config = DatasetConfig(
            input_length=10,
            output_length=5,
            target_name="temperature",
            forcing_features=["precipitation", "humidity"],  # Remove temperature since is_autoregressive=False
            static_features=["elevation", "latitude"],
            target_idx=0,
            forcing_indices=[1, 2],  # Indices for precipitation and humidity
            future_indices=[1, 2],  # Future values for precipitation and humidity
            input_feature_indices=[1, 2],  # Excludes target (non-autoregressive)
            is_autoregressive=False,  # Not using past target values
            include_dates=False,
        )
        # Should not raise an error
        assert config.target_idx == 0
        assert config.is_autoregressive is False
