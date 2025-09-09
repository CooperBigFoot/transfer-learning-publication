"""Tests for BaseConfig class."""

import pytest

from transfer_learning_publication.models.base import BaseConfig
from transfer_learning_publication.models.dummy import NaiveLastValueConfig


class TestBaseConfig:
    """Test suite for BaseConfig functionality."""

    def test_base_config_initialization(self):
        """Test basic initialization with required parameters."""
        config = BaseConfig(
            input_len=30,
            output_len=7,
            input_size=5,
        )

        assert config.input_len == 30
        assert config.output_len == 7
        assert config.input_size == 5
        assert config.static_size == 0  # default
        assert config.learning_rate == 1e-5  # default
        assert config.group_identifier == "gauge_id"  # default
        assert config.use_rev_in is True  # default

    def test_base_config_with_optional_params(self):
        """Test initialization with optional parameters."""
        config = BaseConfig(
            input_len=20,
            output_len=10,
            input_size=3,
            static_size=10,
            future_input_size=2,
            learning_rate=1e-3,
            group_identifier="basin_id",
            use_rev_in=False,
        )

        assert config.static_size == 10
        assert config.future_input_size == 2
        assert config.learning_rate == 1e-3
        assert config.group_identifier == "basin_id"
        assert config.use_rev_in is False

    def test_future_input_size_default(self):
        """Test that future_input_size defaults to max(1, input_size - 1)."""
        config1 = BaseConfig(input_len=10, output_len=5, input_size=5)
        assert config1.future_input_size == 4  # max(1, 5-1)

        config2 = BaseConfig(input_len=10, output_len=5, input_size=1)
        assert config2.future_input_size == 1  # max(1, 1-1)

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "input_len": 25,
            "output_len": 8,
            "input_size": 4,
            "static_size": 5,
            "learning_rate": 2e-4,
        }

        config = BaseConfig.from_dict(config_dict)

        assert config.input_len == 25
        assert config.output_len == 8
        assert config.input_size == 4
        assert config.static_size == 5
        assert config.learning_rate == 2e-4

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = BaseConfig(
            input_len=15,
            output_len=3,
            input_size=2,
            static_size=7,
        )

        config_dict = config.to_dict()

        assert config_dict["input_len"] == 15
        assert config_dict["output_len"] == 3
        assert config_dict["input_size"] == 2
        assert config_dict["static_size"] == 7
        assert config_dict["learning_rate"] == 1e-5
        assert config_dict["use_rev_in"] is True
        assert "_" not in [k[0] for k in config_dict.keys()]  # No private attributes

    def test_update(self):
        """Test updating config parameters."""
        config = BaseConfig(
            input_len=10,
            output_len=5,
            input_size=3,
        )

        config.update(learning_rate=1e-2, static_size=15)

        assert config.learning_rate == 1e-2
        assert config.static_size == 15
        assert config.input_len == 10  # unchanged

    def test_update_invalid_param(self):
        """Test that updating with invalid parameter raises error."""
        config = BaseConfig(
            input_len=10,
            output_len=5,
            input_size=3,
        )

        with pytest.raises(ValueError, match="Unknown parameter 'invalid_param'"):
            config.update(invalid_param=42)

    def test_unknown_kwargs_raises_error(self):
        """Test that unknown kwargs raise ValueError."""
        with pytest.raises(ValueError, match="Unknown parameter 'dropout'"):
            BaseConfig(
                input_len=10,
                output_len=5,
                input_size=3,
                dropout=0.1,  # Not a base parameter
            )


class TestDerivedConfig:
    """Test that derived configs work properly with BaseConfig."""

    def test_naive_config_inherits_properly(self):
        """Test that NaiveLastValueConfig properly inherits from BaseConfig."""
        config = NaiveLastValueConfig(
            input_len=20,
            output_len=7,
            input_size=4,
        )

        # Check base parameters
        assert config.input_len == 20
        assert config.output_len == 7
        assert config.input_size == 4

        # Check that it has specific defaults
        assert config.learning_rate == 1e-10  # NaiveLastValue specific default
        assert config.use_rev_in is False  # NaiveLastValue specific default

    def test_naive_config_from_dict(self):
        """Test creating NaiveLastValueConfig from dictionary."""
        config_dict = {
            "input_len": 30,
            "output_len": 10,
            "input_size": 5,
        }

        config = NaiveLastValueConfig.from_dict(config_dict)

        assert isinstance(config, NaiveLastValueConfig)
        assert config.input_len == 30
        assert config.output_len == 10
        assert config.input_size == 5

    def test_naive_config_validation(self):
        """Test NaiveLastValueConfig parameter validation."""
        # Test invalid input_size
        with pytest.raises(ValueError, match="input_size must be at least 1"):
            NaiveLastValueConfig(
                input_len=10,
                output_len=5,
                input_size=0,
            )

        # Test invalid output_len
        with pytest.raises(ValueError, match="output_len must be at least 1"):
            NaiveLastValueConfig(
                input_len=10,
                output_len=0,
                input_size=3,
            )
