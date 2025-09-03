import numpy as np
import pytest

from transfer_learning_publication.transforms import BaseTransform


class DummyTransform(BaseTransform):
    """Concrete implementation for testing abstract base class."""

    def _fit(self, X: np.ndarray) -> dict:
        return {"dummy": "state"}

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return X * 2

    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X / 2


@pytest.fixture
def dummy_transform():
    return DummyTransform()


@pytest.fixture
def sample_data():
    return np.array([[1, 2], [3, 4]], dtype=float)


class TestBaseTransform:
    def test_initialization(self, dummy_transform):
        assert dummy_transform._is_fitted is False
        assert dummy_transform._fitted_state == {}

    def test_fit_sets_state(self, dummy_transform, sample_data):
        result = dummy_transform.fit(sample_data)
        assert result is dummy_transform
        assert dummy_transform._is_fitted is True
        assert dummy_transform._fitted_state == {"dummy": "state"}

    def test_transform_requires_fitting(self, dummy_transform, sample_data):
        with pytest.raises(RuntimeError, match="Transform must be fitted before transform"):
            dummy_transform.transform(sample_data)

    def test_inverse_transform_requires_fitting(self, dummy_transform, sample_data):
        with pytest.raises(RuntimeError, match="Transform must be fitted before inverse_transform"):
            dummy_transform.inverse_transform(sample_data)

    def test_transform_after_fit(self, dummy_transform, sample_data):
        dummy_transform.fit(sample_data)
        result = dummy_transform.transform(sample_data)
        np.testing.assert_array_equal(result, sample_data * 2)

    def test_inverse_transform_after_fit(self, dummy_transform, sample_data):
        dummy_transform.fit(sample_data)
        result = dummy_transform.inverse_transform(sample_data)
        np.testing.assert_array_equal(result, sample_data / 2)

    @pytest.mark.parametrize(
        "invalid_input,error_type,error_msg",
        [
            ([1, 2, 3], TypeError, "Input must be a numpy array"),
            ([[1, 2], [3, 4]], TypeError, "Input must be a numpy array"),
            (np.array([1, 2, 3]), ValueError, "Input must be 2D array, got 1D"),
            (np.array([[[1, 2]]]), ValueError, "Input must be 2D array, got 3D"),
        ],
    )
    def test_validate_input_errors(self, dummy_transform, invalid_input, error_type, error_msg):
        with pytest.raises(error_type, match=error_msg):
            dummy_transform.fit(invalid_input)

    def test_abstract_methods_not_implemented(self):
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseTransform()

    def test_fit_transform_pipeline(self, dummy_transform, sample_data):
        transformed = dummy_transform.fit(sample_data).transform(sample_data)
        inversed = dummy_transform.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(inversed, sample_data)
