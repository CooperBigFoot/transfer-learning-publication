import numpy as np
import pytest

from transfer_learning_publication.transforms import Log


@pytest.fixture
def log_transform():
    return Log()


@pytest.fixture
def log_transform_custom():
    return Log(epsilon=1e-5)


@pytest.fixture
def positive_data():
    return np.array([[1, 2], [3, 4], [5, 6]], dtype=float)


@pytest.fixture
def zero_data():
    return np.array([[0, 1], [0, 2], [0, 3]], dtype=float)


class TestLogScale:
    def test_initialization_default_epsilon(self, log_transform):
        assert log_transform.epsilon == 1e-8

    def test_initialization_custom_epsilon(self, log_transform_custom):
        assert log_transform_custom.epsilon == 1e-5

    def test_fit_returns_empty_state(self, log_transform, positive_data):
        log_transform.fit(positive_data)
        assert log_transform._fitted_state == {}
        assert log_transform._is_fitted is True

    def test_transform_positive_values(self, log_transform, positive_data):
        log_transform.fit(positive_data)
        transformed = log_transform.transform(positive_data)
        expected = np.log(positive_data + log_transform.epsilon)
        np.testing.assert_array_almost_equal(transformed, expected)

    def test_inverse_transform_restores_values(self, log_transform, positive_data):
        log_transform.fit(positive_data)
        transformed = log_transform.transform(positive_data)
        restored = log_transform.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(restored, positive_data)

    def test_zero_value_handling(self, log_transform, zero_data):
        log_transform.fit(zero_data)
        transformed = log_transform.transform(zero_data)
        assert np.all(np.isfinite(transformed))
        expected_first_col = np.log(log_transform.epsilon)
        np.testing.assert_array_almost_equal(transformed[:, 0], np.full(3, expected_first_col))

    def test_negative_values(self, log_transform):
        negative_data = np.array([[-1, -2], [-3, -4]], dtype=float)
        log_transform.fit(negative_data)
        transformed = log_transform.transform(negative_data)
        assert np.all(np.isnan(transformed))

    def test_very_small_positive_values(self, log_transform):
        small_data = np.array([[1e-10, 1e-11], [1e-12, 1e-13]], dtype=float)
        log_transform.fit(small_data)
        transformed = log_transform.transform(small_data)
        assert np.all(np.isfinite(transformed))
        restored = log_transform.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(restored, small_data, decimal=15)

    def test_large_values(self, log_transform):
        large_data = np.array([[1e10, 1e20], [1e30, 1e40]], dtype=float)
        log_transform.fit(large_data)
        transformed = log_transform.transform(large_data)
        assert np.all(np.isfinite(transformed))
        restored = log_transform.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(restored / large_data, np.ones_like(large_data))

    def test_custom_epsilon_effect(self, log_transform_custom):
        data = np.array([[0, 0], [0, 0]], dtype=float)
        log_transform_custom.fit(data)
        transformed = log_transform_custom.transform(data)
        expected = np.log(1e-5)
        np.testing.assert_array_almost_equal(transformed, np.full_like(data, expected))

    @pytest.mark.parametrize("epsilon", [1e-3, 1e-6, 1e-9, 1e-12])
    def test_different_epsilons(self, epsilon, zero_data):
        log_t = Log(epsilon=epsilon)
        log_t.fit(zero_data)
        transformed = log_t.transform(zero_data)
        restored = log_t.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(restored[:, 1:], zero_data[:, 1:])
        np.testing.assert_array_almost_equal(restored[:, 0], np.zeros(3), decimal=10)

    def test_mixed_values(self, log_transform):
        mixed_data = np.array([[0, 1], [1e-10, 100], [1000, 0.001]], dtype=float)
        log_transform.fit(mixed_data)
        transformed = log_transform.transform(mixed_data)
        assert np.all(np.isfinite(transformed))
        restored = log_transform.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(restored, mixed_data, decimal=8)

    def test_transform_different_shape(self, log_transform, positive_data):
        log_transform.fit(positive_data)
        test_data = np.array([[10], [20]], dtype=float)
        transformed = log_transform.transform(test_data)
        assert transformed.shape == test_data.shape
        expected = np.log(test_data + log_transform.epsilon)
        np.testing.assert_array_almost_equal(transformed, expected)
