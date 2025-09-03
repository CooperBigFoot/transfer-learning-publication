import numpy as np
import pytest

from transfer_learning_publication.transforms import ZScore


@pytest.fixture
def z_score():
    return ZScore()


@pytest.fixture
def standard_data():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)


@pytest.fixture
def constant_column_data():
    return np.array([[1, 5], [2, 5], [3, 5]], dtype=float)


class TestZScore:
    def test_fit_computes_statistics(self, z_score, standard_data):
        z_score.fit(standard_data)
        expected_mean = np.array([4, 5, 6])
        expected_std = np.array([np.std([1, 4, 7]), np.std([2, 5, 8]), np.std([3, 6, 9])])
        np.testing.assert_array_almost_equal(z_score._fitted_state["mean"], expected_mean)
        np.testing.assert_array_almost_equal(z_score._fitted_state["std"], expected_std)

    def test_transform_standardizes_data(self, z_score, standard_data):
        z_score.fit(standard_data)
        transformed = z_score.transform(standard_data)
        np.testing.assert_array_almost_equal(transformed.mean(axis=0), np.zeros(3))
        np.testing.assert_array_almost_equal(transformed.std(axis=0), np.ones(3))

    def test_inverse_transform_restores_data(self, z_score, standard_data):
        z_score.fit(standard_data)
        transformed = z_score.transform(standard_data)
        restored = z_score.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(restored, standard_data)

    def test_zero_std_handling(self, z_score, constant_column_data):
        z_score.fit(constant_column_data)
        assert z_score._fitted_state["std"][1] == 0
        transformed = z_score.transform(constant_column_data)
        assert np.all(transformed[:, 1] == 0)
        restored = z_score.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(restored, constant_column_data)

    def test_single_row_data(self, z_score):
        single_row = np.array([[1, 2, 3]], dtype=float)
        z_score.fit(single_row)
        assert np.all(z_score._fitted_state["std"] == 0)
        transformed = z_score.transform(single_row)
        assert np.all(transformed == 0)

    def test_different_test_data(self, z_score, standard_data):
        z_score.fit(standard_data)
        test_data = np.array([[10, 11, 12], [13, 14, 15]], dtype=float)
        transformed = z_score.transform(test_data)
        restored = z_score.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(restored, test_data)

    @pytest.mark.parametrize(
        "data",
        [
            np.array([[1e-10, 2e-10], [3e-10, 4e-10]]),
            np.array([[1e10, 2e10], [3e10, 4e10]]),
            np.array([[-100, -200], [-300, -400]]),
        ],
    )
    def test_extreme_values(self, z_score, data):
        z_score.fit(data)
        transformed = z_score.transform(data)
        restored = z_score.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(restored, data, decimal=5)

    def test_nan_handling(self, z_score):
        data_with_nan = np.array([[1, 2], [3, np.nan], [5, 6]], dtype=float)
        z_score.fit(data_with_nan)
        assert np.isnan(z_score._fitted_state["mean"][1])
        assert np.isnan(z_score._fitted_state["std"][1])

    def test_empty_columns(self, z_score):
        data = np.array([[1], [2], [3]], dtype=float)
        z_score.fit(data)
        transformed = z_score.transform(data)
        assert transformed.shape == data.shape
        np.testing.assert_array_almost_equal(transformed.mean(axis=0), [0])
        np.testing.assert_array_almost_equal(transformed.std(axis=0), [1])

    def test_multi_column_mixed_edge_cases(self, z_score):
        # Column 0: normal data
        # Column 1: constant values (zero std)
        # Column 2: contains NaN
        # Column 3: mix of zeros and ones
        mixed_data = np.array(
            [
                [1.0, 5.0, 2.0, 0.0],
                [2.0, 5.0, np.nan, 1.0],
                [3.0, 5.0, 4.0, 0.0],
                [4.0, 5.0, 6.0, 1.0],
                [5.0, 5.0, 8.0, 0.0],
            ],
            dtype=float,
        )

        z_score.fit(mixed_data)

        # Check fitted statistics
        assert z_score._fitted_state["std"][1] == 0  # Constant column
        assert np.isnan(z_score._fitted_state["mean"][2])  # NaN column
        assert np.isnan(z_score._fitted_state["std"][2])  # NaN column

        transformed = z_score.transform(mixed_data)

        # Column 0: should be properly normalized
        col0_no_nan = transformed[:, 0]
        np.testing.assert_array_almost_equal(col0_no_nan.mean(), 0)
        np.testing.assert_array_almost_equal(col0_no_nan.std(), 1)

        # Column 1: constant column should be all zeros after transform
        assert np.all(transformed[:, 1] == 0)

        # Column 2: NaN should propagate
        assert np.isnan(transformed[1, 2])
        assert np.all(np.isnan(transformed[[0, 2, 3, 4], 2]))

        # Column 3: binary data should be transformed correctly
        col3_mean = mixed_data[:, 3].mean()
        col3_std = mixed_data[:, 3].std()
        expected_col3 = (mixed_data[:, 3] - col3_mean) / col3_std
        np.testing.assert_array_almost_equal(transformed[:, 3], expected_col3)

        # Test inverse transform preserves all edge cases
        restored = z_score.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(restored[:, 0], mixed_data[:, 0])
        np.testing.assert_array_almost_equal(restored[:, 1], mixed_data[:, 1])
        np.testing.assert_array_almost_equal(restored[:, 3], mixed_data[:, 3])
        assert np.isnan(restored[1, 2])
