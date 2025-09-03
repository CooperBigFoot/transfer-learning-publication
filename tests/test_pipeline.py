import warnings
from typing import Any

import numpy as np
import pytest

from transfer_learning_publication.transforms import BaseTransform
from transfer_learning_publication.transforms.pipeline import GlobalPipeline, PerBasinPipeline


class MockTransform(BaseTransform):
    """Mock transform that multiplies by a factor for testing."""

    def __init__(self, factor: float = 2.0):
        super().__init__()
        self.factor = factor

    def _fit(self, X: np.ndarray) -> dict[str, Any]:
        return {"factor": self.factor, "input_shape": X.shape}

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return X * self._fitted_state["factor"]

    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X / self._fitted_state["factor"]


class MockAdditiveTransform(BaseTransform):
    """Mock transform that adds a constant for testing chaining."""

    def __init__(self, offset: float = 1.0):
        super().__init__()
        self.offset = offset

    def _fit(self, X: np.ndarray) -> dict[str, Any]:
        return {"offset": self.offset, "mean": np.mean(X)}

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return X + self._fitted_state["offset"]

    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X - self._fitted_state["offset"]


@pytest.fixture
def sample_data():
    """Sample data with 2 groups and 2 features."""
    return np.array(
        [
            [1.0, 2.0, 0],  # Group 0
            [3.0, 4.0, 0],  # Group 0
            [5.0, 6.0, 1],  # Group 1
            [7.0, 8.0, 1],  # Group 1
        ]
    )


@pytest.fixture
def single_group_data():
    """Sample data with only one group."""
    return np.array(
        [
            [1.0, 2.0, 0],
            [3.0, 4.0, 0],
            [5.0, 6.0, 0],
        ]
    )


@pytest.fixture
def single_transform():
    """Single mock transform."""
    return MockTransform(factor=2.0)


@pytest.fixture
def multiple_transforms():
    """Multiple transforms for testing chaining."""
    return [MockTransform(factor=2.0), MockAdditiveTransform(offset=1.0)]


@pytest.fixture
def pipeline_single(single_transform):
    """Pipeline with single transform."""
    return PerBasinPipeline([single_transform])


@pytest.fixture
def pipeline_multiple(multiple_transforms):
    """Pipeline with multiple transforms."""
    return PerBasinPipeline(multiple_transforms)


@pytest.fixture
def global_pipeline_single(single_transform):
    """Global pipeline with single transform."""
    return GlobalPipeline([single_transform])


@pytest.fixture
def global_pipeline_multiple(multiple_transforms):
    """Global pipeline with multiple transforms."""
    return GlobalPipeline(multiple_transforms)


class TestPerBasinPipelineInit:
    """Test pipeline initialization."""

    def test_init_with_single_transform(self, single_transform):
        pipeline = PerBasinPipeline([single_transform])
        assert len(pipeline.transforms) == 1
        assert pipeline.transforms[0] is single_transform
        assert pipeline.fitted_pipelines == {}
        assert pipeline._is_fitted is False

    def test_init_with_multiple_transforms(self, multiple_transforms):
        pipeline = PerBasinPipeline(multiple_transforms)
        assert len(pipeline.transforms) == 2
        assert pipeline.transforms == multiple_transforms
        assert pipeline.fitted_pipelines == {}
        assert pipeline._is_fitted is False

    def test_init_with_empty_list(self):
        pipeline = PerBasinPipeline([])
        assert pipeline.transforms == []
        assert pipeline.fitted_pipelines == {}
        assert pipeline._is_fitted is False


class TestPerBasinPipelineValidation:
    """Test input validation."""

    @pytest.mark.parametrize(
        "invalid_input,error_type,error_msg",
        [
            ([1, 2, 3], TypeError, "Input must be a numpy array"),
            ([[1, 2, 3]], TypeError, "Input must be a numpy array"),
            (np.array([1, 2, 3]), ValueError, "Input must be 2D array, got 1D"),
            (np.array([[[1, 2]]]), ValueError, "Input must be 2D array, got 3D"),
            (np.array([[1]]), ValueError, "Input must have at least 2 columns"),
            (np.array([[]]), ValueError, "Input must have at least 2 columns"),
        ],
    )
    def test_validate_input_errors(self, pipeline_single, invalid_input, error_type, error_msg):
        with pytest.raises(error_type, match=error_msg):
            pipeline_single.fit(invalid_input)


class TestPerBasinPipelineFit:
    """Test fitting functionality."""

    def test_fit_single_transform(self, pipeline_single, sample_data):
        result = pipeline_single.fit(sample_data)

        # Should return self for method chaining
        assert result is pipeline_single
        assert pipeline_single._is_fitted is True

        # Should have fitted pipelines for each group
        assert len(pipeline_single.fitted_pipelines) == 2
        assert 0 in pipeline_single.fitted_pipelines
        assert 1 in pipeline_single.fitted_pipelines

        # Each group should have its own copy of transforms
        group_0_transforms = pipeline_single.fitted_pipelines[0]
        group_1_transforms = pipeline_single.fitted_pipelines[1]

        assert len(group_0_transforms) == 1
        assert len(group_1_transforms) == 1
        assert group_0_transforms[0] is not group_1_transforms[0]  # Different instances
        assert group_0_transforms[0]._is_fitted is True
        assert group_1_transforms[0]._is_fitted is True

    def test_fit_multiple_transforms(self, pipeline_multiple, sample_data):
        pipeline_multiple.fit(sample_data)

        assert pipeline_multiple._is_fitted is True
        assert len(pipeline_multiple.fitted_pipelines) == 2

        # Each group should have chain of transforms
        for group_id in [0, 1]:
            group_transforms = pipeline_multiple.fitted_pipelines[group_id]
            assert len(group_transforms) == 2
            assert all(t._is_fitted for t in group_transforms)

    def test_fit_single_group(self, pipeline_single, single_group_data):
        pipeline_single.fit(single_group_data)

        assert pipeline_single._is_fitted is True
        assert len(pipeline_single.fitted_pipelines) == 1
        assert 0 in pipeline_single.fitted_pipelines

    def test_fit_preserves_group_column(self, pipeline_single, sample_data):
        original_groups = sample_data[:, -1].copy()
        pipeline_single.fit(sample_data)

        # Original data should be unchanged
        np.testing.assert_array_equal(sample_data[:, -1], original_groups)

    def test_get_unique_groups(self, pipeline_single):
        data = np.array([[1, 2, "a"], [3, 4, "b"], [5, 6, "a"]], dtype=object)
        groups = pipeline_single._get_unique_groups(data)
        expected = np.array(["a", "b"])
        np.testing.assert_array_equal(np.sort(groups), np.sort(expected))


class TestPerBasinPipelineTransform:
    """Test transform functionality."""

    def test_transform_not_fitted_error(self, pipeline_single, sample_data):
        with pytest.raises(RuntimeError, match="Pipeline must be fitted before transform"):
            pipeline_single.transform(sample_data)

    def test_transform_single_transform(self, pipeline_single, sample_data):
        pipeline_single.fit(sample_data)
        result = pipeline_single.transform(sample_data)

        # Should have same shape
        assert result.shape == sample_data.shape

        # Group column should be unchanged
        np.testing.assert_array_equal(result[:, -1], sample_data[:, -1])

        # Features should be transformed (multiplied by 2)
        expected_features = sample_data[:, :-1] * 2
        np.testing.assert_array_equal(result[:, :-1], expected_features)

    def test_transform_multiple_transforms(self, pipeline_multiple, sample_data):
        pipeline_multiple.fit(sample_data)
        result = pipeline_multiple.transform(sample_data)

        # Features should be: (original * 2) + 1
        expected_features = (sample_data[:, :-1] * 2) + 1
        np.testing.assert_array_equal(result[:, :-1], expected_features)
        np.testing.assert_array_equal(result[:, -1], sample_data[:, -1])

    def test_transform_unseen_groups_warning(self, pipeline_single, sample_data):
        # Fit on subset of groups
        train_data = sample_data[sample_data[:, -1] == 0]  # Only group 0
        pipeline_single.fit(train_data)

        # Transform data with both groups
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pipeline_single.transform(sample_data)

            # Should issue warning about unseen groups
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
            assert "were not seen during fit and will not be transformed" in str(w[0].message)

        # Unseen group features should be zero
        group_1_mask = sample_data[:, -1] == 1
        np.testing.assert_array_equal(result[group_1_mask, :-1], 0)

        # Seen group should be transformed
        group_0_mask = sample_data[:, -1] == 0
        expected_group_0 = sample_data[group_0_mask, :-1] * 2
        np.testing.assert_array_equal(result[group_0_mask, :-1], expected_group_0)


class TestPerBasinPipelineInverseTransform:
    """Test inverse transform functionality."""

    def test_inverse_transform_not_fitted_error(self, pipeline_single, sample_data):
        with pytest.raises(RuntimeError, match="Pipeline must be fitted before inverse_transform"):
            pipeline_single.inverse_transform(sample_data)

    def test_inverse_transform_single_transform(self, pipeline_single, sample_data):
        pipeline_single.fit(sample_data)
        transformed = pipeline_single.transform(sample_data)
        result = pipeline_single.inverse_transform(transformed)

        # Should recover original data
        np.testing.assert_array_almost_equal(result, sample_data)

    def test_inverse_transform_multiple_transforms(self, pipeline_multiple, sample_data):
        pipeline_multiple.fit(sample_data)
        transformed = pipeline_multiple.transform(sample_data)
        result = pipeline_multiple.inverse_transform(transformed)

        # Should recover original data
        np.testing.assert_array_almost_equal(result, sample_data)

    def test_inverse_transform_reverse_order(self, pipeline_multiple, sample_data):
        """Test that inverse transforms are applied in reverse order."""
        pipeline_multiple.fit(sample_data)

        # Get transforms for group 0
        group_transforms = pipeline_multiple.fitted_pipelines[0]

        # Manually apply transforms in forward order
        group_0_data = sample_data[sample_data[:, -1] == 0, :-1]
        forward_step1 = group_transforms[0].transform(group_0_data)  # *2
        forward_step2 = group_transforms[1].transform(forward_step1)  # +1

        # Apply inverse through pipeline (should be in reverse order)
        test_data = np.column_stack([forward_step2, np.zeros(len(forward_step2))])
        result = pipeline_multiple.inverse_transform(test_data)

        expected = np.column_stack([group_0_data, np.zeros(len(group_0_data))])
        np.testing.assert_array_almost_equal(result, expected)

    def test_inverse_transform_unseen_groups_warning(self, pipeline_single, sample_data):
        # Fit on subset of groups
        train_data = sample_data[sample_data[:, -1] == 0]
        pipeline_single.fit(train_data)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipeline_single.inverse_transform(sample_data)

            assert len(w) == 1
            assert "were not seen during fit and will not be inverse transformed" in str(w[0].message)


class TestPerBasinPipelineGroupAccess:
    """Test group pipeline access."""

    def test_get_group_pipeline_not_fitted_error(self, pipeline_single):
        with pytest.raises(RuntimeError, match="Pipeline must be fitted before accessing group pipelines"):
            pipeline_single.get_group_pipeline(0)

    def test_get_group_pipeline_valid_group(self, pipeline_single, sample_data):
        pipeline_single.fit(sample_data)

        group_transforms = pipeline_single.get_group_pipeline(0)
        assert len(group_transforms) == 1
        assert isinstance(group_transforms[0], MockTransform)
        assert group_transforms[0]._is_fitted is True

    def test_get_group_pipeline_invalid_group(self, pipeline_single, sample_data):
        pipeline_single.fit(sample_data)

        with pytest.raises(ValueError, match="Group 999 not found"):
            pipeline_single.get_group_pipeline(999)


class TestPerBasinPipelineEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_transforms_list(self, sample_data):
        pipeline = PerBasinPipeline([])
        pipeline.fit(sample_data)

        # Should still work, just no transformation
        result = pipeline.transform(sample_data)
        np.testing.assert_array_equal(result, sample_data)

    def test_clone_transforms(self, pipeline_multiple):
        original_transforms = pipeline_multiple.transforms
        cloned = pipeline_multiple._clone_transforms()

        # Should be different instances
        assert len(cloned) == len(original_transforms)
        for orig, clone in zip(original_transforms, cloned, strict=False):
            assert orig is not clone
            assert type(orig) is type(clone)
            assert orig.factor == clone.factor if hasattr(orig, "factor") else True

    def test_different_group_types(self, pipeline_single):
        """Test with string group identifiers."""
        data = np.array(
            [
                [1.0, 2.0, "A"],
                [3.0, 4.0, "A"],
                [5.0, 6.0, "B"],
            ],
            dtype=object,
        )

        pipeline_single.fit(data)
        result = pipeline_single.transform(data)

        assert pipeline_single._is_fitted is True
        assert "A" in pipeline_single.fitted_pipelines
        assert "B" in pipeline_single.fitted_pipelines
        np.testing.assert_array_equal(result[:, -1], data[:, -1])

    def test_single_sample_per_group(self, pipeline_single):
        """Test with only one sample per group."""
        data = np.array(
            [
                [1.0, 2.0, 0],
                [3.0, 4.0, 1],
            ]
        )

        pipeline_single.fit(data)
        result = pipeline_single.transform(data)

        expected_features = data[:, :-1] * 2
        np.testing.assert_array_equal(result[:, :-1], expected_features)
        np.testing.assert_array_equal(result[:, -1], data[:, -1])

    def test_many_features(self, pipeline_single):
        """Test with many features."""
        data = np.random.rand(10, 6)  # 5 features + 1 group
        data[:, -1] = np.repeat([0, 1], 5)  # Two groups

        pipeline_single.fit(data)
        result = pipeline_single.transform(data)

        assert result.shape == data.shape
        np.testing.assert_array_equal(result[:, -1], data[:, -1])
        # Features should be doubled
        np.testing.assert_array_equal(result[:, :-1], data[:, :-1] * 2)


# GlobalPipeline Tests


class TestGlobalPipelineInit:
    """Test global pipeline initialization."""

    def test_init_with_single_transform(self, single_transform):
        pipeline = GlobalPipeline([single_transform])
        assert len(pipeline.transforms) == 1
        assert pipeline.transforms[0] is single_transform
        assert pipeline.fitted_transforms == []
        assert pipeline._is_fitted is False

    def test_init_with_multiple_transforms(self, multiple_transforms):
        pipeline = GlobalPipeline(multiple_transforms)
        assert len(pipeline.transforms) == 2
        assert pipeline.transforms == multiple_transforms
        assert pipeline.fitted_transforms == []
        assert pipeline._is_fitted is False

    def test_init_with_empty_list(self):
        pipeline = GlobalPipeline([])
        assert pipeline.transforms == []
        assert pipeline.fitted_transforms == []
        assert pipeline._is_fitted is False


class TestGlobalPipelineValidation:
    """Test input validation."""

    @pytest.mark.parametrize(
        "invalid_input,error_type,error_msg",
        [
            ([1, 2, 3], TypeError, "Input must be a numpy array"),
            ([[1, 2, 3]], TypeError, "Input must be a numpy array"),
            (np.array([1, 2, 3]), ValueError, "Input must be 2D array, got 1D"),
            (np.array([[[1, 2]]]), ValueError, "Input must be 2D array, got 3D"),
            (np.array([[1]]), ValueError, "Input must have at least 2 columns"),
            (np.array([[]]), ValueError, "Input must have at least 2 columns"),
        ],
    )
    def test_validate_input_errors(self, global_pipeline_single, invalid_input, error_type, error_msg):
        with pytest.raises(error_type, match=error_msg):
            global_pipeline_single.fit(invalid_input)


class TestGlobalPipelineFit:
    """Test fitting functionality."""

    def test_fit_single_transform(self, global_pipeline_single, sample_data):
        result = global_pipeline_single.fit(sample_data)

        # Should return self for method chaining
        assert result is global_pipeline_single
        assert global_pipeline_single._is_fitted is True

        # Should have fitted transforms (globally fitted, not per-group)
        assert len(global_pipeline_single.fitted_transforms) == 1
        assert global_pipeline_single.fitted_transforms[0]._is_fitted is True
        # Should be different instance from original
        assert global_pipeline_single.fitted_transforms[0] is not global_pipeline_single.transforms[0]

    def test_fit_multiple_transforms(self, global_pipeline_multiple, sample_data):
        global_pipeline_multiple.fit(sample_data)

        assert global_pipeline_multiple._is_fitted is True
        assert len(global_pipeline_multiple.fitted_transforms) == 2

        # All transforms should be fitted
        for transform in global_pipeline_multiple.fitted_transforms:
            assert transform._is_fitted is True

    def test_fit_single_group(self, global_pipeline_single, single_group_data):
        global_pipeline_single.fit(single_group_data)

        assert global_pipeline_single._is_fitted is True
        assert len(global_pipeline_single.fitted_transforms) == 1
        assert global_pipeline_single.fitted_transforms[0]._is_fitted is True

    def test_fit_preserves_group_column(self, global_pipeline_single, sample_data):
        original_groups = sample_data[:, -1].copy()
        global_pipeline_single.fit(sample_data)

        # Original data should be unchanged
        np.testing.assert_array_equal(sample_data[:, -1], original_groups)

    def test_fit_pools_data_globally(self, global_pipeline_single, sample_data):
        """Test that fitting pools data from all groups together."""
        global_pipeline_single.fit(sample_data)

        # Get the fitted transform
        fitted_transform = global_pipeline_single.fitted_transforms[0]

        # The input shape should reflect all samples combined (4 samples, 2 features)
        assert fitted_transform._fitted_state["input_shape"] == (4, 2)

        # The factor should be the same as the original transform
        assert fitted_transform._fitted_state["factor"] == 2.0


class TestGlobalPipelineTransform:
    """Test transform functionality."""

    def test_transform_not_fitted_error(self, global_pipeline_single, sample_data):
        with pytest.raises(RuntimeError, match="Pipeline must be fitted before transform"):
            global_pipeline_single.transform(sample_data)

    def test_transform_single_transform(self, global_pipeline_single, sample_data):
        global_pipeline_single.fit(sample_data)
        result = global_pipeline_single.transform(sample_data)

        # Should have same shape
        assert result.shape == sample_data.shape

        # Group column should be unchanged
        np.testing.assert_array_equal(result[:, -1], sample_data[:, -1])

        # Features should be transformed (multiplied by 2)
        expected_features = sample_data[:, :-1] * 2
        np.testing.assert_array_equal(result[:, :-1], expected_features)

    def test_transform_multiple_transforms(self, global_pipeline_multiple, sample_data):
        global_pipeline_multiple.fit(sample_data)
        result = global_pipeline_multiple.transform(sample_data)

        # Features should be: (original * 2) + 1
        expected_features = (sample_data[:, :-1] * 2) + 1
        np.testing.assert_array_equal(result[:, :-1], expected_features)
        np.testing.assert_array_equal(result[:, -1], sample_data[:, -1])

    def test_transform_global_behavior(self, global_pipeline_single, sample_data):
        """Test that transform applies same transformation to all groups."""
        global_pipeline_single.fit(sample_data)
        result = global_pipeline_single.transform(sample_data)

        # All groups should get the same transformation
        # Group 0 features: [1,2], [3,4] -> [2,4], [6,8]
        # Group 1 features: [5,6], [7,8] -> [10,12], [14,16]
        expected_group_0 = np.array([[2.0, 4.0], [6.0, 8.0]])
        expected_group_1 = np.array([[10.0, 12.0], [14.0, 16.0]])

        group_0_mask = sample_data[:, -1] == 0
        group_1_mask = sample_data[:, -1] == 1

        np.testing.assert_array_equal(result[group_0_mask, :-1], expected_group_0)
        np.testing.assert_array_equal(result[group_1_mask, :-1], expected_group_1)

    def test_transform_new_groups(self, global_pipeline_single, sample_data):
        """Test transform with groups not seen during fit."""
        global_pipeline_single.fit(sample_data)

        # Create test data with new groups
        new_data = np.array(
            [
                [10.0, 20.0, 2],  # Group 2 (new)
                [30.0, 40.0, 3],  # Group 3 (new)
            ]
        )

        result = global_pipeline_single.transform(new_data)

        # Should apply same global transform to new groups
        expected_features = new_data[:, :-1] * 2  # multiply by 2
        np.testing.assert_array_equal(result[:, :-1], expected_features)
        np.testing.assert_array_equal(result[:, -1], new_data[:, -1])


class TestGlobalPipelineInverseTransform:
    """Test inverse transform functionality."""

    def test_inverse_transform_not_fitted_error(self, global_pipeline_single, sample_data):
        with pytest.raises(RuntimeError, match="Pipeline must be fitted before inverse_transform"):
            global_pipeline_single.inverse_transform(sample_data)

    def test_inverse_transform_single_transform(self, global_pipeline_single, sample_data):
        global_pipeline_single.fit(sample_data)
        transformed = global_pipeline_single.transform(sample_data)
        result = global_pipeline_single.inverse_transform(transformed)

        # Should recover original data
        np.testing.assert_array_almost_equal(result, sample_data)

    def test_inverse_transform_multiple_transforms(self, global_pipeline_multiple, sample_data):
        global_pipeline_multiple.fit(sample_data)
        transformed = global_pipeline_multiple.transform(sample_data)
        result = global_pipeline_multiple.inverse_transform(transformed)

        # Should recover original data
        np.testing.assert_array_almost_equal(result, sample_data)

    def test_inverse_transform_reverse_order(self, global_pipeline_multiple, sample_data):
        """Test that inverse transforms are applied in reverse order."""
        global_pipeline_multiple.fit(sample_data)

        # Get fitted transforms
        fitted_transforms = global_pipeline_multiple.fitted_transforms

        # Manually apply transforms in forward order
        features = sample_data[:, :-1]
        forward_step1 = fitted_transforms[0].transform(features)  # *2
        forward_step2 = fitted_transforms[1].transform(forward_step1)  # +1

        # Apply inverse through pipeline (should be in reverse order)
        test_data = np.column_stack([forward_step2, sample_data[:, -1]])
        result = global_pipeline_multiple.inverse_transform(test_data)

        np.testing.assert_array_almost_equal(result, sample_data)

    def test_inverse_transform_new_groups(self, global_pipeline_single, sample_data):
        """Test inverse transform with groups not seen during fit."""
        global_pipeline_single.fit(sample_data)

        # Create transformed data with new groups
        new_data = np.array(
            [
                [20.0, 40.0, 2],  # Group 2 (new), features are 2x original
                [60.0, 80.0, 3],  # Group 3 (new), features are 2x original
            ]
        )

        result = global_pipeline_single.inverse_transform(new_data)

        # Should apply same global inverse transform to new groups
        expected_features = new_data[:, :-1] / 2  # divide by 2
        np.testing.assert_array_equal(result[:, :-1], expected_features)
        np.testing.assert_array_equal(result[:, -1], new_data[:, -1])


class TestGlobalPipelineAccess:
    """Test fitted transforms access."""

    def test_get_fitted_transforms_not_fitted_error(self, global_pipeline_single):
        with pytest.raises(RuntimeError, match="Pipeline must be fitted before accessing fitted transforms"):
            global_pipeline_single.get_fitted_transforms()

    def test_get_fitted_transforms_single_transform(self, global_pipeline_single, sample_data):
        global_pipeline_single.fit(sample_data)

        fitted_transforms = global_pipeline_single.get_fitted_transforms()
        assert len(fitted_transforms) == 1
        assert isinstance(fitted_transforms[0], MockTransform)
        assert fitted_transforms[0]._is_fitted is True

    def test_get_fitted_transforms_multiple_transforms(self, global_pipeline_multiple, sample_data):
        global_pipeline_multiple.fit(sample_data)

        fitted_transforms = global_pipeline_multiple.get_fitted_transforms()
        assert len(fitted_transforms) == 2
        assert isinstance(fitted_transforms[0], MockTransform)
        assert isinstance(fitted_transforms[1], MockAdditiveTransform)
        assert all(t._is_fitted for t in fitted_transforms)

    def test_get_fitted_transforms_returns_references(self, global_pipeline_single, sample_data):
        """Test that get_fitted_transforms returns the actual fitted transforms."""
        global_pipeline_single.fit(sample_data)

        fitted_transforms = global_pipeline_single.get_fitted_transforms()
        # Should be the same objects as stored internally
        assert fitted_transforms is global_pipeline_single.fitted_transforms


class TestGlobalPipelineEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_transforms_list(self, sample_data):
        pipeline = GlobalPipeline([])
        pipeline.fit(sample_data)

        # Should still work, just no transformation
        result = pipeline.transform(sample_data)
        np.testing.assert_array_equal(result, sample_data)

    def test_clone_transforms(self, global_pipeline_multiple):
        original_transforms = global_pipeline_multiple.transforms
        cloned = global_pipeline_multiple._clone_transforms()

        # Should be different instances
        assert len(cloned) == len(original_transforms)
        for orig, clone in zip(original_transforms, cloned, strict=False):
            assert orig is not clone
            assert type(orig) is type(clone)
            assert orig.factor == clone.factor if hasattr(orig, "factor") else True

    def test_different_group_types(self, global_pipeline_single):
        """Test with string group identifiers."""
        data = np.array(
            [
                [1.0, 2.0, "A"],
                [3.0, 4.0, "A"],
                [5.0, 6.0, "B"],
            ],
            dtype=object,
        )

        global_pipeline_single.fit(data)
        result = global_pipeline_single.transform(data)

        assert global_pipeline_single._is_fitted is True
        np.testing.assert_array_equal(result[:, -1], data[:, -1])
        # Features should be doubled globally
        expected_features = data[:, :-1].astype(float) * 2
        np.testing.assert_array_equal(result[:, :-1].astype(float), expected_features)

    def test_single_sample_per_group(self, global_pipeline_single):
        """Test with only one sample per group."""
        data = np.array(
            [
                [1.0, 2.0, 0],
                [3.0, 4.0, 1],
            ]
        )

        global_pipeline_single.fit(data)
        result = global_pipeline_single.transform(data)

        expected_features = data[:, :-1] * 2
        np.testing.assert_array_equal(result[:, :-1], expected_features)
        np.testing.assert_array_equal(result[:, -1], data[:, -1])

    def test_many_features(self, global_pipeline_single):
        """Test with many features."""
        data = np.random.rand(10, 6)  # 5 features + 1 group
        data[:, -1] = np.repeat([0, 1], 5)  # Two groups

        global_pipeline_single.fit(data)
        result = global_pipeline_single.transform(data)

        assert result.shape == data.shape
        np.testing.assert_array_equal(result[:, -1], data[:, -1])
        # Features should be doubled
        np.testing.assert_array_equal(result[:, :-1], data[:, :-1] * 2)

    def test_round_trip_consistency(self, global_pipeline_multiple, sample_data):
        """Test that fit -> transform -> inverse_transform recovers original data."""
        global_pipeline_multiple.fit(sample_data)
        transformed = global_pipeline_multiple.transform(sample_data)
        recovered = global_pipeline_multiple.inverse_transform(transformed)

        np.testing.assert_array_almost_equal(recovered, sample_data)
