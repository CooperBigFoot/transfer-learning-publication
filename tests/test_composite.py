import tempfile
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest

from transfer_learning_publication.transforms import (
    CompositePipeline,
    CompositePipelineStep,
    Log,
    ZScore,
)


class TestCompositePipelineStep:
    """Test CompositePipelineStep dataclass."""

    def test_step_creation(self):
        """Test creating a pipeline step."""
        transforms = [Log(), ZScore()]
        step = CompositePipelineStep(
            pipeline_type="per_basin", transforms=transforms, columns=["streamflow", "precipitation"]
        )

        assert step.pipeline_type == "per_basin"
        assert len(step.transforms) == 2
        assert isinstance(step.transforms[0], Log)
        assert isinstance(step.transforms[1], ZScore)
        assert step.columns == ["streamflow", "precipitation"]


class TestCompositePipeline:
    """Test CompositePipeline orchestration functionality."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample test DataFrame."""
        return pl.DataFrame(
            {
                "basin_id": [1, 1, 1, 2, 2, 2],
                "streamflow": [10.0, 15.0, 12.0, 8.0, 11.0, 9.0],
                "precipitation": [2.0, 3.0, 2.5, 1.5, 2.8, 1.8],
                "temperature": [20.0, 22.0, 21.0, 18.0, 19.0, 17.0],
                "other_col": [1, 2, 3, 4, 5, 6],  # Should be preserved
            }
        )

    @pytest.fixture
    def simple_pipeline(self):
        """Create simple test pipeline."""
        steps = [CompositePipelineStep(pipeline_type="per_basin", transforms=[Log()], columns=["streamflow"])]
        return CompositePipeline(steps, group_identifier="basin_id")

    def test_pipeline_creation(self, simple_pipeline):
        """Test creating a composite pipeline."""
        assert simple_pipeline.group_identifier == "basin_id"
        assert len(simple_pipeline.steps) == 1
        assert not simple_pipeline._is_fitted

    def test_dataframe_validation(self, simple_pipeline):
        """Test DataFrame validation."""
        # Valid DataFrame
        valid_df = pl.DataFrame({"basin_id": [1, 2], "streamflow": [10.0, 15.0]})
        simple_pipeline._validate_dataframe(valid_df)  # Should not raise

        # Invalid input type
        with pytest.raises(TypeError, match="Input must be a Polars DataFrame"):
            simple_pipeline._validate_dataframe("not a dataframe")

        # Missing group identifier
        invalid_df = pl.DataFrame({"streamflow": [10.0, 15.0]})
        with pytest.raises(ValueError, match="Group identifier 'basin_id' not found"):
            simple_pipeline._validate_dataframe(invalid_df)

        # Empty DataFrame
        empty_df = pl.DataFrame({"basin_id": [], "streamflow": []})
        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            simple_pipeline._validate_dataframe(empty_df)

    def test_step_column_validation(self, simple_pipeline, sample_dataframe):
        """Test step column validation."""
        step = simple_pipeline.steps[0]

        # Valid columns
        simple_pipeline._validate_step_columns(sample_dataframe, step)  # Should not raise

        # Missing columns
        df_missing_cols = sample_dataframe.drop("streamflow")
        with pytest.raises(ValueError, match="Columns \\['streamflow'\\] not found"):
            simple_pipeline._validate_step_columns(df_missing_cols, step)

    def test_dataframe_to_numpy_conversion(self, simple_pipeline, sample_dataframe):
        """Test DataFrame to numpy conversion."""
        columns = ["streamflow", "precipitation"]
        array = simple_pipeline._dataframe_to_numpy(sample_dataframe, columns)

        # Check shape: 6 rows, 3 columns (2 features + group_id)
        assert array.shape == (6, 3)

        # Check group identifiers are in last column
        expected_groups = np.array([1, 1, 1, 2, 2, 2])
        np.testing.assert_array_equal(array[:, -1], expected_groups)

        # Check feature columns
        expected_streamflow = np.array([10.0, 15.0, 12.0, 8.0, 11.0, 9.0])
        expected_precipitation = np.array([2.0, 3.0, 2.5, 1.5, 2.8, 1.8])
        np.testing.assert_array_equal(array[:, 0], expected_streamflow)
        np.testing.assert_array_equal(array[:, 1], expected_precipitation)

    def test_numpy_to_dataframe_update(self, simple_pipeline, sample_dataframe):
        """Test updating DataFrame with transformed numpy values."""
        # Create mock transformed array
        transformed_array = np.array(
            [
                [1.0, 2.0, 1],  # transformed streamflow, precipitation, group_id
                [1.5, 2.5, 1],
                [1.2, 2.2, 1],
                [0.8, 1.8, 2],
                [1.1, 2.1, 2],
                [0.9, 1.9, 2],
            ]
        )

        columns = ["streamflow", "precipitation"]
        updated_df = simple_pipeline._numpy_to_dataframe_update(sample_dataframe, transformed_array, columns)

        # Check that specified columns were updated
        expected_streamflow = [1.0, 1.5, 1.2, 0.8, 1.1, 0.9]
        expected_precipitation = [2.0, 2.5, 2.2, 1.8, 2.1, 1.9]

        assert updated_df["streamflow"].to_list() == expected_streamflow
        assert updated_df["precipitation"].to_list() == expected_precipitation

        # Check that other columns were preserved
        assert updated_df["temperature"].to_list() == sample_dataframe["temperature"].to_list()
        assert updated_df["other_col"].to_list() == sample_dataframe["other_col"].to_list()
        assert updated_df["basin_id"].to_list() == sample_dataframe["basin_id"].to_list()

    def test_fit_single_step(self, simple_pipeline, sample_dataframe):
        """Test fitting pipeline with single step."""
        result = simple_pipeline.fit(sample_dataframe)

        assert result is simple_pipeline  # Test method chaining
        assert simple_pipeline._is_fitted
        assert len(simple_pipeline._fitted_steps) == 1

    def test_fit_multiple_steps(self, sample_dataframe):
        """Test fitting pipeline with multiple steps."""
        steps = [
            CompositePipelineStep(pipeline_type="per_basin", transforms=[Log()], columns=["streamflow"]),
            CompositePipelineStep(
                pipeline_type="global", transforms=[ZScore()], columns=["streamflow", "precipitation"]
            ),
        ]
        pipeline = CompositePipeline(steps, group_identifier="basin_id")

        pipeline.fit(sample_dataframe)
        assert pipeline._is_fitted
        assert len(pipeline._fitted_steps) == 2

    def test_transform_requires_fit(self, simple_pipeline, sample_dataframe):
        """Test transform requires pipeline to be fitted first."""
        with pytest.raises(RuntimeError, match="Pipeline must be fitted before transform"):
            simple_pipeline.transform(sample_dataframe)

    def test_transform_single_step(self, simple_pipeline, sample_dataframe):
        """Test transforming data with single step."""
        simple_pipeline.fit(sample_dataframe)
        transformed_df = simple_pipeline.transform(sample_dataframe)

        # Check output structure
        assert isinstance(transformed_df, pl.DataFrame)
        assert transformed_df.shape[0] == sample_dataframe.shape[0]
        assert transformed_df.columns == sample_dataframe.columns

        # Check that streamflow was log-transformed (should be positive, smaller values)
        original_streamflow = sample_dataframe["streamflow"].to_numpy()
        transformed_streamflow = transformed_df["streamflow"].to_numpy()

        # Log transform should make values smaller for values > 1
        assert np.all(transformed_streamflow < original_streamflow)

        # Check other columns unchanged
        for col in ["precipitation", "temperature", "other_col", "basin_id"]:
            np.testing.assert_array_equal(transformed_df[col].to_numpy(), sample_dataframe[col].to_numpy())

    def test_fit_transform(self, simple_pipeline, sample_dataframe):
        """Test fit_transform convenience method."""
        transformed_df = simple_pipeline.fit_transform(sample_dataframe)

        assert simple_pipeline._is_fitted
        assert isinstance(transformed_df, pl.DataFrame)
        assert transformed_df.shape == sample_dataframe.shape

    def test_inverse_transform_requires_fit(self, simple_pipeline, sample_dataframe):
        """Test inverse transform requires pipeline to be fitted first."""
        with pytest.raises(RuntimeError, match="Pipeline must be fitted before inverse_transform"):
            simple_pipeline.inverse_transform(sample_dataframe)

    def test_inverse_transform_single_step(self, simple_pipeline, sample_dataframe):
        """Test inverse transforming data with single step."""
        # Fit and transform
        transformed_df = simple_pipeline.fit_transform(sample_dataframe)

        # Inverse transform
        inverse_df = simple_pipeline.inverse_transform(transformed_df)

        # Should get back approximately original values
        original_streamflow = sample_dataframe["streamflow"].to_numpy()
        inverse_streamflow = inverse_df["streamflow"].to_numpy()

        np.testing.assert_allclose(inverse_streamflow, original_streamflow, rtol=1e-10)

        # Other columns should be unchanged
        for col in ["precipitation", "temperature", "other_col", "basin_id"]:
            np.testing.assert_array_equal(inverse_df[col].to_numpy(), sample_dataframe[col].to_numpy())

    def test_multi_step_pipeline(self, sample_dataframe):
        """Test complete pipeline with multiple steps."""
        steps = [
            CompositePipelineStep(pipeline_type="per_basin", transforms=[Log()], columns=["streamflow"]),
            CompositePipelineStep(
                pipeline_type="global", transforms=[ZScore()], columns=["streamflow", "precipitation"]
            ),
        ]
        pipeline = CompositePipeline(steps, group_identifier="basin_id")

        # Test full cycle
        transformed_df = pipeline.fit_transform(sample_dataframe)
        inverse_df = pipeline.inverse_transform(transformed_df)

        # Check that we can recover original streamflow values (within tolerance)
        original_streamflow = sample_dataframe["streamflow"].to_numpy()
        recovered_streamflow = inverse_df["streamflow"].to_numpy()

        np.testing.assert_allclose(recovered_streamflow, original_streamflow, rtol=1e-10)

        # Precipitation should be recovered exactly (only z-score was applied)
        np.testing.assert_allclose(
            inverse_df["precipitation"].to_numpy(), sample_dataframe["precipitation"].to_numpy(), rtol=1e-10
        )

    def test_get_step_summary(self, sample_dataframe):
        """Test step summary functionality."""
        steps = [
            CompositePipelineStep(pipeline_type="per_basin", transforms=[Log()], columns=["streamflow"]),
            CompositePipelineStep(pipeline_type="global", transforms=[ZScore()], columns=["precipitation"]),
        ]
        pipeline = CompositePipeline(steps, group_identifier="basin_id")

        # Before fitting
        summary = pipeline.get_step_summary()
        assert len(summary) == 2
        assert summary[0]["fitted"] is False
        assert summary[1]["fitted"] is False

        # After fitting
        pipeline.fit(sample_dataframe)
        summary = pipeline.get_step_summary()
        assert summary[0]["fitted"] is True
        assert summary[1]["fitted"] is True

        # Check summary content
        assert summary[0]["step"] == 0
        assert summary[0]["pipeline_type"] == "per_basin"
        assert summary[0]["transforms"] == ["Log"]
        assert summary[0]["columns"] == ["streamflow"]

        assert summary[1]["step"] == 1
        assert summary[1]["pipeline_type"] == "global"
        assert summary[1]["transforms"] == ["ZScore"]
        assert summary[1]["columns"] == ["precipitation"]


class TestCompositePipelineJoblib:
    """Test joblib serialization/deserialization of CompositePipeline."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample test DataFrame."""
        return pl.DataFrame(
            {
                "basin_id": [1, 1, 1, 2, 2, 2],
                "streamflow": [10.0, 15.0, 12.0, 8.0, 11.0, 9.0],
                "precipitation": [2.0, 3.0, 2.5, 1.5, 2.8, 1.8],
                "temperature": [20.0, 22.0, 21.0, 18.0, 19.0, 17.0],
                "other_col": [1, 2, 3, 4, 5, 6],
            }
        )

    @pytest.fixture
    def complex_dataframe(self):
        """Create more complex test DataFrame with string group identifiers."""
        return pl.DataFrame(
            {
                "watershed": ["ws_a", "ws_a", "ws_a", "ws_b", "ws_b", "ws_b", "ws_c", "ws_c"],
                "streamflow": [10.0, 15.0, 12.0, 8.0, 11.0, 9.0, 25.0, 30.0],
                "precipitation": [2.0, 3.0, 2.5, 1.5, 2.8, 1.8, 4.0, 5.0],
                "temperature": [20.0, 22.0, 21.0, 18.0, 19.0, 17.0, 25.0, 28.0],
                "elevation": [100.0, 105.0, 102.0, 80.0, 85.0, 82.0, 200.0, 210.0],
            }
        )

    @pytest.fixture
    def single_step_pipeline(self):
        """Create single-step pipeline for testing."""
        steps = [CompositePipelineStep(pipeline_type="per_basin", transforms=[Log()], columns=["streamflow"])]
        return CompositePipeline(steps, group_identifier="basin_id")

    @pytest.fixture
    def multi_step_pipeline(self):
        """Create multi-step pipeline for testing."""
        steps = [
            CompositePipelineStep(pipeline_type="per_basin", transforms=[Log()], columns=["streamflow"]),
            CompositePipelineStep(
                pipeline_type="global", transforms=[ZScore()], columns=["streamflow", "precipitation"]
            ),
        ]
        return CompositePipeline(steps, group_identifier="basin_id")

    @pytest.fixture
    def complex_pipeline(self):
        """Create complex multi-step pipeline with string group identifiers."""
        steps = [
            CompositePipelineStep(
                pipeline_type="per_basin", transforms=[Log(), ZScore()], columns=["streamflow", "precipitation"]
            ),
            CompositePipelineStep(pipeline_type="global", transforms=[ZScore()], columns=["temperature", "elevation"]),
            CompositePipelineStep(pipeline_type="per_basin", transforms=[Log()], columns=["elevation"]),
        ]
        return CompositePipeline(steps, group_identifier="watershed")

    def test_unfitted_pipeline_serialization(self, single_step_pipeline):
        """Test serialization of unfitted pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "unfitted_pipeline.joblib"

            # Save unfitted pipeline
            joblib.dump(single_step_pipeline, filepath)

            # Load and verify
            loaded_pipeline = joblib.load(filepath)

            assert isinstance(loaded_pipeline, CompositePipeline)
            assert loaded_pipeline.group_identifier == single_step_pipeline.group_identifier
            assert len(loaded_pipeline.steps) == len(single_step_pipeline.steps)
            assert loaded_pipeline._is_fitted == single_step_pipeline._is_fitted is False
            assert len(loaded_pipeline._fitted_steps) == len(single_step_pipeline._fitted_steps) == 0
            assert loaded_pipeline._group_mapping == single_step_pipeline._group_mapping == {}

    def test_fitted_single_step_pipeline_serialization(self, single_step_pipeline, sample_dataframe):
        """Test serialization of fitted single-step pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "fitted_pipeline.joblib"

            # Fit pipeline
            single_step_pipeline.fit(sample_dataframe)

            # Save fitted pipeline
            joblib.dump(single_step_pipeline, filepath)

            # Load and verify basic properties
            loaded_pipeline = joblib.load(filepath)

            assert isinstance(loaded_pipeline, CompositePipeline)
            assert loaded_pipeline._is_fitted
            assert len(loaded_pipeline._fitted_steps) == 1
            assert loaded_pipeline.group_identifier == "basin_id"

            # Verify group mapping is preserved
            assert loaded_pipeline._group_mapping == single_step_pipeline._group_mapping

    def test_fitted_single_step_transform_consistency(self, single_step_pipeline, sample_dataframe):
        """Test that serialized pipeline produces identical transform results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "fitted_pipeline.joblib"

            # Fit and get original transform result
            original_transformed = single_step_pipeline.fit_transform(sample_dataframe)

            # Save and load pipeline
            joblib.dump(single_step_pipeline, filepath)
            loaded_pipeline = joblib.load(filepath)

            # Transform with loaded pipeline
            loaded_transformed = loaded_pipeline.transform(sample_dataframe)

            # Results should be identical
            assert original_transformed.equals(loaded_transformed)

    def test_fitted_single_step_inverse_transform_consistency(self, single_step_pipeline, sample_dataframe):
        """Test that serialized pipeline produces identical inverse transform results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "fitted_pipeline.joblib"

            # Fit and transform
            transformed_data = single_step_pipeline.fit_transform(sample_dataframe)

            # Get original inverse transform
            original_inverse = single_step_pipeline.inverse_transform(transformed_data)

            # Save and load pipeline
            joblib.dump(single_step_pipeline, filepath)
            loaded_pipeline = joblib.load(filepath)

            # Inverse transform with loaded pipeline
            loaded_inverse = loaded_pipeline.inverse_transform(transformed_data)

            # Results should be identical
            assert original_inverse.equals(loaded_inverse)

    def test_multi_step_pipeline_serialization(self, multi_step_pipeline, sample_dataframe):
        """Test serialization of multi-step pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "multi_step_pipeline.joblib"

            # Fit pipeline
            original_transformed = multi_step_pipeline.fit_transform(sample_dataframe)

            # Save pipeline
            joblib.dump(multi_step_pipeline, filepath)

            # Load and verify
            loaded_pipeline = joblib.load(filepath)

            assert loaded_pipeline._is_fitted
            assert len(loaded_pipeline._fitted_steps) == 2

            # Test transform consistency
            loaded_transformed = loaded_pipeline.transform(sample_dataframe)
            assert original_transformed.equals(loaded_transformed)

            # Test inverse transform consistency
            original_inverse = multi_step_pipeline.inverse_transform(original_transformed)
            loaded_inverse = loaded_pipeline.inverse_transform(loaded_transformed)
            assert original_inverse.equals(loaded_inverse)

    def test_complex_pipeline_with_string_groups(self, complex_pipeline, complex_dataframe):
        """Test serialization with string group identifiers and complex pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "complex_pipeline.joblib"

            # Fit and get original results
            original_transformed = complex_pipeline.fit_transform(complex_dataframe)

            # Save pipeline
            joblib.dump(complex_pipeline, filepath)

            # Load pipeline
            loaded_pipeline = joblib.load(filepath)

            # Verify basic properties
            assert loaded_pipeline._is_fitted
            assert len(loaded_pipeline._fitted_steps) == 3
            assert loaded_pipeline.group_identifier == "watershed"

            # Verify group mapping preservation (string identifiers)
            assert loaded_pipeline._group_mapping == complex_pipeline._group_mapping
            assert all(isinstance(v, str) for v in loaded_pipeline._group_mapping.values())

            # Test transform consistency
            loaded_transformed = loaded_pipeline.transform(complex_dataframe)
            assert original_transformed.equals(loaded_transformed)

            # Test full roundtrip consistency
            original_inverse = complex_pipeline.inverse_transform(original_transformed)
            loaded_inverse = loaded_pipeline.inverse_transform(loaded_transformed)
            assert original_inverse.equals(loaded_inverse)

    def test_pipeline_state_isolation_after_serialization(self, single_step_pipeline, sample_dataframe):
        """Test that serialized pipeline doesn't affect original pipeline state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "isolated_pipeline.joblib"

            # Fit original pipeline
            single_step_pipeline.fit(sample_dataframe)
            original_fitted_steps_count = len(single_step_pipeline._fitted_steps)

            # Save and load
            joblib.dump(single_step_pipeline, filepath)
            loaded_pipeline = joblib.load(filepath)

            # Modify loaded pipeline's internal state (simulate usage)
            test_data = sample_dataframe.with_columns(pl.col("streamflow") * 2)
            loaded_pipeline.transform(test_data)

            # Original pipeline should be unchanged
            assert len(single_step_pipeline._fitted_steps) == original_fitted_steps_count
            assert single_step_pipeline._is_fitted

    def test_multiple_save_load_cycles(self, multi_step_pipeline, sample_dataframe):
        """Test multiple save/load cycles maintain consistency."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initial fit
            original_result = multi_step_pipeline.fit_transform(sample_dataframe)

            for cycle in range(3):
                filepath = Path(temp_dir) / f"cycle_{cycle}_pipeline.joblib"

                # Save current pipeline
                joblib.dump(multi_step_pipeline, filepath)

                # Load pipeline
                multi_step_pipeline = joblib.load(filepath)

                # Verify consistency
                current_result = multi_step_pipeline.transform(sample_dataframe)
                assert original_result.equals(current_result)

    def test_serialization_with_new_data_after_load(self, single_step_pipeline, sample_dataframe):
        """Test that loaded pipeline works with new data (same structure)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "pipeline_new_data.joblib"

            # Fit with original data
            single_step_pipeline.fit(sample_dataframe)

            # Save pipeline
            joblib.dump(single_step_pipeline, filepath)

            # Create new data with same structure but different values
            new_data = pl.DataFrame(
                {
                    "basin_id": [1, 1, 2, 2, 3, 3],  # Note: includes new group '3'
                    "streamflow": [20.0, 25.0, 18.0, 22.0, 30.0, 35.0],
                    "precipitation": [3.0, 4.0, 2.5, 3.5, 5.0, 6.0],
                    "temperature": [25.0, 27.0, 23.0, 26.0, 30.0, 32.0],
                    "other_col": [7, 8, 9, 10, 11, 12],
                }
            )

            # Load pipeline and test with new data
            loaded_pipeline = joblib.load(filepath)

            # Transform should work (with warning for unseen group)
            with pytest.warns(RuntimeWarning, match="Groups .* were not seen during fit"):
                result = loaded_pipeline.transform(new_data)

            assert isinstance(result, pl.DataFrame)
            assert result.shape == new_data.shape

    def test_get_step_summary_after_serialization(self, multi_step_pipeline, sample_dataframe):
        """Test that step summary works correctly after serialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "pipeline_summary.joblib"

            # Fit and get original summary
            multi_step_pipeline.fit(sample_dataframe)
            original_summary = multi_step_pipeline.get_step_summary()

            # Save and load
            joblib.dump(multi_step_pipeline, filepath)
            loaded_pipeline = joblib.load(filepath)

            # Get loaded summary
            loaded_summary = loaded_pipeline.get_step_summary()

            # Should be identical
            assert loaded_summary == original_summary
            assert all(step["fitted"] for step in loaded_summary)

    def test_error_handling_after_serialization(self, single_step_pipeline, sample_dataframe):
        """Test that error handling works correctly after serialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "pipeline_errors.joblib"

            # Fit and save
            single_step_pipeline.fit(sample_dataframe)
            joblib.dump(single_step_pipeline, filepath)

            # Load pipeline
            loaded_pipeline = joblib.load(filepath)

            # Test error cases still work
            invalid_df = pl.DataFrame({"wrong_column": [1, 2, 3]})

            with pytest.raises(ValueError, match="Group identifier 'basin_id' not found"):
                loaded_pipeline.transform(invalid_df)

            missing_col_df = sample_dataframe.drop("streamflow")
            with pytest.raises(ValueError, match="Columns \\['streamflow'\\] not found"):
                loaded_pipeline.transform(missing_col_df)
