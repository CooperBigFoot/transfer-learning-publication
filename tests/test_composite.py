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
