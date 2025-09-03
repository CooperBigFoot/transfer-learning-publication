import polars as pl
import pytest

from transfer_learning_publication.transforms import Log, PipelineBuilder, ZScore


class TestPipelineBuilder:
    """Test PipelineBuilder fluent API and optimization logic."""

    def test_builder_initialization(self):
        """Test builder can be initialized with group identifier."""
        builder = PipelineBuilder(group_identifier="basin_id")
        assert builder.group_identifier == "basin_id"
        assert builder._steps == []

    def test_add_per_basin_single_transform(self):
        """Test adding single per-basin transform."""
        builder = PipelineBuilder(group_identifier="basin_id")
        result = builder.add_per_basin(Log(), columns=["streamflow"])

        assert result is builder  # Test method chaining
        assert len(builder._steps) == 1
        assert builder._steps[0].pipeline_type == "per_basin"
        assert len(builder._steps[0].transforms) == 1
        assert isinstance(builder._steps[0].transforms[0], Log)
        assert builder._steps[0].columns == ["streamflow"]

    def test_add_per_basin_multiple_transforms(self):
        """Test adding multiple per-basin transforms in one call."""
        builder = PipelineBuilder(group_identifier="basin_id")
        transforms = [Log(), ZScore()]
        builder.add_per_basin(transforms, columns=["streamflow", "precipitation"])

        assert len(builder._steps) == 1
        assert len(builder._steps[0].transforms) == 2
        assert isinstance(builder._steps[0].transforms[0], Log)
        assert isinstance(builder._steps[0].transforms[1], ZScore)
        assert builder._steps[0].columns == ["streamflow", "precipitation"]

    def test_add_global_single_transform(self):
        """Test adding single global transform."""
        builder = PipelineBuilder(group_identifier="basin_id")
        builder.add_global(ZScore(), columns=["temperature"])

        assert len(builder._steps) == 1
        assert builder._steps[0].pipeline_type == "global"
        assert len(builder._steps[0].transforms) == 1
        assert isinstance(builder._steps[0].transforms[0], ZScore)
        assert builder._steps[0].columns == ["temperature"]

    def test_method_chaining(self):
        """Test fluent API method chaining."""
        builder = (
            PipelineBuilder(group_identifier="basin_id")
            .add_per_basin(Log(), columns=["streamflow"])
            .add_global(ZScore(), columns=["streamflow", "precipitation"])
        )

        assert len(builder._steps) == 2
        assert builder._steps[0].pipeline_type == "per_basin"
        assert builder._steps[1].pipeline_type == "global"

    def test_optimization_same_type_same_columns(self):
        """Test automatic optimization of consecutive same operations on same columns."""
        builder = (
            PipelineBuilder(group_identifier="basin_id")
            .add_per_basin(Log(), columns=["streamflow"])
            .add_per_basin(ZScore(), columns=["streamflow"])  # Should merge
        )

        # Should have optimized into single step
        assert len(builder._steps) == 1
        assert len(builder._steps[0].transforms) == 2
        assert isinstance(builder._steps[0].transforms[0], Log)
        assert isinstance(builder._steps[0].transforms[1], ZScore)
        assert builder._steps[0].columns == ["streamflow"]

    def test_no_optimization_different_types(self):
        """Test no optimization when pipeline types differ."""
        builder = (
            PipelineBuilder(group_identifier="basin_id")
            .add_per_basin(Log(), columns=["streamflow"])
            .add_global(ZScore(), columns=["streamflow"])  # Different type - no merge
        )

        assert len(builder._steps) == 2
        assert builder._steps[0].pipeline_type == "per_basin"
        assert builder._steps[1].pipeline_type == "global"

    def test_no_optimization_different_columns(self):
        """Test no optimization when column sets differ."""
        builder = (
            PipelineBuilder(group_identifier="basin_id")
            .add_per_basin(Log(), columns=["streamflow"])
            .add_per_basin(ZScore(), columns=["precipitation"])  # Different columns - no merge
        )

        assert len(builder._steps) == 2
        assert builder._steps[0].columns == ["streamflow"]
        assert builder._steps[1].columns == ["precipitation"]

    def test_optimization_column_order_irrelevant(self):
        """Test optimization works regardless of column order."""
        builder = (
            PipelineBuilder(group_identifier="basin_id")
            .add_per_basin(Log(), columns=["streamflow", "precipitation"])
            .add_per_basin(ZScore(), columns=["precipitation", "streamflow"])  # Same cols, different order
        )

        # Should optimize (order doesn't matter)
        assert len(builder._steps) == 1
        assert len(builder._steps[0].transforms) == 2

    def test_get_step_preview(self):
        """Test step preview functionality."""
        builder = (
            PipelineBuilder(group_identifier="basin_id")
            .add_per_basin(Log(), columns=["streamflow"])
            .add_global([ZScore(), Log()], columns=["precipitation"])
        )

        preview = builder.get_step_preview()
        assert len(preview) == 2

        assert preview[0]["step"] == 0
        assert preview[0]["pipeline_type"] == "per_basin"
        assert preview[0]["transforms"] == ["Log"]
        assert preview[0]["columns"] == ["streamflow"]
        assert preview[0]["num_transforms"] == 1

        assert preview[1]["step"] == 1
        assert preview[1]["pipeline_type"] == "global"
        assert preview[1]["transforms"] == ["ZScore", "Log"]
        assert preview[1]["columns"] == ["precipitation"]
        assert preview[1]["num_transforms"] == 2

    def test_reset(self):
        """Test resetting builder state."""
        builder = (
            PipelineBuilder(group_identifier="basin_id")
            .add_per_basin(Log(), columns=["streamflow"])
            .add_global(ZScore(), columns=["precipitation"])
        )

        assert len(builder._steps) == 2

        result = builder.reset()
        assert result is builder  # Test method chaining
        assert len(builder._steps) == 0

    def test_build_success(self):
        """Test successful pipeline building."""
        builder = (
            PipelineBuilder(group_identifier="basin_id")
            .add_per_basin(Log(), columns=["streamflow"])
            .add_global(ZScore(), columns=["precipitation"])
        )

        pipeline = builder.build()
        assert pipeline.group_identifier == "basin_id"
        assert len(pipeline.steps) == 2

    def test_build_empty_pipeline_fails(self):
        """Test building empty pipeline raises error."""
        builder = PipelineBuilder(group_identifier="basin_id")

        with pytest.raises(ValueError, match="Cannot build pipeline: no steps have been added"):
            builder.build()

    def test_invalid_inputs(self):
        """Test validation of invalid inputs."""
        builder = PipelineBuilder(group_identifier="basin_id")

        # Empty transforms list
        with pytest.raises(ValueError, match="Must provide at least one transform"):
            builder.add_per_basin([], columns=["streamflow"])

        # Empty columns list
        with pytest.raises(ValueError, match="Must specify at least one column"):
            builder.add_per_basin(Log(), columns=[])

    def test_validate_against_dataframe_success(self):
        """Test successful DataFrame validation."""
        df = pl.DataFrame(
            {"basin_id": [1, 1, 2, 2], "streamflow": [10.0, 15.0, 8.0, 12.0], "precipitation": [2.0, 3.0, 1.5, 2.5]}
        )

        builder = PipelineBuilder(group_identifier="basin_id").add_per_basin(Log(), columns=["streamflow"])

        errors = builder.validate_against_dataframe(df)
        assert len(errors) == 0

    def test_validate_against_dataframe_errors(self):
        """Test DataFrame validation catches errors."""
        builder = PipelineBuilder(group_identifier="missing_col").add_per_basin(Log(), columns=["missing_feature"])

        df = pl.DataFrame({"basin_id": [1, 2], "streamflow": [10.0, 15.0]})

        errors = builder.validate_against_dataframe(df)
        assert len(errors) == 2
        assert "Group identifier 'missing_col' not found" in errors[0]
        assert "columns ['missing_feature'] not found" in errors[1]

    def test_validate_against_non_dataframe(self):
        """Test validation rejects non-DataFrame input."""
        builder = PipelineBuilder(group_identifier="basin_id")

        errors = builder.validate_against_dataframe("not a dataframe")
        assert len(errors) == 1
        assert "Input must be a Polars DataFrame" in errors[0]

    def test_validate_against_empty_dataframe(self):
        """Test validation catches empty DataFrame."""
        builder = PipelineBuilder(group_identifier="basin_id")
        empty_df = pl.DataFrame({"basin_id": [], "streamflow": []})

        errors = builder.validate_against_dataframe(empty_df)
        assert len(errors) == 1
        assert "DataFrame cannot be empty" in errors[0]
