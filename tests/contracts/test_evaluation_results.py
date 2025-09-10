"""Tests for EvaluationResults contract."""

import polars as pl
import pytest
import torch

from transfer_learning_publication.contracts import EvaluationResults, ForecastOutput


class TestEvaluationResults:
    """Tests for EvaluationResults class."""

    def test_init_with_valid_results(self):
        """Test initialization with valid ForecastOutput objects."""
        # Create test ForecastOutputs
        predictions1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        observations1 = torch.tensor([[1.1, 2.1], [3.1, 4.1]])
        groups1 = ["basin1", "basin2"]

        fo1 = ForecastOutput(
            predictions=predictions1,
            observations=observations1,
            group_identifiers=groups1,
        )

        predictions2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        observations2 = torch.tensor([[5.1, 6.1], [7.1, 8.1]])
        groups2 = ["basin1", "basin2"]

        fo2 = ForecastOutput(
            predictions=predictions2,
            observations=observations2,
            group_identifiers=groups2,
        )

        # Create EvaluationResults
        results = EvaluationResults(
            results_dict={"model1": fo1, "model2": fo2},
            output_length=2,
        )

        assert len(results.results_dict) == 2
        assert results.output_length == 2
        assert not results.include_dates

    def test_init_empty_results_dict(self):
        """Test initialization with empty results dict raises error."""
        with pytest.raises(ValueError, match="results_dict cannot be empty"):
            EvaluationResults(results_dict={}, output_length=2)

    def test_init_inconsistent_output_lengths(self):
        """Test initialization with inconsistent output lengths."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0]]),
            observations=torch.tensor([[1.1, 2.1]]),
            group_identifiers=["basin1"],
        )

        fo2 = ForecastOutput(
            predictions=torch.tensor([[3.0, 4.0, 5.0]]),  # Different output length
            observations=torch.tensor([[3.1, 4.1, 5.1]]),
            group_identifiers=["basin1"],
        )

        with pytest.raises(ValueError, match="Inconsistent output lengths"):
            EvaluationResults(
                results_dict={"model1": fo1, "model2": fo2},
                output_length=None,
            )

    def test_init_invalid_forecast_output_type(self):
        """Test initialization with non-ForecastOutput object."""
        with pytest.raises(TypeError, match="Expected ForecastOutput"):
            EvaluationResults(
                results_dict={"model1": "not_a_forecast_output"},
                output_length=2,
            )

    def test_raw_data_property(self):
        """Test raw_data property returns correct DataFrame."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            observations=torch.tensor([[1.1, 2.1], [3.1, 4.1]]),
            group_identifiers=["basin1", "basin2"],
        )

        results = EvaluationResults(
            results_dict={"model1": fo1},
            output_length=2,
        )

        lf = results.raw_data
        assert isinstance(lf, pl.LazyFrame)

        df = lf.collect()
        # Check structure
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 4  # 2 basins * 2 lead times
        assert set(df.columns) == {"model_name", "group_identifier", "lead_time", "prediction", "observation"}

        # Check values
        assert df["model_name"].unique().to_list() == ["model1"]
        assert set(df["group_identifier"].to_list()) == {"basin1", "basin2"}
        assert set(df["lead_time"].to_list()) == {1, 2}

    def test_raw_data_with_dates(self):
        """Test raw_data includes dates when available."""
        # Create timestamps in milliseconds (as per actual data format)
        # Using specific dates for testing: 2024-01-01 00:00:00 UTC and 2024-01-02 00:00:00 UTC
        import datetime

        date1 = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.UTC)
        date2 = datetime.datetime(2024, 1, 2, 0, 0, 0, tzinfo=datetime.UTC)
        timestamp1_ms = int(date1.timestamp() * 1000)
        timestamp2_ms = int(date2.timestamp() * 1000)
        input_end_dates = torch.tensor([timestamp1_ms, timestamp2_ms], dtype=torch.long)

        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            observations=torch.tensor([[1.1, 2.1], [3.1, 4.1]]),
            group_identifiers=["basin1", "basin2"],
            input_end_dates=input_end_dates,
        )

        results = EvaluationResults(
            results_dict={"model1": fo1},
            output_length=2,
        )

        lf = results.raw_data
        assert isinstance(lf, pl.LazyFrame)

        df = lf.collect()
        # Check date columns are included
        assert "issue_date" in df.columns
        assert "prediction_date" in df.columns
        assert results.include_dates

        # Verify dates are correctly parsed
        # First row should have issue_date = 2024-01-01
        first_row = df.row(0, named=True)
        issue_date = first_row["issue_date"]
        prediction_date = first_row["prediction_date"]

        # For polars datetime, we need to convert to Python datetime for attribute access
        import datetime

        issue_dt = datetime.datetime.fromisoformat(str(issue_date).replace(" ", "T"))
        pred_dt = datetime.datetime.fromisoformat(str(prediction_date).replace(" ", "T"))

        assert issue_dt.year == 2024
        assert issue_dt.month == 1
        assert issue_dt.day == 1

        # Prediction date should be issue_date + lead_time days
        # For lead_time=1, prediction_date should be 2024-01-02
        assert pred_dt.year == 2024
        assert pred_dt.month == 1
        assert pred_dt.day == 2

    def test_by_model(self):
        """Test filtering by model name."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0]]),
            observations=torch.tensor([[1.1, 2.1]]),
            group_identifiers=["basin1"],
        )

        fo2 = ForecastOutput(
            predictions=torch.tensor([[3.0, 4.0]]),
            observations=torch.tensor([[3.1, 4.1]]),
            group_identifiers=["basin1"],
        )

        results = EvaluationResults(
            results_dict={"model1": fo1, "model2": fo2},
            output_length=2,
        )

        # Test valid model
        lf_model1 = results.by_model("model1")
        assert isinstance(lf_model1, pl.LazyFrame)
        df_model1 = lf_model1.collect()
        assert len(df_model1) == 2
        assert df_model1["model_name"].unique().to_list() == ["model1"]

        # Test invalid model
        with pytest.raises(KeyError, match="Model 'model3' not found"):
            results.by_model("model3")

    def test_by_basin(self):
        """Test filtering by basin ID."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            observations=torch.tensor([[1.1, 2.1], [3.1, 4.1]]),
            group_identifiers=["basin1", "basin2"],
        )

        results = EvaluationResults(
            results_dict={"model1": fo1},
            output_length=2,
        )

        lf_basin1 = results.by_basin("basin1")
        assert isinstance(lf_basin1, pl.LazyFrame)
        df_basin1 = lf_basin1.collect()
        assert len(df_basin1) == 2  # 2 lead times
        assert df_basin1["group_identifier"].unique().to_list() == ["basin1"]

        # Test non-existent basin (should return empty)
        lf_basin3 = results.by_basin("basin3")
        assert isinstance(lf_basin3, pl.LazyFrame)
        df_basin3 = lf_basin3.collect()
        assert len(df_basin3) == 0

    def test_by_lead_time(self):
        """Test filtering by lead time."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0, 3.0]]),
            observations=torch.tensor([[1.1, 2.1, 3.1]]),
            group_identifiers=["basin1"],
        )

        results = EvaluationResults(
            results_dict={"model1": fo1},
            output_length=3,
        )

        # Test valid lead time
        lf_lead1 = results.by_lead_time(1)
        assert isinstance(lf_lead1, pl.LazyFrame)
        df_lead1 = lf_lead1.collect()
        assert len(df_lead1) == 1
        assert df_lead1["lead_time"].unique().to_list() == [1]

        # Test out of range lead time
        with pytest.raises(ValueError, match="lead_time must be between"):
            results.by_lead_time(0)

        with pytest.raises(ValueError, match="lead_time must be between"):
            results.by_lead_time(4)

    def test_to_parquet(self, tmp_path):
        """Test exporting to parquet files."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            observations=torch.tensor([[1.1, 2.1], [3.1, 4.1]]),
            group_identifiers=["basin1", "basin2"],
        )

        fo2 = ForecastOutput(
            predictions=torch.tensor([[5.0, 6.0]]),
            observations=torch.tensor([[5.1, 6.1]]),
            group_identifiers=["basin3"],
        )

        results = EvaluationResults(
            results_dict={"model1": fo1, "model2": fo2},
            output_length=2,
        )

        output_dir = tmp_path / "parquet_output"
        results.to_parquet(output_dir)

        # Check partitioned structure
        assert output_dir.exists()
        assert (output_dir / "model_name=model1").exists()
        assert (output_dir / "model_name=model2").exists()

        # Read back and verify
        df_loaded = pl.read_parquet(output_dir / "**/*.parquet")
        assert len(df_loaded) == 6  # (2+1) basins * 2 lead times

    def test_to_parquet_custom_partitions(self, tmp_path):
        """Test exporting with custom partition columns."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0]]),
            observations=torch.tensor([[1.1, 2.1]]),
            group_identifiers=["basin1"],
        )

        results = EvaluationResults(
            results_dict={"model1": fo1},
            output_length=2,
        )

        output_dir = tmp_path / "custom_parquet"
        results.to_parquet(output_dir, partition_cols=["lead_time"])

        # Check partitioned by lead_time
        assert (output_dir / "lead_time=1").exists()
        assert (output_dir / "lead_time=2").exists()

    def test_list_models(self):
        """Test listing model names."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0]]),
            observations=torch.tensor([[1.1]]),
            group_identifiers=["basin1"],
        )

        fo2 = ForecastOutput(
            predictions=torch.tensor([[2.0]]),
            observations=torch.tensor([[2.1]]),
            group_identifiers=["basin1"],
        )

        results = EvaluationResults(
            results_dict={"model_a": fo1, "model_b": fo2},
            output_length=1,
        )

        models = results.list_models()
        assert sorted(models) == ["model_a", "model_b"]

    def test_list_basins(self):
        """Test listing unique basin IDs."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0], [2.0]]),
            observations=torch.tensor([[1.1], [2.1]]),
            group_identifiers=["basin1", "basin2"],
        )

        fo2 = ForecastOutput(
            predictions=torch.tensor([[3.0], [4.0]]),
            observations=torch.tensor([[3.1], [4.1]]),
            group_identifiers=["basin2", "basin3"],  # Overlapping basin2
        )

        results = EvaluationResults(
            results_dict={"model1": fo1, "model2": fo2},
            output_length=1,
        )

        basins = results.list_basins()
        assert basins == ["basin1", "basin2", "basin3"]  # Sorted and unique

    def test_summary(self):
        """Test summary statistics."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            observations=torch.tensor([[1.1, 2.1], [3.1, 4.1]]),
            group_identifiers=["basin1", "basin1"],  # Same basin, multiple samples
        )

        fo2 = ForecastOutput(
            predictions=torch.tensor([[5.0, 6.0]]),
            observations=torch.tensor([[5.1, 6.1]]),
            group_identifiers=["basin2"],
            input_end_dates=torch.tensor([100.0]),
        )

        results = EvaluationResults(
            results_dict={"model1": fo1, "model2": fo2},
            output_length=2,
        )

        summary_df = results.summary()

        assert len(summary_df) == 2
        assert set(summary_df.columns) == {"model_name", "n_samples", "n_basins", "output_length", "has_dates"}

        # Check model1 stats
        model1_row = summary_df.filter(pl.col("model_name") == "model1").row(0, named=True)
        assert model1_row["n_samples"] == 2
        assert model1_row["n_basins"] == 1  # Only basin1
        assert model1_row["output_length"] == 2
        assert not model1_row["has_dates"]

        # Check model2 stats
        model2_row = summary_df.filter(pl.col("model_name") == "model2").row(0, named=True)
        assert model2_row["n_samples"] == 1
        assert model2_row["n_basins"] == 1
        assert model2_row["has_dates"]

    def test_update_method(self):
        """Test updating results with additional models."""
        # Create initial results
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0]]),
            observations=torch.tensor([[1.1, 2.1]]),
            group_identifiers=["basin1"],
        )

        results1 = EvaluationResults(
            results_dict={"model1": fo1},
            output_length=2,
        )

        # Create additional results
        fo2 = ForecastOutput(
            predictions=torch.tensor([[3.0, 4.0]]),
            observations=torch.tensor([[3.1, 4.1]]),
            group_identifiers=["basin2"],
        )

        fo3 = ForecastOutput(
            predictions=torch.tensor([[5.0, 6.0]]),
            observations=torch.tensor([[5.1, 6.1]]),
            group_identifiers=["basin3"],
        )

        results2 = EvaluationResults(
            results_dict={"model2": fo2, "model3": fo3},
            output_length=2,
        )

        # Update results1 with results2
        results1.update(results2)

        # Check that all models are present
        assert len(results1.results_dict) == 3
        assert "model1" in results1.results_dict
        assert "model2" in results1.results_dict
        assert "model3" in results1.results_dict

        # Verify the data is correct
        assert results1.results_dict["model1"] == fo1
        assert results1.results_dict["model2"] == fo2
        assert results1.results_dict["model3"] == fo3

    def test_update_method_overwrite_warning(self, caplog):
        """Test that update warns when overwriting existing models."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0]]),
            observations=torch.tensor([[1.1, 2.1]]),
            group_identifiers=["basin1"],
        )

        fo2 = ForecastOutput(
            predictions=torch.tensor([[3.0, 4.0]]),
            observations=torch.tensor([[3.1, 4.1]]),
            group_identifiers=["basin2"],
        )

        results1 = EvaluationResults(
            results_dict={"model1": fo1},
            output_length=2,
        )

        results2 = EvaluationResults(
            results_dict={"model1": fo2},  # Same model name
            output_length=2,
        )

        # Update with overlapping model name
        with caplog.at_level("WARNING"):
            results1.update(results2)

        # Check warning was logged
        assert "Overwriting existing results for 'model1'" in caplog.text

        # Check that model1 was overwritten
        assert results1.results_dict["model1"] == fo2

    def test_update_method_incompatible_output_length(self):
        """Test that update raises error for incompatible output lengths."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0]]),
            observations=torch.tensor([[1.1, 2.1]]),
            group_identifiers=["basin1"],
        )

        fo2 = ForecastOutput(
            predictions=torch.tensor([[3.0, 4.0, 5.0]]),  # Different output length
            observations=torch.tensor([[3.1, 4.1, 5.1]]),
            group_identifiers=["basin2"],
        )

        results1 = EvaluationResults(
            results_dict={"model1": fo1},
            output_length=2,
        )

        results2 = EvaluationResults(
            results_dict={"model2": fo2},
            output_length=3,
        )

        # Should raise error due to mismatched output lengths
        with pytest.raises(ValueError, match="Cannot merge results with different output_lengths"):
            results1.update(results2)

    def test_update_method_dates_flag(self):
        """Test that update correctly handles include_dates flag."""
        # Results without dates
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0]]),
            observations=torch.tensor([[1.1]]),
            group_identifiers=["basin1"],
        )

        results1 = EvaluationResults(
            results_dict={"model1": fo1},
            output_length=1,
        )

        assert not results1.include_dates

        # Results with dates
        fo2 = ForecastOutput(
            predictions=torch.tensor([[2.0]]),
            observations=torch.tensor([[2.1]]),
            group_identifiers=["basin2"],
            input_end_dates=torch.tensor([100.0]),
        )

        results2 = EvaluationResults(
            results_dict={"model2": fo2},
            output_length=1,
        )

        assert results2.include_dates

        # Update results1 (no dates) with results2 (has dates)
        results1.update(results2)

        # Should now have include_dates = True
        assert results1.include_dates

    def test_multiple_models_integration(self):
        """Test full workflow with multiple models."""
        # Create three models with different characteristics
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            observations=torch.tensor([[1.1, 2.1], [3.1, 4.1]]),
            group_identifiers=["basin1", "basin2"],
        )

        fo2 = ForecastOutput(
            predictions=torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
            observations=torch.tensor([[5.1, 6.1], [7.1, 8.1], [9.1, 10.1]]),
            group_identifiers=["basin1", "basin2", "basin3"],
        )

        fo3 = ForecastOutput(
            predictions=torch.tensor([[11.0, 12.0]]),
            observations=torch.tensor([[11.1, 12.1]]),
            group_identifiers=["basin4"],
        )

        results = EvaluationResults(
            results_dict={"lstm": fo1, "gru": fo2, "naive": fo3},
            output_length=2,
        )

        # Test combined LazyFrame
        lf = results.raw_data
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 12  # (2+3+1) samples * 2 lead times
        assert df["model_name"].n_unique() == 3

        # Test filtering combinations
        lstm_basin1 = df.filter((pl.col("model_name") == "lstm") & (pl.col("group_identifier") == "basin1"))
        assert len(lstm_basin1) == 2  # 2 lead times

        lead1_lf = results.by_lead_time(1)
        assert isinstance(lead1_lf, pl.LazyFrame)
        lead1_results = lead1_lf.collect()
        assert len(lead1_results) == 6  # 6 total samples at lead time 1

    def test_filter_single_dimension(self):
        """Test filter method with single dimension."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            observations=torch.tensor([[1.1, 2.1], [3.1, 4.1]]),
            group_identifiers=["basin1", "basin2"],
        )

        fo2 = ForecastOutput(
            predictions=torch.tensor([[5.0, 6.0]]),
            observations=torch.tensor([[5.1, 6.1]]),
            group_identifiers=["basin3"],
        )

        results = EvaluationResults(
            results_dict={"model1": fo1, "model2": fo2},
            output_length=2,
        )

        # Filter by model_name
        lf = results.filter(model_name="model1")
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 4  # 2 basins * 2 lead times
        assert df["model_name"].unique().to_list() == ["model1"]

        # Filter by basin_id
        lf = results.filter(basin_id="basin1")
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 2  # 1 basin * 2 lead times, only from model1
        assert df["group_identifier"].unique().to_list() == ["basin1"]

        # Filter by lead_time
        lf = results.filter(lead_time=1)
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 3  # 3 basins * 1 lead time
        assert df["lead_time"].unique().to_list() == [1]

    def test_filter_multiple_dimensions(self):
        """Test filter method with multiple dimensions."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            observations=torch.tensor([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]]),
            group_identifiers=["basin1", "basin2"],
        )

        fo2 = ForecastOutput(
            predictions=torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
            observations=torch.tensor([[7.1, 8.1, 9.1], [10.1, 11.1, 12.1]]),
            group_identifiers=["basin1", "basin3"],
        )

        results = EvaluationResults(
            results_dict={"model1": fo1, "model2": fo2},
            output_length=3,
        )

        # Filter by model AND lead_time
        lf = results.filter(model_name="model1", lead_time=2)
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 2  # 2 basins for model1 at lead_time 2
        assert df["model_name"].unique().to_list() == ["model1"]
        assert df["lead_time"].unique().to_list() == [2]

        # Filter by model AND basin
        lf = results.filter(model_name="model2", basin_id="basin1")
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 3  # 1 basin * 3 lead times
        assert df["model_name"].unique().to_list() == ["model2"]
        assert df["group_identifier"].unique().to_list() == ["basin1"]

        # Filter by basin AND lead_time (across all models)
        lf = results.filter(basin_id="basin1", lead_time=1)
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 2  # 2 models have basin1
        assert df["group_identifier"].unique().to_list() == ["basin1"]
        assert df["lead_time"].unique().to_list() == [1]
        assert set(df["model_name"].to_list()) == {"model1", "model2"}

        # Filter all three dimensions
        lf = results.filter(model_name="model1", basin_id="basin2", lead_time=3)
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 1
        assert df["model_name"][0] == "model1"
        assert df["group_identifier"][0] == "basin2"
        assert df["lead_time"][0] == 3

    def test_filter_with_lists(self):
        """Test filter method with list inputs."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            observations=torch.tensor([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]]),
            group_identifiers=["basin1", "basin2", "basin3"],
        )

        fo2 = ForecastOutput(
            predictions=torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
            observations=torch.tensor([[7.1, 8.1], [9.1, 10.1]]),
            group_identifiers=["basin1", "basin4"],
        )

        results = EvaluationResults(
            results_dict={"model1": fo1, "model2": fo2},
            output_length=2,
        )

        # Filter with list of models
        lf = results.filter(model_name=["model1", "model2"])
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 10  # All data
        assert set(df["model_name"].unique().to_list()) == {"model1", "model2"}

        # Filter with list of basins
        lf = results.filter(basin_id=["basin1", "basin3"])
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 6  # (2 + 1) basins * 2 lead times
        assert set(df["group_identifier"].unique().to_list()) == {"basin1", "basin3"}

        # Filter with list of lead times
        lf = results.filter(lead_time=[1, 2])
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 10  # All data (since we have 2 lead times)
        assert set(df["lead_time"].unique().to_list()) == {1, 2}

        # Combined: multiple models and multiple basins
        lf = results.filter(model_name=["model1"], basin_id=["basin1", "basin2"])
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 4  # 2 basins * 2 lead times for model1
        assert df["model_name"].unique().to_list() == ["model1"]
        assert set(df["group_identifier"].unique().to_list()) == {"basin1", "basin2"}

    def test_filter_empty_results(self):
        """Test filter method that returns empty results."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0]]),
            observations=torch.tensor([[1.1, 2.1]]),
            group_identifiers=["basin1"],
        )

        results = EvaluationResults(
            results_dict={"model1": fo1},
            output_length=2,
        )

        # Filter for non-existent basin
        lf = results.filter(basin_id="basin_not_exists")
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 0

        # Filter for non-existent model (list)
        lf = results.filter(model_name=["model_not_exists"])
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 0

    def test_filter_validation(self):
        """Test filter method validation."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0, 3.0]]),
            observations=torch.tensor([[1.1, 2.1, 3.1]]),
            group_identifiers=["basin1"],
        )

        results = EvaluationResults(
            results_dict={"model1": fo1},
            output_length=3,
        )

        # Test invalid lead_time (single value)
        with pytest.raises(ValueError, match="lead_time must be between"):
            results.filter(lead_time=0)

        with pytest.raises(ValueError, match="lead_time must be between"):
            results.filter(lead_time=4)

        # Test invalid lead_time in list
        with pytest.raises(ValueError, match="lead_time must be between"):
            results.filter(lead_time=[1, 2, 5])

    def test_filter_no_criteria(self):
        """Test filter method with no criteria returns all data."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0]]),
            observations=torch.tensor([[1.1, 2.1]]),
            group_identifiers=["basin1"],
        )

        results = EvaluationResults(
            results_dict={"model1": fo1},
            output_length=2,
        )

        # No filter criteria should return all data
        lf = results.filter()
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert df.equals(results.raw_data.collect())

    def test_by_methods_use_filter(self):
        """Test that by_* methods properly use the filter method."""
        fo1 = ForecastOutput(
            predictions=torch.tensor([[1.0, 2.0]]),
            observations=torch.tensor([[1.1, 2.1]]),
            group_identifiers=["basin1"],
        )

        fo2 = ForecastOutput(
            predictions=torch.tensor([[3.0, 4.0]]),
            observations=torch.tensor([[3.1, 4.1]]),
            group_identifiers=["basin2"],
        )

        results = EvaluationResults(
            results_dict={"model1": fo1, "model2": fo2},
            output_length=2,
        )

        # Test by_model uses filter
        by_model_lf = results.by_model("model1")
        filter_lf = results.filter(model_name="model1")
        assert isinstance(by_model_lf, pl.LazyFrame)
        assert isinstance(filter_lf, pl.LazyFrame)
        assert by_model_lf.collect().equals(filter_lf.collect())

        # Test by_basin uses filter
        by_basin_lf = results.by_basin("basin1")
        filter_lf = results.filter(basin_id="basin1")
        assert isinstance(by_basin_lf, pl.LazyFrame)
        assert isinstance(filter_lf, pl.LazyFrame)
        assert by_basin_lf.collect().equals(filter_lf.collect())

        # Test by_lead_time uses filter
        by_lead_lf = results.by_lead_time(1)
        filter_lf = results.filter(lead_time=1)
        assert isinstance(by_lead_lf, pl.LazyFrame)
        assert isinstance(filter_lf, pl.LazyFrame)
        assert by_lead_lf.collect().equals(filter_lf.collect())
