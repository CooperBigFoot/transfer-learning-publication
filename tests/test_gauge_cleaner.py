from unittest.mock import Mock, call

import polars as pl
import pytest

from transfer_learning_publication.cleaners import GaugeCleaner


class TestGaugeCleaner:
    """Test suite for the GaugeCleaner builder pattern class."""

    def test_initialization(self):
        """Test cleaner initializes with empty step list."""
        cleaner = GaugeCleaner()

        assert cleaner.steps == []

    def test_ensure_temporal_consistency_method_chaining(self):
        """Test ensure_temporal_consistency method returns self for chaining."""
        cleaner = GaugeCleaner()
        result = cleaner.ensure_temporal_consistency()

        assert result is cleaner
        assert len(cleaner.steps) == 1

    def test_ensure_temporal_consistency_parameter_handling(self):
        """Test ensure_temporal_consistency stores correct function and parameters."""
        cleaner = GaugeCleaner()
        cleaner.ensure_temporal_consistency(date_column="timestamp")

        func, kwargs = cleaner.steps[0]
        assert func.__name__ == "ensure_temporal_consistency"
        assert kwargs == {"date_column": "timestamp"}

    def test_ensure_temporal_consistency_default_parameter(self):
        """Test ensure_temporal_consistency uses default date_column parameter."""
        cleaner = GaugeCleaner()
        cleaner.ensure_temporal_consistency()

        _, kwargs = cleaner.steps[0]
        assert kwargs == {"date_column": "date"}

    def test_trim_to_column_method_chaining(self):
        """Test trim_to_column method returns self for chaining."""
        cleaner = GaugeCleaner()
        result = cleaner.trim_to_column("streamflow")

        assert result is cleaner
        assert len(cleaner.steps) == 1

    def test_trim_to_column_parameter_handling(self):
        """Test trim_to_column stores correct function and parameters."""
        cleaner = GaugeCleaner()
        cleaner.trim_to_column("streamflow")

        func, kwargs = cleaner.steps[0]
        assert func.__name__ == "trim_to_column"
        assert kwargs == {"priority_column": "streamflow"}

    def test_fill_na_columns_method_chaining(self):
        """Test fill_na_columns method returns self for chaining."""
        cleaner = GaugeCleaner()
        result = cleaner.fill_na_columns(["precipitation", "temperature"], 0.0)

        assert result is cleaner
        assert len(cleaner.steps) == 1

    def test_fill_na_columns_parameter_handling(self):
        """Test fill_na_columns stores correct function and parameters."""
        cleaner = GaugeCleaner()
        cleaner.fill_na_columns(["precipitation", "temperature"], 0.5, add_binary_flag=True)

        func, kwargs = cleaner.steps[0]
        assert func.__name__ == "fill_na_columns"
        assert kwargs == {"columns": ["precipitation", "temperature"], "fill_value": 0.5, "add_binary_flag": True}

    def test_fill_na_columns_default_parameter(self):
        """Test fill_na_columns uses default add_binary_flag parameter."""
        cleaner = GaugeCleaner()
        cleaner.fill_na_columns(["temp"], 0.0)

        _, kwargs = cleaner.steps[0]
        assert kwargs["add_binary_flag"] is False

    def test_clip_columns_method_chaining(self):
        """Test clip_columns method returns self for chaining."""
        cleaner = GaugeCleaner()
        result = cleaner.clip_columns(["streamflow"], min_value=0.0)

        assert result is cleaner
        assert len(cleaner.steps) == 1

    def test_clip_columns_parameter_handling(self):
        """Test clip_columns stores correct function and parameters."""
        cleaner = GaugeCleaner()
        cleaner.clip_columns(["streamflow", "precipitation"], min_value=0.0, max_value=100.0)

        func, kwargs = cleaner.steps[0]
        assert func.__name__ == "clip_columns"
        assert kwargs == {"columns": ["streamflow", "precipitation"], "min_value": 0.0, "max_value": 100.0}

    def test_clip_columns_default_parameters(self):
        """Test clip_columns uses default None parameters."""
        cleaner = GaugeCleaner()
        cleaner.clip_columns(["streamflow"])

        _, kwargs = cleaner.steps[0]
        assert kwargs == {"columns": ["streamflow"], "min_value": None, "max_value": None}

    def test_add_cyclical_date_encoding_method_chaining(self):
        """Test add_cyclical_date_encoding method returns self for chaining."""
        cleaner = GaugeCleaner()
        result = cleaner.add_cyclical_date_encoding()

        assert result is cleaner
        assert len(cleaner.steps) == 1

    def test_add_cyclical_date_encoding_parameter_handling(self):
        """Test add_cyclical_date_encoding stores correct function and parameters."""
        cleaner = GaugeCleaner()
        cleaner.add_cyclical_date_encoding(date_column="timestamp")

        func, kwargs = cleaner.steps[0]
        assert func.__name__ == "add_cyclical_date_encoding"
        assert kwargs == {"date_column": "timestamp"}

    def test_add_cyclical_date_encoding_default_parameter(self):
        """Test add_cyclical_date_encoding uses default date_column parameter."""
        cleaner = GaugeCleaner()
        cleaner.add_cyclical_date_encoding()

        _, kwargs = cleaner.steps[0]
        assert kwargs == {"date_column": "date"}

    def test_complex_method_chaining(self):
        """Test complex method chaining builds correct pipeline."""
        cleaner = (
            GaugeCleaner()
            .ensure_temporal_consistency()
            .trim_to_column("streamflow")
            .fill_na_columns(["precipitation", "temperature"], 0.0)
            .clip_columns(["streamflow"], min_value=0.0)
            .add_cyclical_date_encoding()
        )

        assert len(cleaner.steps) == 5

        # Check step order and types
        assert cleaner.steps[0][0].__name__ == "ensure_temporal_consistency"
        assert cleaner.steps[1][0].__name__ == "trim_to_column"
        assert cleaner.steps[2][0].__name__ == "fill_na_columns"
        assert cleaner.steps[3][0].__name__ == "clip_columns"
        assert cleaner.steps[4][0].__name__ == "add_cyclical_date_encoding"

    def test_step_accumulation_order(self):
        """Test steps are accumulated in correct order."""
        cleaner = GaugeCleaner()
        cleaner.trim_to_column("streamflow")
        cleaner.clip_columns(["streamflow"], min_value=0.0)
        cleaner.fill_na_columns(["temperature"], 0.0)

        # Should be in order of addition
        assert cleaner.steps[0][0].__name__ == "trim_to_column"
        assert cleaner.steps[1][0].__name__ == "clip_columns"
        assert cleaner.steps[2][0].__name__ == "fill_na_columns"

    def test_apply_empty_pipeline(self):
        """Test apply method on empty pipeline returns input unchanged."""
        input_lf = pl.LazyFrame({"date": ["2020-01-01", "2020-01-02"], "streamflow": [1.0, 2.0]})
        cleaner = GaugeCleaner()

        result = cleaner.apply(input_lf)

        # Should return the same LazyFrame reference
        assert result is input_lf

    def test_apply_sequential_execution_with_mocks(self):
        """Test apply method executes steps in correct order using mocks."""
        # Create mock functions
        mock_func1 = Mock(return_value="result1")
        mock_func2 = Mock(return_value="result2")
        mock_func3 = Mock(return_value="result3")

        # Create cleaner and manually add mock steps
        cleaner = GaugeCleaner()
        cleaner.steps = [
            (mock_func1, {"param1": "value1"}),
            (mock_func2, {"param2": "value2"}),
            (mock_func3, {"param3": "value3"}),
        ]

        input_lf = pl.LazyFrame({"data": [1, 2, 3]})
        result = cleaner.apply(input_lf)

        # Check functions were called in order with correct parameters
        assert mock_func1.call_args_list == [call(input_lf, param1="value1")]
        assert mock_func2.call_args_list == [call("result1", param2="value2")]
        assert mock_func3.call_args_list == [call("result2", param3="value3")]

        # Result should be the final step's output
        assert result == "result3"

    def test_apply_with_real_single_step(self):
        """Test apply method with one real cleaning step."""
        # Create test data with temporal issues
        input_lf = pl.LazyFrame(
            {
                "date": ["2020-01-02", "2020-01-01", "2020-01-03"],  # Out of order
                "streamflow": [2.0, 1.0, 3.0],
            }
        ).with_columns(pl.col("date").str.to_date())

        cleaner = GaugeCleaner().ensure_temporal_consistency()
        result = cleaner.apply(input_lf)

        # Should return LazyFrame
        assert isinstance(result, pl.LazyFrame)

        # Check it's sorted when collected
        collected = result.collect()
        from datetime import date

        assert collected["date"].to_list() == [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
        assert collected["streamflow"].to_list() == [1.0, 2.0, 3.0]

    def test_apply_with_real_multiple_steps(self):
        """Test apply method with multiple real cleaning steps."""
        # Create test data
        input_lf = pl.LazyFrame(
            {
                "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "streamflow": [-1.0, 2.0, 100.0],  # Needs clipping
                "temperature": [10.0, None, 15.0],  # Needs filling
            }
        ).with_columns(pl.col("date").str.to_date())

        cleaner = (
            GaugeCleaner()
            .ensure_temporal_consistency()
            .fill_na_columns(["temperature"], 12.0)
            .clip_columns(["streamflow"], min_value=0.0, max_value=50.0)
        )

        result = cleaner.apply(input_lf)

        # Should return LazyFrame
        assert isinstance(result, pl.LazyFrame)

        # Check transformations applied correctly
        collected = result.collect()
        assert collected["streamflow"].to_list() == [0.0, 2.0, 50.0]  # Clipped
        assert collected["temperature"].to_list() == [10.0, 12.0, 15.0]  # Filled

    def test_apply_preserves_lazy_evaluation(self):
        """Test apply method preserves lazy evaluation."""
        input_lf = pl.LazyFrame({"date": ["2020-01-01"], "streamflow": [1.0]}).with_columns(
            pl.col("date").str.to_date()
        )
        cleaner = GaugeCleaner().ensure_temporal_consistency()

        result = cleaner.apply(input_lf)

        # Should still be LazyFrame, not DataFrame
        assert isinstance(result, pl.LazyFrame)
        assert not isinstance(result, pl.DataFrame)

    def test_realistic_hydrological_pipeline(self):
        """Test realistic hydrological data cleaning pipeline."""
        # Create realistic hydrological time series data
        input_lf = pl.LazyFrame(
            {
                "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
                "streamflow": [None, 5.0, None, -1.0],  # Has nulls and negative value
                "precipitation": [0.0, 2.5, None, 1.2],  # Has null
                "temperature": [10.0, 15.0, 12.0, 8.0],
            }
        ).with_columns(pl.col("date").str.to_date())

        # Build comprehensive cleaning pipeline
        cleaner = (
            GaugeCleaner()
            .ensure_temporal_consistency()
            .trim_to_column("streamflow")  # Trims to first->last valid streamflow (rows 2-4)
            .fill_na_columns(["precipitation"], 0.0)
            .clip_columns(["streamflow"], min_value=0.0)
            .add_cyclical_date_encoding()
        )

        result = cleaner.apply(input_lf)
        collected = result.collect()

        # Verify LazyFrame returned
        assert isinstance(result, pl.LazyFrame)

        # Should have trimmed to valid streamflow range (from first to last non-null = rows 2-4)
        assert len(collected) == 3
        from datetime import date

        assert collected["date"].to_list() == [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)]

        # Should have filled precipitation nulls and clipped negative streamflow
        assert collected["precipitation"].to_list() == [2.5, 0.0, 1.2]
        assert collected["streamflow"].to_list() == [5.0, None, 0.0]  # null preserved, -1.0 clipped

        # Should have cyclical encoding columns
        assert "sin_day_of_year" in collected.columns
        assert "cos_day_of_year" in collected.columns

    def test_error_propagation(self):
        """Test that errors from atomic cleaners propagate correctly."""
        # Create cleaner with step that will fail
        cleaner = GaugeCleaner().ensure_temporal_consistency(date_column="nonexistent")

        input_lf = pl.LazyFrame({"date": ["2020-01-01"], "streamflow": [1.0]})

        # Should raise ValueError from ensure_temporal_consistency
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            result = cleaner.apply(input_lf)
            result.collect()  # Force evaluation to trigger error

    def test_builder_pattern_immutability(self):
        """Test that builder methods don't modify existing cleaner instances."""
        cleaner1 = GaugeCleaner()
        cleaner2 = cleaner1.ensure_temporal_consistency()

        # Should be same instance (builder pattern)
        assert cleaner1 is cleaner2

        # But calling methods should accumulate steps
        original_steps = len(cleaner1.steps)
        cleaner1.trim_to_column("streamflow")

        assert len(cleaner1.steps) == original_steps + 1

    def test_multiple_independent_cleaners(self):
        """Test that multiple cleaner instances are independent."""
        cleaner1 = GaugeCleaner().ensure_temporal_consistency()
        cleaner2 = GaugeCleaner().trim_to_column("streamflow")

        assert len(cleaner1.steps) == 1
        assert len(cleaner2.steps) == 1
        assert cleaner1.steps[0][0].__name__ == "ensure_temporal_consistency"
        assert cleaner2.steps[0][0].__name__ == "trim_to_column"

    def test_apply_does_not_mutate_original(self):
        """Test that apply() doesn't modify the input LazyFrame."""
        # LazyFrames are immutable, but good to verify
        input_lf = pl.LazyFrame({"date": ["2020-01-01", "2020-01-02"], "streamflow": [1.0, 2.0]}).with_columns(
            pl.col("date").str.to_date()
        )

        # Store reference to original
        original_lf = input_lf

        cleaner = GaugeCleaner().ensure_temporal_consistency()
        result = cleaner.apply(input_lf)

        # Original should be unchanged (same reference)
        assert input_lf is original_lf

        # Result should be different LazyFrame
        assert result is not input_lf

        # Original should still have same schema and can be collected
        original_collected = input_lf.collect()
        assert len(original_collected) == 2
        assert "date" in original_collected.columns
        assert "streamflow" in original_collected.columns

    def test_builder_always_returns_self(self):
        """Test that all builder methods return self."""
        cleaner = GaugeCleaner()

        # Test each builder method returns the same instance
        methods_and_args = [
            ("ensure_temporal_consistency", []),
            ("trim_to_column", ["streamflow"]),
            ("fill_na_columns", [["precipitation"], 0.0]),
            ("clip_columns", [["streamflow"]]),
            ("add_cyclical_date_encoding", []),
        ]

        for method_name, args in methods_and_args:
            method = getattr(cleaner, method_name)
            result = method(*args)
            assert result is cleaner, f"Method {method_name} should return self"

        # Also test that we can chain all methods and they all return the same instance
        chained_result = (
            cleaner.ensure_temporal_consistency()
            .trim_to_column("streamflow")
            .fill_na_columns(["temp"], 0.0)
            .clip_columns(["flow"], min_value=0.0)
            .add_cyclical_date_encoding()
        )

        assert chained_result is cleaner

    def test_apply_is_idempotent(self):
        """Test that calling apply() multiple times with same cleaner works."""
        input_lf = pl.LazyFrame({"date": ["2020-01-01", "2020-01-02"], "streamflow": [1.0, 2.0]}).with_columns(
            pl.col("date").str.to_date()
        )

        cleaner = GaugeCleaner().ensure_temporal_consistency()

        result1 = cleaner.apply(input_lf)
        result2 = cleaner.apply(input_lf)

        # Both calls should work and produce equivalent results
        assert result1.collect().equals(result2.collect())

        # Cleaner should still have same steps
        assert len(cleaner.steps) == 1
