import polars as pl
import pytest

from transfer_learning_publication.cleaners.train_val_test import train_val_test_split


class TestTrainValTestSplit:
    def test_basic_split_standard_proportions(self):
        """Test basic functionality with 60/20/20 split."""
        data = pl.LazyFrame({"value": list(range(10)), "category": ["A"] * 10})

        train_lf, val_lf, test_lf = train_val_test_split(data, 0.6, 0.2)

        train = train_lf.collect()
        val = val_lf.collect()
        test = test_lf.collect()

        # Check sizes: 10 * 0.6 = 6, 10 * 0.2 = 2, remainder = 2
        assert len(train) == 6
        assert len(val) == 2
        assert len(test) == 2

        # Check sequential split - train should be rows 0-5
        assert train["value"].to_list() == [0, 1, 2, 3, 4, 5]
        # val should be rows 6-7
        assert val["value"].to_list() == [6, 7]
        # test should be rows 8-9
        assert test["value"].to_list() == [8, 9]

    def test_split_with_different_proportions(self):
        """Test split with 70/15/15 proportions."""
        data = pl.LazyFrame({"value": list(range(20))})

        train_lf, val_lf, test_lf = train_val_test_split(data, 0.7, 0.15)

        train = train_lf.collect()
        val = val_lf.collect()
        test = test_lf.collect()

        # 20 * 0.7 = 14, 20 * 0.15 = 3, remainder = 3
        assert len(train) == 14
        assert len(val) == 3
        assert len(test) == 3

    def test_split_preserves_schema(self):
        """Test that column types and schema are preserved."""
        data = pl.LazyFrame(
            {"int_col": [1, 2, 3, 4, 5], "float_col": [1.1, 2.2, 3.3, 4.4, 5.5], "str_col": ["a", "b", "c", "d", "e"]}
        )

        train_lf, val_lf, test_lf = train_val_test_split(data, 0.6, 0.2)

        # All splits should have same schema as original
        original_schema = data.collect_schema()
        assert train_lf.collect_schema() == original_schema
        assert val_lf.collect_schema() == original_schema
        assert test_lf.collect_schema() == original_schema

    def test_returns_lazy_frames(self):
        """Test that function returns LazyFrames, not DataFrames."""
        data = pl.LazyFrame({"value": [1, 2, 3, 4, 5]})

        train_lf, val_lf, test_lf = train_val_test_split(data, 0.6, 0.2)

        assert isinstance(train_lf, pl.LazyFrame)
        assert isinstance(val_lf, pl.LazyFrame)
        assert isinstance(test_lf, pl.LazyFrame)

    def test_only_train_set_proportion_one(self):
        """Test creating only train set with proportion 1.0."""
        data = pl.LazyFrame({"value": [1, 2, 3, 4, 5]})

        with pytest.warns(UserWarning, match="train_prop is 1.0. Creating empty validation and test sets."):
            train_lf, val_lf, test_lf = train_val_test_split(data, 1.0, 0.0)

        train = train_lf.collect()
        val = val_lf.collect()
        test = test_lf.collect()

        assert len(train) == 5
        assert len(val) == 0
        assert len(test) == 0

        assert train["value"].to_list() == [1, 2, 3, 4, 5]

    def test_no_test_set_proportions_sum_to_one(self):
        """Test creating no test set when train + val = 1.0."""
        data = pl.LazyFrame({"value": list(range(10))})

        with pytest.warns(UserWarning, match="No test set created \\(train_prop \\+ val_prop = 1.0\\)."):
            train_lf, val_lf, test_lf = train_val_test_split(data, 0.8, 0.2)

        train = train_lf.collect()
        val = val_lf.collect()
        test = test_lf.collect()

        assert len(train) == 8
        assert len(val) == 2
        assert len(test) == 0

    def test_no_validation_set(self):
        """Test creating no validation set when val_prop = 0."""
        data = pl.LazyFrame({"value": list(range(10))})

        with pytest.warns(UserWarning, match="No validation set created \\(val_prop = 0\\)."):
            train_lf, val_lf, test_lf = train_val_test_split(data, 0.7, 0.0)

        train = train_lf.collect()
        val = val_lf.collect()
        test = test_lf.collect()

        assert len(train) == 7
        assert len(val) == 0
        assert len(test) == 3

    def test_empty_dataframe(self):
        """Test handling of empty input DataFrame."""
        data = pl.LazyFrame({"value": []}, schema={"value": pl.Int64})

        with pytest.warns(UserWarning, match="Input LazyFrame is empty. Returning three empty LazyFrames."):
            train_lf, val_lf, test_lf = train_val_test_split(data, 0.6, 0.2)

        train = train_lf.collect()
        val = val_lf.collect()
        test = test_lf.collect()

        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0

        # Should preserve schema
        assert train.schema == data.collect_schema()

    def test_single_row_dataframe(self):
        """Test handling of single row DataFrame."""
        data = pl.LazyFrame({"value": [42]})

        train_lf, val_lf, test_lf = train_val_test_split(data, 0.6, 0.2)

        train = train_lf.collect()
        val = val_lf.collect()
        test = test_lf.collect()

        # With 1 row: int(1 * 0.6) = 0, int(1 * 0.2) = 0, remainder = 1
        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 1
        assert test["value"].to_list() == [42]

    def test_rounding_behavior_with_small_dataset(self):
        """Test rounding behavior with dataset that doesn't divide evenly."""
        data = pl.LazyFrame({"value": [1, 2, 3]})

        train_lf, val_lf, test_lf = train_val_test_split(data, 0.5, 0.3)

        train = train_lf.collect()
        val = val_lf.collect()
        test = test_lf.collect()

        # 3 * 0.5 = 1.5 -> int(1.5) = 1
        # 3 * 0.3 = 0.9 -> int(0.9) = 0
        # remainder = 3 - 1 - 0 = 2
        assert len(train) == 1
        assert len(val) == 0
        assert len(test) == 2

        assert train["value"].to_list() == [1]
        assert test["value"].to_list() == [2, 3]

    def test_negative_proportions_error(self):
        """Test error handling for negative proportions."""
        data = pl.LazyFrame({"value": [1, 2, 3]})

        with pytest.raises(ValueError, match="Proportions must be non-negative. Got train_prop=-0.1, val_prop=0.2"):
            train_val_test_split(data, -0.1, 0.2)

        with pytest.raises(ValueError, match="Proportions must be non-negative. Got train_prop=0.6, val_prop=-0.2"):
            train_val_test_split(data, 0.6, -0.2)

    def test_proportions_greater_than_one_error(self):
        """Test error handling for proportions > 1."""
        data = pl.LazyFrame({"value": [1, 2, 3]})

        with pytest.raises(ValueError, match="Proportions must be <= 1. Got train_prop=1.5, val_prop=0.2"):
            train_val_test_split(data, 1.5, 0.2)

        with pytest.raises(ValueError, match="Proportions must be <= 1. Got train_prop=0.6, val_prop=1.2"):
            train_val_test_split(data, 0.6, 1.2)

    def test_proportions_sum_exceeds_one_error(self):
        """Test error handling when sum of proportions exceeds 1.0."""
        data = pl.LazyFrame({"value": [1, 2, 3]})

        with pytest.raises(ValueError, match="Sum of proportions \\(1.1\\) exceeds 1.0"):
            train_val_test_split(data, 0.7, 0.4)

    def test_zero_proportions_valid(self):
        """Test that zero proportions are valid."""
        data = pl.LazyFrame({"value": list(range(10))})

        # Test with zero train proportion
        train_lf, val_lf, test_lf = train_val_test_split(data, 0.0, 0.5)

        train = train_lf.collect()
        val = val_lf.collect()
        test = test_lf.collect()

        assert len(train) == 0
        assert len(val) == 5
        assert len(test) == 5

    def test_maintains_lazy_evaluation(self):
        """Test that operations don't force collection unnecessarily."""
        # Create a LazyFrame that would be expensive to collect
        data = pl.LazyFrame({"value": list(range(1000))})

        # The function should work without collecting the full dataset
        # (it only collects to get the length)
        train_lf, val_lf, test_lf = train_val_test_split(data, 0.6, 0.2)

        # Results should still be lazy
        assert isinstance(train_lf, pl.LazyFrame)
        assert isinstance(val_lf, pl.LazyFrame)
        assert isinstance(test_lf, pl.LazyFrame)

        # Test that we can collect each split
        train = train_lf.collect()
        val = val_lf.collect()
        test = test_lf.collect()

        assert len(train) == 600
        assert len(val) == 200
        assert len(test) == 200

    def test_column_order_preservation(self):
        """Test that column order is preserved in all splits."""
        data = pl.LazyFrame(
            {
                "first": [1, 2, 3, 4, 5],
                "second": ["a", "b", "c", "d", "e"],
                "third": [1.1, 2.2, 3.3, 4.4, 5.5],
                "fourth": [True, False, True, False, True],
            }
        )

        train_lf, val_lf, test_lf = train_val_test_split(data, 0.6, 0.2)

        original_columns = data.collect().columns

        assert train_lf.collect().columns == original_columns
        assert val_lf.collect().columns == original_columns
        assert test_lf.collect().columns == original_columns
