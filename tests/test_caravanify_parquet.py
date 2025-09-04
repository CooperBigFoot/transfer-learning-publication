from datetime import date, datetime
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from transfer_learning_publication.data.caravanify_parquet import CaravanDataSource


@pytest.fixture
def temp_hive_data(tmp_path):
    """Create a temporary hive-partitioned directory structure with test data."""
    base_path = tmp_path / "caravan_data"
    base_path.mkdir()

    # Create test data for two regions
    regions = ["camels", "hysets"]
    gauge_ids = {"camels": ["G01013500", "G01030500", "G01054200"], "hysets": ["02LE024", "02OA016"]}

    # Create timeseries data
    for region in regions:
        for gauge_id in gauge_ids[region]:
            ts_path = base_path / f"REGION_NAME={region}" / "data_type=timeseries" / f"gauge_id={gauge_id}"
            ts_path.mkdir(parents=True)

            # Create sample timeseries data
            dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
            df = pl.DataFrame(
                {
                    "date": dates,
                    "streamflow": [10.5, 12.3, 14.1, 11.8, 13.5, 15.2, 14.8, 16.1, 17.3, 18.5],
                    "precipitation": [0.0, 2.5, 5.1, 0.5, 1.2, 3.3, 0.0, 4.5, 2.1, 0.8],
                    "temperature": [5.2, 6.1, 4.8, 3.9, 5.5, 6.8, 7.2, 8.1, 7.5, 6.9],
                }
            )
            # Note: gauge_id is a hive partition column and will be added automatically by polars
            df.write_parquet(ts_path / "data.parquet")

    # Create attributes data with different schemas per attribute type
    attribute_types = ["caravan", "hydroatlas"]

    for region in regions:
        for attr_type in attribute_types:
            attr_path = base_path / f"REGION_NAME={region}" / "data_type=attributes" / f"attribute_type={attr_type}"
            attr_path.mkdir(parents=True)

            # Create different schemas for each attribute type
            if attr_type == "caravan":
                df = pl.DataFrame(
                    {
                        "gauge_id": gauge_ids[region],
                        "area": [100.5, 250.3, 175.8] if region == "camels" else [320.1, 410.5],
                        "elevation": [500, 750, 600] if region == "camels" else [1200, 950],
                        "slope": [0.05, 0.12, 0.08] if region == "camels" else [0.15, 0.10],
                    }
                )
            else:  # hydroatlas
                df = pl.DataFrame(
                    {
                        "gauge_id": gauge_ids[region],
                        "forest_cover": [0.45, 0.62, 0.38] if region == "camels" else [0.71, 0.55],
                        "urban_area": [0.10, 0.05, 0.15] if region == "camels" else [0.02, 0.08],
                    }
                )

            df.write_parquet(attr_path / "data.parquet")

    return base_path


@pytest.fixture
def temp_hive_data_string_dates(tmp_path):
    """Create test data with string date format."""
    base_path = tmp_path / "caravan_string_dates"
    base_path.mkdir()

    ts_path = base_path / "REGION_NAME=test" / "data_type=timeseries" / "gauge_id=test001"
    ts_path.mkdir(parents=True)

    # Create timeseries with string dates
    df = pl.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "streamflow": [10.0, 11.0, 12.0],
        }
    )
    df.write_parquet(ts_path / "data.parquet")

    return base_path


@pytest.fixture
def temp_hive_data_datetime(tmp_path):
    """Create test data with datetime format."""
    base_path = tmp_path / "caravan_datetime"
    base_path.mkdir()

    ts_path = base_path / "REGION_NAME=test" / "data_type=timeseries" / "gauge_id=test001"
    ts_path.mkdir(parents=True)

    # Create timeseries with datetime format
    df = pl.DataFrame(
        {
            "date": [datetime(2020, 1, 1, 12, 0), datetime(2020, 1, 2, 12, 0), datetime(2020, 1, 3, 12, 0)],
            "streamflow": [10.0, 11.0, 12.0],
        }
    )
    df.write_parquet(ts_path / "data.parquet")

    return base_path


@pytest.fixture
def empty_hive_data(tmp_path):
    """Create an empty hive directory structure."""
    base_path = tmp_path / "empty_caravan"
    base_path.mkdir()
    return base_path


class TestCaravanDataSourceInit:
    """Test initialization of CaravanDataSource."""

    def test_init_with_base_path_only(self, temp_hive_data):
        """Test initialization without region filter."""
        ds = CaravanDataSource(temp_hive_data)
        assert ds.base_path == Path(temp_hive_data)
        assert ds.region is None
        assert "REGION_NAME=*" in ds._ts_glob
        assert "REGION_NAME=*" in ds._attr_glob

    def test_init_with_specific_region(self, temp_hive_data):
        """Test initialization with specific region."""
        ds = CaravanDataSource(temp_hive_data, region="camels")
        assert ds.base_path == Path(temp_hive_data)
        assert ds.region == "camels"
        assert "REGION_NAME=camels" in ds._ts_glob
        assert "REGION_NAME=camels" in ds._attr_glob

    def test_init_with_string_path(self, temp_hive_data):
        """Test initialization with string path."""
        ds = CaravanDataSource(str(temp_hive_data))
        assert ds.base_path == Path(temp_hive_data)
        assert isinstance(ds.base_path, Path)


class TestListMethods:
    """Test listing methods."""

    def test_list_regions(self, temp_hive_data):
        """Test listing all available regions."""
        ds = CaravanDataSource(temp_hive_data)
        regions = ds.list_regions()
        assert regions == ["camels", "hysets"]

    def test_list_regions_empty(self, empty_hive_data):
        """Test listing regions from empty directory."""
        ds = CaravanDataSource(empty_hive_data)
        regions = ds.list_regions()
        assert regions == []

    def test_list_gauge_ids(self, temp_hive_data):
        """Test listing all gauge IDs."""
        ds = CaravanDataSource(temp_hive_data)
        gauge_ids = ds.list_gauge_ids()
        expected = ["02LE024", "02OA016", "G01013500", "G01030500", "G01054200"]
        assert sorted(gauge_ids) == sorted(expected)

    def test_list_gauge_ids_with_region_filter(self, temp_hive_data):
        """Test listing gauge IDs for specific region."""
        ds = CaravanDataSource(temp_hive_data, region="camels")
        gauge_ids = ds.list_gauge_ids()
        assert sorted(gauge_ids) == ["G01013500", "G01030500", "G01054200"]

    def test_list_timeseries_variables(self, temp_hive_data):
        """Test listing timeseries variables."""
        ds = CaravanDataSource(temp_hive_data)
        variables = ds.list_timeseries_variables()
        assert sorted(variables) == ["precipitation", "streamflow", "temperature"]

    def test_list_static_attributes(self, temp_hive_data):
        """Test listing all attribute columns."""
        ds = CaravanDataSource(temp_hive_data)
        attributes = ds.list_static_attributes()
        expected = ["area", "elevation", "forest_cover", "slope", "urban_area"]
        assert sorted(attributes) == sorted(expected)

    def test_list_static_attributes_empty(self, empty_hive_data):
        """Test listing attributes from empty directory."""
        ds = CaravanDataSource(empty_hive_data)
        attributes = ds.list_static_attributes()
        assert attributes == []


class TestDateRanges:
    """Test date range methods."""

    def test_get_date_ranges(self, temp_hive_data):
        """Test getting date ranges for all gauges."""
        ds = CaravanDataSource(temp_hive_data)
        date_ranges = ds.get_date_ranges().collect()

        assert len(date_ranges) == 5  # Total number of gauges
        assert set(date_ranges["gauge_id"]) == {"G01013500", "G01030500", "G01054200", "02LE024", "02OA016"}

        # Check date range values
        for row in date_ranges.iter_rows(named=True):
            assert row["min_date"] == date(2020, 1, 1)
            assert row["max_date"] == date(2020, 1, 10)

    def test_get_date_ranges_with_gauge_filter(self, temp_hive_data):
        """Test getting date ranges for specific gauges."""
        ds = CaravanDataSource(temp_hive_data)
        gauge_ids = ["G01013500", "02LE024"]
        date_ranges = ds.get_date_ranges(gauge_ids=gauge_ids).collect()

        assert len(date_ranges) == 2
        assert set(date_ranges["gauge_id"]) == set(gauge_ids)

    def test_get_date_ranges_with_string_dates(self, temp_hive_data_string_dates):
        """Test handling string date format."""
        ds = CaravanDataSource(temp_hive_data_string_dates)
        date_ranges = ds.get_date_ranges().collect()

        assert len(date_ranges) == 1
        assert date_ranges["min_date"][0] == date(2020, 1, 1)
        assert date_ranges["max_date"][0] == date(2020, 1, 3)

    def test_get_date_ranges_with_datetime_dates(self, temp_hive_data_datetime):
        """Test handling datetime format."""
        ds = CaravanDataSource(temp_hive_data_datetime)
        date_ranges = ds.get_date_ranges().collect()

        assert len(date_ranges) == 1
        assert date_ranges["min_date"][0] == date(2020, 1, 1)
        assert date_ranges["max_date"][0] == date(2020, 1, 3)


class TestTimeseriesData:
    """Test timeseries data retrieval."""

    def test_get_timeseries_all_data(self, temp_hive_data):
        """Test getting all timeseries data."""
        ds = CaravanDataSource(temp_hive_data)
        ts_data = ds.get_timeseries().collect()

        # 5 gauges * 10 days = 50 rows
        assert len(ts_data) == 50
        # Note: data_type is added by hive partitioning
        expected_columns = {
            "REGION_NAME",
            "gauge_id",
            "date",
            "streamflow",
            "precipitation",
            "temperature",
            "data_type",
        }
        assert set(ts_data.columns) == expected_columns

    def test_get_timeseries_with_gauge_filter(self, temp_hive_data):
        """Test filtering by gauge IDs."""
        ds = CaravanDataSource(temp_hive_data)
        gauge_ids = ["G01013500", "02LE024"]
        ts_data = ds.get_timeseries(gauge_ids=gauge_ids).collect()

        # 2 gauges * 10 days = 20 rows
        assert len(ts_data) == 20
        assert set(ts_data["gauge_id"].unique()) == set(gauge_ids)

    def test_get_timeseries_with_variable_filter(self, temp_hive_data):
        """Test filtering by variables."""
        ds = CaravanDataSource(temp_hive_data)
        variables = ["streamflow", "temperature"]
        ts_data = ds.get_timeseries(variables=variables).collect()

        assert len(ts_data) == 50
        assert set(ts_data.columns) == {"REGION_NAME", "gauge_id", "date", "streamflow", "temperature"}
        assert "precipitation" not in ts_data.columns

    def test_get_timeseries_with_date_range(self, temp_hive_data):
        """Test filtering by date range."""
        ds = CaravanDataSource(temp_hive_data)
        date_range = ("2020-01-03", "2020-01-05")
        ts_data = ds.get_timeseries(date_range=date_range).collect()

        # 5 gauges * 3 days = 15 rows
        assert len(ts_data) == 15

        min_date = ts_data["date"].min()
        max_date = ts_data["date"].max()
        assert min_date == date(2020, 1, 3)
        assert max_date == date(2020, 1, 5)

    def test_get_timeseries_combined_filters(self, temp_hive_data):
        """Test multiple filters combined."""
        ds = CaravanDataSource(temp_hive_data, region="camels")
        ts_data = ds.get_timeseries(
            gauge_ids=["G01013500"], variables=["streamflow"], date_range=("2020-01-01", "2020-01-03")
        ).collect()

        assert len(ts_data) == 3
        assert ts_data["gauge_id"].unique()[0] == "G01013500"
        assert set(ts_data.columns) == {"REGION_NAME", "gauge_id", "date", "streamflow"}
        assert ts_data["REGION_NAME"].unique()[0] == "camels"

    def test_get_timeseries_lazy_evaluation(self, temp_hive_data):
        """Test that get_timeseries returns a LazyFrame."""
        ds = CaravanDataSource(temp_hive_data)
        ts_lazy = ds.get_timeseries()

        assert isinstance(ts_lazy, pl.LazyFrame)
        # LazyFrame should not be evaluated until collect()
        ts_data = ts_lazy.limit(5).collect()
        assert len(ts_data) == 5


class TestStaticAttributes:
    """Test static attributes retrieval."""

    def test_get_static_attributes_all(self, temp_hive_data):
        """Test getting all attributes."""
        ds = CaravanDataSource(temp_hive_data)
        attrs = ds.get_static_attributes().collect()

        # 5 gauges * 2 attribute types = 10 rows
        assert len(attrs) == 10
        assert "gauge_id" in attrs.columns
        assert "attribute_type" in attrs.columns
        assert "REGION_NAME" in attrs.columns

    def test_get_static_attributes_with_gauge_filter(self, temp_hive_data):
        """Test filtering by gauge IDs."""
        ds = CaravanDataSource(temp_hive_data)
        gauge_ids = ["G01013500", "02LE024"]
        attrs = ds.get_static_attributes(gauge_ids=gauge_ids).collect()

        # 2 gauges * 2 attribute types = 4 rows
        assert len(attrs) == 4
        assert set(attrs["gauge_id"].unique()) == set(gauge_ids)

    def test_get_static_attributes_with_column_filter(self, temp_hive_data):
        """Test filtering by columns."""
        ds = CaravanDataSource(temp_hive_data)
        columns = ["area", "elevation"]
        attrs = ds.get_static_attributes(columns=columns).collect()

        # Only caravan type has these columns
        assert "area" in attrs.columns
        assert "elevation" in attrs.columns
        assert "forest_cover" not in attrs.columns
        assert "urban_area" not in attrs.columns

    def test_get_static_attributes_with_type_filter(self, temp_hive_data):
        """Test filtering by attribute types."""
        ds = CaravanDataSource(temp_hive_data)
        attrs = ds.get_static_attributes(attribute_types=["hydroatlas"]).collect()

        # 5 gauges * 1 attribute type = 5 rows
        assert len(attrs) == 5
        assert attrs["attribute_type"].unique()[0] == "hydroatlas"
        assert "forest_cover" in attrs.columns
        assert "urban_area" in attrs.columns
        # Note: diagonal concatenation may include columns with nulls from other attribute types
        # Check that caravan-specific columns are null for hydroatlas rows
        if "area" in attrs.columns:
            assert attrs["area"].is_null().all()
        if "elevation" in attrs.columns:
            assert attrs["elevation"].is_null().all()

    def test_get_static_attributes_schema_differences(self, temp_hive_data):
        """Test handling different schemas across files."""
        ds = CaravanDataSource(temp_hive_data)
        attrs = ds.get_static_attributes().collect()

        # Should handle missing columns with nulls
        assert len(attrs) == 10

        # Check that schema union worked (diagonal concatenation)
        # Note: data_type is added by hive partitioning
        all_columns = {
            "gauge_id",
            "REGION_NAME",
            "attribute_type",
            "area",
            "elevation",
            "slope",
            "forest_cover",
            "urban_area",
            "data_type",
        }
        assert set(attrs.columns) == all_columns

        # Check nulls are properly placed for missing columns
        caravan_rows = attrs.filter(pl.col("attribute_type") == "caravan")
        assert caravan_rows["forest_cover"].is_null().all()
        assert caravan_rows["urban_area"].is_null().all()

        hydroatlas_rows = attrs.filter(pl.col("attribute_type") == "hydroatlas")
        assert hydroatlas_rows["area"].is_null().all()
        assert hydroatlas_rows["elevation"].is_null().all()

    def test_get_static_attributes_empty_result(self, temp_hive_data):
        """Test handling empty results."""
        ds = CaravanDataSource(temp_hive_data)
        attrs = ds.get_static_attributes(gauge_ids=["nonexistent_gauge"]).collect()

        assert len(attrs) == 0
        assert "gauge_id" in attrs.columns

    def test_get_static_attributes_lazy_evaluation(self, temp_hive_data):
        """Test that get_static_attributes returns a LazyFrame."""
        ds = CaravanDataSource(temp_hive_data)
        attrs_lazy = ds.get_static_attributes()

        assert isinstance(attrs_lazy, pl.LazyFrame)
        # LazyFrame should not be evaluated until collect()
        attrs_data = attrs_lazy.limit(2).collect()
        assert len(attrs_data) == 2


class TestErrorHandling:
    """Test error handling."""

    def test_empty_directory_timeseries(self, empty_hive_data):
        """Test handling empty directory for timeseries."""
        ds = CaravanDataSource(empty_hive_data)

        # Should not raise error during list operations
        gauge_ids = ds.list_gauge_ids()
        assert gauge_ids == []

        variables = ds.list_timeseries_variables()
        assert variables == []

    def test_empty_directory_attributes(self, empty_hive_data):
        """Test handling empty directory for attributes."""
        ds = CaravanDataSource(empty_hive_data)

        attributes = ds.list_static_attributes()
        assert attributes == []

        # get_static_attributes should return empty LazyFrame
        attrs = ds.get_static_attributes().collect()
        assert len(attrs) == 0
        assert "gauge_id" in attrs.columns

    def test_nonexistent_path(self):
        """Test initialization with non-existent path."""
        # Should not raise during initialization
        ds = CaravanDataSource("/nonexistent/path")
        assert ds.base_path == Path("/nonexistent/path")

        # Should handle gracefully when trying to list
        regions = ds.list_regions()
        assert regions == []

    def test_filter_nonexistent_gauge(self, temp_hive_data):
        """Test filtering with non-existent gauge ID."""
        ds = CaravanDataSource(temp_hive_data)

        ts_data = ds.get_timeseries(gauge_ids=["nonexistent"]).collect()
        assert len(ts_data) == 0

        attrs = ds.get_static_attributes(gauge_ids=["nonexistent"]).collect()
        assert len(attrs) == 0

    def test_filter_nonexistent_variable(self, temp_hive_data):
        """Test filtering with non-existent variable."""
        ds = CaravanDataSource(temp_hive_data)

        # Should only return metadata columns when variable doesn't exist
        ts_data = ds.get_timeseries(variables=["nonexistent"]).collect()
        assert len(ts_data) == 50  # Still returns rows
        assert "nonexistent" not in ts_data.columns
        assert set(ts_data.columns) == {"REGION_NAME", "gauge_id", "date"}

    def test_invalid_date_range_format(self, temp_hive_data):
        """Test date range with valid format strings."""
        ds = CaravanDataSource(temp_hive_data)

        # Should handle date strings that can be parsed
        date_range = ("2020-01-01", "2020-01-05")
        ts_data = ds.get_timeseries(date_range=date_range).collect()
        assert len(ts_data) == 25  # 5 gauges * 5 days


class TestPartitionPruning:
    """Test that partition pruning is working efficiently."""

    def test_region_partition_pruning(self, temp_hive_data):
        """Test that region filter uses partition pruning."""
        # With region filter, should only read from one region's partitions
        ds = CaravanDataSource(temp_hive_data, region="camels")

        ts_data = ds.get_timeseries().collect()
        assert ts_data["REGION_NAME"].unique()[0] == "camels"
        assert len(ts_data) == 30  # 3 gauges * 10 days

        attrs = ds.get_static_attributes().collect()
        assert attrs["REGION_NAME"].unique()[0] == "camels"
        assert len(attrs) == 6  # 3 gauges * 2 attribute types

    def test_gauge_partition_pruning_timeseries(self, temp_hive_data):
        """Test that gauge filter uses partition pruning for timeseries."""
        ds = CaravanDataSource(temp_hive_data)

        # Should only read from specific gauge partitions
        gauge_ids = ["G01013500"]
        ts_data = ds.get_timeseries(gauge_ids=gauge_ids).collect()

        assert ts_data["gauge_id"].unique()[0] == "G01013500"
        assert len(ts_data) == 10  # 1 gauge * 10 days

    def test_attribute_type_partition_pruning(self, temp_hive_data):
        """Test that attribute type filter uses partition pruning."""
        ds = CaravanDataSource(temp_hive_data)

        # Should only read from specific attribute type partitions
        attrs = ds.get_static_attributes(attribute_types=["caravan"]).collect()

        assert attrs["attribute_type"].unique()[0] == "caravan"
        assert len(attrs) == 5  # 5 gauges * 1 attribute type
        assert "area" in attrs.columns
        assert "forest_cover" not in attrs.columns or attrs["forest_cover"].is_null().all()

    def test_get_static_attributes_with_unreadable_file(self, temp_hive_data, capfd):
        """Test warning when a file can't be read."""
        # Corrupt one of the parquet files
        attr_file = list(temp_hive_data.glob("**/attribute_type=*/data.parquet"))[0]
        attr_file.write_text("corrupted")

        ds = CaravanDataSource(temp_hive_data)
        attrs = ds.get_static_attributes().collect()

        # Should still return data from readable files
        assert len(attrs) > 0

        # Should print warning
        captured = capfd.readouterr()
        assert "Warning: Could not read" in captured.out

    def test_completely_disjoint_schemas(self, tmp_path):
        """Test handling completely different schemas between files."""
        base_path = tmp_path / "disjoint"

        # Create files with no overlapping columns except gauge_id
        path1 = base_path / "REGION_NAME=r1" / "data_type=attributes" / "attribute_type=t1"
        path1.mkdir(parents=True)
        pl.DataFrame({"gauge_id": ["g1"], "col_a": [1]}).write_parquet(path1 / "data.parquet")

        path2 = base_path / "REGION_NAME=r2" / "data_type=attributes" / "attribute_type=t2"
        path2.mkdir(parents=True)
        pl.DataFrame({"gauge_id": ["g2"], "col_b": [2]}).write_parquet(path2 / "data.parquet")

        ds = CaravanDataSource(base_path)
        attrs = ds.get_static_attributes().collect()

        # Should handle with nulls
        assert set(attrs.columns) >= {"gauge_id", "col_a", "col_b"}
        assert attrs.filter(pl.col("gauge_id") == "g1")["col_b"].is_null().all()
        assert attrs.filter(pl.col("gauge_id") == "g2")["col_a"].is_null().all()

    def test_date_range_boundary_conditions(self, temp_hive_data):
        """Test date range with exact boundaries."""
        ds = CaravanDataSource(temp_hive_data)

        # Test inclusive boundaries
        ts_data = ds.get_timeseries(date_range=("2020-01-01", "2020-01-01")).collect()
        assert len(ts_data) == 5  # 5 gauges * 1 day
        assert ts_data["date"].unique()[0] == date(2020, 1, 1)


class TestWriteTimeseries:
    """Test write_timeseries method."""

    def test_write_timeseries_basic(self, tmp_path):
        """Test basic timeseries writing functionality."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        # Create sample timeseries data
        df = pl.DataFrame(
            {
                "gauge_id": ["G001", "G001", "G002", "G002"],
                "date": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
                "streamflow": [10.5, 12.0, 15.2, 14.8],
                "temperature": [5.0, 6.2, 4.8, 5.5],
            }
        )

        ds.write_timeseries(df, tmp_path)

        # Verify directory structure
        expected_path = tmp_path / "REGION_NAME=test_region" / "data_type=timeseries"
        assert expected_path.exists()

        # Check gauge partitions
        gauge1_path = expected_path / "gauge_id=G001" / "data.parquet"
        gauge2_path = expected_path / "gauge_id=G002" / "data.parquet"
        assert gauge1_path.exists()
        assert gauge2_path.exists()

        # Verify data can be read back
        gauge1_data = pl.read_parquet(gauge1_path)
        assert len(gauge1_data) == 2
        assert "gauge_id" not in gauge1_data.columns  # Should be removed as it's in partition
        assert set(gauge1_data.columns) == {"date", "streamflow", "temperature"}

    def test_write_timeseries_with_lazyframe(self, tmp_path):
        """Test writing with LazyFrame input."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        df = pl.DataFrame(
            {
                "gauge_id": ["G001"],
                "date": ["2020-01-01"],
                "streamflow": [10.5],
            }
        ).lazy()

        ds.write_timeseries(df, tmp_path)

        # Verify data was written
        expected_file = tmp_path / "REGION_NAME=test_region" / "data_type=timeseries" / "gauge_id=G001" / "data.parquet"
        assert expected_file.exists()

    def test_write_timeseries_no_region_error(self, tmp_path):
        """Test that writing without region raises error."""
        ds = CaravanDataSource(tmp_path)  # No region specified

        df = pl.DataFrame(
            {
                "gauge_id": ["G001"],
                "date": ["2020-01-01"],
                "streamflow": [10.5],
            }
        )

        with pytest.raises(ValueError, match="Region must be set to write timeseries"):
            ds.write_timeseries(df, tmp_path)

    def test_write_timeseries_missing_gauge_id_error(self, tmp_path):
        """Test that missing gauge_id column raises error."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        df = pl.DataFrame(
            {
                "date": ["2020-01-01"],
                "streamflow": [10.5],
            }
        )

        with pytest.raises(ValueError, match="DataFrame must contain 'gauge_id' column"):
            ds.write_timeseries(df, tmp_path)

    def test_write_timeseries_missing_date_warning(self, tmp_path, caplog):
        """Test warning when date column is missing."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        df = pl.DataFrame(
            {
                "gauge_id": ["G001"],
                "streamflow": [10.5],
            }
        )

        ds.write_timeseries(df, tmp_path)

        # Check for warning in log records
        assert "No 'date' column found" in caplog.text

    def test_write_timeseries_overwrite_false_error(self, tmp_path):
        """Test that existing partitions raise error when overwrite=False."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        df = pl.DataFrame(
            {
                "gauge_id": ["G001"],
                "date": ["2020-01-01"],
                "streamflow": [10.5],
            }
        )

        # First write should succeed
        ds.write_timeseries(df, tmp_path)

        # Second write should fail with overwrite=False
        with pytest.raises(ValueError, match="Gauge partitions already exist"):
            ds.write_timeseries(df, tmp_path, overwrite=False)

    def test_write_timeseries_overwrite_true_success(self, tmp_path):
        """Test that overwrite=True replaces existing data."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        # First dataset
        df1 = pl.DataFrame(
            {
                "gauge_id": ["G001"],
                "date": ["2020-01-01"],
                "streamflow": [10.5],
            }
        )
        ds.write_timeseries(df1, tmp_path)

        # Second dataset with different values
        df2 = pl.DataFrame(
            {
                "gauge_id": ["G001"],
                "date": ["2020-01-01"],
                "streamflow": [99.9],
            }
        )
        ds.write_timeseries(df2, tmp_path, overwrite=True)

        # Verify new data was written
        written_file = tmp_path / "REGION_NAME=test_region" / "data_type=timeseries" / "gauge_id=G001" / "data.parquet"
        data = pl.read_parquet(written_file)
        assert data["streamflow"][0] == 99.9

    def test_write_timeseries_multiple_gauges(self, tmp_path):
        """Test writing data for multiple gauges creates separate partitions."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        df = pl.DataFrame(
            {
                "gauge_id": ["G001", "G001", "G002", "G003"],
                "date": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-01"],
                "streamflow": [10.5, 12.0, 15.2, 8.8],
            }
        )

        ds.write_timeseries(df, tmp_path)

        # Check all gauge partitions exist
        base_path = tmp_path / "REGION_NAME=test_region" / "data_type=timeseries"
        assert (base_path / "gauge_id=G001" / "data.parquet").exists()
        assert (base_path / "gauge_id=G002" / "data.parquet").exists()
        assert (base_path / "gauge_id=G003" / "data.parquet").exists()

        # Verify data separation
        g001_data = pl.read_parquet(base_path / "gauge_id=G001" / "data.parquet")
        g002_data = pl.read_parquet(base_path / "gauge_id=G002" / "data.parquet")
        g003_data = pl.read_parquet(base_path / "gauge_id=G003" / "data.parquet")

        assert len(g001_data) == 2
        assert len(g002_data) == 1
        assert len(g003_data) == 1

    def test_write_timeseries_empty_dataframe(self, tmp_path):
        """Test writing empty DataFrame."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        # Empty DataFrame with correct schema
        df = pl.DataFrame(
            {
                "gauge_id": [],
                "date": [],
                "streamflow": [],
            }
        ).cast({"gauge_id": pl.Utf8, "date": pl.Utf8, "streamflow": pl.Float64})

        ds.write_timeseries(df, tmp_path)

        # Should create directory structure but no partition files
        # Directory may or may not be created for empty data - this is implementation dependent
        # The main thing is it shouldn't crash


class TestWriteStaticAttributes:
    """Test write_static_attributes method."""

    def test_write_static_attributes_basic(self, tmp_path):
        """Test basic static attributes writing functionality."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        # Create sample attributes data
        df = pl.DataFrame(
            {
                "gauge_id": ["G001", "G002", "G003"],
                "attribute_type": ["caravan", "caravan", "hydroatlas"],
                "area": [100.5, 250.3, None],
                "elevation": [500, 750, None],
                "forest_cover": [None, None, 0.45],
            }
        )

        ds.write_static_attributes(df, tmp_path)

        # Verify directory structure
        expected_path = tmp_path / "REGION_NAME=test_region" / "data_type=attributes"
        assert expected_path.exists()

        # Check attribute type partitions
        caravan_path = expected_path / "attribute_type=caravan" / "data.parquet"
        hydroatlas_path = expected_path / "attribute_type=hydroatlas" / "data.parquet"
        assert caravan_path.exists()
        assert hydroatlas_path.exists()

        # Verify data can be read back
        caravan_data = pl.read_parquet(caravan_path)
        hydroatlas_data = pl.read_parquet(hydroatlas_path)

        assert len(caravan_data) == 2
        assert len(hydroatlas_data) == 1
        assert "attribute_type" not in caravan_data.columns  # Should be removed as it's in partition
        assert "attribute_type" not in hydroatlas_data.columns

    def test_write_static_attributes_with_lazyframe(self, tmp_path):
        """Test writing with LazyFrame input."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        df = pl.DataFrame(
            {
                "gauge_id": ["G001"],
                "attribute_type": ["caravan"],
                "area": [100.5],
            }
        ).lazy()

        ds.write_static_attributes(df, tmp_path)

        # Verify data was written
        expected_file = (
            tmp_path / "REGION_NAME=test_region" / "data_type=attributes" / "attribute_type=caravan" / "data.parquet"
        )
        assert expected_file.exists()

    def test_write_static_attributes_no_region_error(self, tmp_path):
        """Test that writing without region raises error."""
        ds = CaravanDataSource(tmp_path)  # No region specified

        df = pl.DataFrame(
            {
                "gauge_id": ["G001"],
                "attribute_type": ["caravan"],
                "area": [100.5],
            }
        )

        with pytest.raises(ValueError, match="Region must be set to write attributes"):
            ds.write_static_attributes(df, tmp_path)

    def test_write_static_attributes_missing_gauge_id_error(self, tmp_path):
        """Test that missing gauge_id column raises error."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        df = pl.DataFrame(
            {
                "attribute_type": ["caravan"],
                "area": [100.5],
            }
        )

        with pytest.raises(ValueError, match="DataFrame must contain 'gauge_id' column"):
            ds.write_static_attributes(df, tmp_path)

    def test_write_static_attributes_missing_attribute_type_error(self, tmp_path):
        """Test that missing attribute_type column raises error."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        df = pl.DataFrame(
            {
                "gauge_id": ["G001"],
                "area": [100.5],
            }
        )

        with pytest.raises(ValueError, match="DataFrame must contain 'attribute_type' column"):
            ds.write_static_attributes(df, tmp_path)

    def test_write_static_attributes_overwrite_false_error(self, tmp_path):
        """Test that existing partitions raise error when overwrite=False."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        df = pl.DataFrame(
            {
                "gauge_id": ["G001"],
                "attribute_type": ["caravan"],
                "area": [100.5],
            }
        )

        # First write should succeed
        ds.write_static_attributes(df, tmp_path)

        # Second write should fail with overwrite=False
        with pytest.raises(ValueError, match="Attribute type partitions already exist"):
            ds.write_static_attributes(df, tmp_path, overwrite=False)

    def test_write_static_attributes_overwrite_true_success(self, tmp_path):
        """Test that overwrite=True replaces existing data."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        # First dataset
        df1 = pl.DataFrame(
            {
                "gauge_id": ["G001"],
                "attribute_type": ["caravan"],
                "area": [100.5],
            }
        )
        ds.write_static_attributes(df1, tmp_path)

        # Second dataset with different values
        df2 = pl.DataFrame(
            {
                "gauge_id": ["G001"],
                "attribute_type": ["caravan"],
                "area": [999.9],
            }
        )
        ds.write_static_attributes(df2, tmp_path, overwrite=True)

        # Verify new data was written
        written_file = (
            tmp_path / "REGION_NAME=test_region" / "data_type=attributes" / "attribute_type=caravan" / "data.parquet"
        )
        data = pl.read_parquet(written_file)
        assert data["area"][0] == 999.9

    def test_write_static_attributes_multiple_types(self, tmp_path):
        """Test writing data for multiple attribute types creates separate partitions."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        df = pl.DataFrame(
            {
                "gauge_id": ["G001", "G002", "G001", "G003"],
                "attribute_type": ["caravan", "caravan", "hydroatlas", "topo"],
                "area": [100.5, 250.3, None, None],
                "forest_cover": [None, None, 0.45, None],
                "slope": [None, None, None, 0.05],
            }
        )

        ds.write_static_attributes(df, tmp_path)

        # Check all attribute type partitions exist
        base_path = tmp_path / "REGION_NAME=test_region" / "data_type=attributes"
        assert (base_path / "attribute_type=caravan" / "data.parquet").exists()
        assert (base_path / "attribute_type=hydroatlas" / "data.parquet").exists()
        assert (base_path / "attribute_type=topo" / "data.parquet").exists()

        # Verify data separation
        caravan_data = pl.read_parquet(base_path / "attribute_type=caravan" / "data.parquet")
        hydroatlas_data = pl.read_parquet(base_path / "attribute_type=hydroatlas" / "data.parquet")
        topo_data = pl.read_parquet(base_path / "attribute_type=topo" / "data.parquet")

        assert len(caravan_data) == 2
        assert len(hydroatlas_data) == 1
        assert len(topo_data) == 1

    def test_write_static_attributes_mixed_schemas(self, tmp_path):
        """Test writing data with different schemas for different attribute types."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        df = pl.DataFrame(
            {
                "gauge_id": ["G001", "G001"],
                "attribute_type": ["caravan", "hydroatlas"],
                "area": [100.5, None],  # Only in caravan
                "elevation": [500, None],  # Only in caravan
                "forest_cover": [None, 0.45],  # Only in hydroatlas
                "urban_area": [None, 0.10],  # Only in hydroatlas
            }
        )

        ds.write_static_attributes(df, tmp_path)

        # Read back and verify schemas
        base_path = tmp_path / "REGION_NAME=test_region" / "data_type=attributes"
        caravan_data = pl.read_parquet(base_path / "attribute_type=caravan" / "data.parquet")
        hydroatlas_data = pl.read_parquet(base_path / "attribute_type=hydroatlas" / "data.parquet")

        # Caravan should have area and elevation, but not forest_cover or urban_area
        assert "area" in caravan_data.columns
        assert "elevation" in caravan_data.columns
        assert "forest_cover" in caravan_data.columns  # May be present as null
        assert "urban_area" in caravan_data.columns  # May be present as null

        # Hydroatlas should have forest_cover and urban_area
        assert "forest_cover" in hydroatlas_data.columns
        assert "urban_area" in hydroatlas_data.columns

    def test_write_static_attributes_empty_dataframe(self, tmp_path):
        """Test writing empty DataFrame."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        # Empty DataFrame with correct schema
        df = pl.DataFrame(
            {
                "gauge_id": [],
                "attribute_type": [],
                "area": [],
            }
        ).cast({"gauge_id": pl.Utf8, "attribute_type": pl.Utf8, "area": pl.Float64})

        ds.write_static_attributes(df, tmp_path)

        # Should not crash - directory structure may or may not be created


class TestWriteReadIntegration:
    """Test integration between write and read methods (round-trip testing)."""

    def test_timeseries_write_read_roundtrip(self, tmp_path):
        """Test writing timeseries data and reading it back."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        # Create original data
        original_df = pl.DataFrame(
            {
                "gauge_id": ["G001", "G001", "G002", "G002"],
                "date": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
                "streamflow": [10.5, 12.0, 15.2, 14.8],
                "temperature": [5.0, 6.2, 4.8, 5.5],
                "precipitation": [0.0, 2.5, 1.0, 0.5],
            }
        )

        # Write data
        ds.write_timeseries(original_df, tmp_path)

        # Read data back using the CaravanDataSource
        ds_read = CaravanDataSource(tmp_path, region="test_region")
        read_df = ds_read.get_timeseries().collect()

        # Sort both dataframes for comparison
        original_sorted = original_df.sort(["gauge_id", "date"])
        read_sorted = read_df.sort(["gauge_id", "date"])

        # Compare core data (excluding hive partition columns added during read)
        core_columns = ["gauge_id", "date", "streamflow", "temperature", "precipitation"]
        for col in core_columns:
            assert col in read_sorted.columns
            if col == "date":
                # Convert string dates back to date objects for comparison
                original_dates = original_sorted[col].str.strptime(pl.Date, "%Y-%m-%d")
                assert original_dates.equals(read_sorted[col])
            else:
                assert original_sorted[col].equals(read_sorted[col])

    def test_static_attributes_write_read_roundtrip(self, tmp_path):
        """Test writing static attributes and reading them back."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        # Create original data
        original_df = pl.DataFrame(
            {
                "gauge_id": ["G001", "G002", "G001", "G003"],
                "attribute_type": ["caravan", "caravan", "hydroatlas", "topo"],
                "area": [100.5, 250.3, None, None],
                "elevation": [500, 750, None, None],
                "forest_cover": [None, None, 0.45, None],
                "slope": [None, None, None, 0.05],
            }
        )

        # Write data
        ds.write_static_attributes(original_df, tmp_path)

        # Read data back using the CaravanDataSource
        ds_read = CaravanDataSource(tmp_path, region="test_region")
        read_df = ds_read.get_static_attributes().collect()

        # Sort both dataframes for comparison
        original_sorted = original_df.sort(["gauge_id", "attribute_type"])
        read_sorted = read_df.sort(["gauge_id", "attribute_type"])

        # Compare core data
        assert len(original_sorted) == len(read_sorted)
        assert set(original_sorted["gauge_id"]) == set(read_sorted["gauge_id"])
        assert set(original_sorted["attribute_type"]) == set(read_sorted["attribute_type"])

    def test_write_read_with_filters(self, tmp_path):
        """Test writing data and reading with various filters."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        # Write timeseries data
        ts_df = pl.DataFrame(
            {
                "gauge_id": ["G001", "G001", "G002", "G002", "G003", "G003"],
                "date": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
                "streamflow": [10.5, 12.0, 15.2, 14.8, 8.8, 9.2],
                "temperature": [5.0, 6.2, 4.8, 5.5, 3.2, 4.1],
            }
        )
        ds.write_timeseries(ts_df, tmp_path)

        # Write attributes data
        attr_df = pl.DataFrame(
            {
                "gauge_id": ["G001", "G002", "G003", "G001", "G002", "G003"],
                "attribute_type": ["caravan", "caravan", "caravan", "hydroatlas", "hydroatlas", "hydroatlas"],
                "area": [100.5, 250.3, 175.8, None, None, None],
                "forest_cover": [None, None, None, 0.45, 0.62, 0.38],
            }
        )
        ds.write_static_attributes(attr_df, tmp_path)

        # Test reading with gauge filter
        ds_read = CaravanDataSource(tmp_path, region="test_region")
        filtered_ts = ds_read.get_timeseries(gauge_ids=["G001", "G003"]).collect()
        filtered_attrs = ds_read.get_static_attributes(gauge_ids=["G001", "G003"]).collect()

        assert set(filtered_ts["gauge_id"].unique()) == {"G001", "G003"}
        assert set(filtered_attrs["gauge_id"].unique()) == {"G001", "G003"}
        assert len(filtered_ts) == 4  # 2 gauges * 2 dates
        assert len(filtered_attrs) == 4  # 2 gauges * 2 attribute types

    def test_write_read_different_regions(self, tmp_path):
        """Test writing to different regions and reading them separately."""
        # Write data for region 1
        ds1 = CaravanDataSource(tmp_path, region="region1")
        df1 = pl.DataFrame(
            {
                "gauge_id": ["R1G001", "R1G002"],
                "date": ["2020-01-01", "2020-01-01"],
                "streamflow": [10.5, 12.0],
            }
        )
        ds1.write_timeseries(df1, tmp_path)

        # Write data for region 2
        ds2 = CaravanDataSource(tmp_path, region="region2")
        df2 = pl.DataFrame(
            {
                "gauge_id": ["R2G001", "R2G002"],
                "date": ["2020-01-01", "2020-01-01"],
                "streamflow": [15.5, 18.0],
            }
        )
        ds2.write_timeseries(df2, tmp_path)

        # Read from region 1 only
        read_r1 = ds1.get_timeseries().collect()
        assert set(read_r1["gauge_id"].unique()) == {"R1G001", "R1G002"}
        assert read_r1["REGION_NAME"].unique()[0] == "region1"

        # Read from region 2 only
        read_r2 = ds2.get_timeseries().collect()
        assert set(read_r2["gauge_id"].unique()) == {"R2G001", "R2G002"}
        assert read_r2["REGION_NAME"].unique()[0] == "region2"

        # Read from all regions
        ds_all = CaravanDataSource(tmp_path)
        read_all = ds_all.get_timeseries().collect()
        all_gauges = {"R1G001", "R1G002", "R2G001", "R2G002"}
        assert set(read_all["gauge_id"].unique()) == all_gauges
        assert set(read_all["REGION_NAME"].unique()) == {"region1", "region2"}

    def test_write_read_data_types_preservation(self, tmp_path):
        """Test that data types are preserved during write/read cycle."""
        ds = CaravanDataSource(tmp_path, region="test_region")

        # Create data with specific types
        original_df = pl.DataFrame(
            {
                "gauge_id": ["G001"],
                "date": ["2020-01-01"],
                "streamflow": [10.5],
                "count": [42],
                "flag": [True],
            }
        ).cast(
            {
                "gauge_id": pl.Utf8,
                "date": pl.Utf8,
                "streamflow": pl.Float64,
                "count": pl.Int32,
                "flag": pl.Boolean,
            }
        )

        ds.write_timeseries(original_df, tmp_path)

        # Read back and check types
        ds_read = CaravanDataSource(tmp_path, region="test_region")
        read_df = ds_read.get_timeseries().collect()

        # Note: date gets converted to Date type during reading
        assert read_df["streamflow"].dtype == pl.Float64
        assert read_df["count"].dtype == pl.Int32
        assert read_df["flag"].dtype == pl.Boolean
        assert read_df["date"].dtype == pl.Date

    def test_write_read_large_dataset_simulation(self, tmp_path):
        """Test write/read with a larger simulated dataset."""
        ds = CaravanDataSource(tmp_path, region="large_test")

        # Create larger dataset
        import random

        gauge_ids = [f"G{i:03d}" for i in range(1, 11)]  # 10 gauges
        dates = pd.date_range("2020-01-01", "2020-01-31", freq="D")  # 31 days

        data = []
        for gauge_id in gauge_ids:
            for date_val in dates:
                data.append(
                    {
                        "gauge_id": gauge_id,
                        "date": date_val.strftime("%Y-%m-%d"),
                        "streamflow": random.uniform(5.0, 20.0),
                        "temperature": random.uniform(-5.0, 15.0),
                    }
                )

        large_df = pl.DataFrame(data)

        # Write data
        ds.write_timeseries(large_df, tmp_path)

        # Read back and verify
        ds_read = CaravanDataSource(tmp_path, region="large_test")
        read_df = ds_read.get_timeseries().collect()

        assert len(read_df) == len(large_df)  # 10 gauges * 31 days = 310 rows
        assert set(read_df["gauge_id"].unique()) == set(gauge_ids)
        assert len(read_df["date"].unique()) == 31

        # Test filtered reading
        subset_gauges = ["G001", "G005", "G010"]
        filtered_df = ds_read.get_timeseries(gauge_ids=subset_gauges).collect()
        assert len(filtered_df) == 3 * 31  # 3 gauges * 31 days
        assert set(filtered_df["gauge_id"].unique()) == set(subset_gauges)
