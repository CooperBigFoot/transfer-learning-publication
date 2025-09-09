from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl

from ..containers import StaticAttributeCollection, TimeSeriesCollection


class CaravanDataSource:
    """
    Lazy data access for Caravan datasets using Polars and hive partitioning.

    The data is organized in a hive-partitioned structure that enables efficient
    filtering at the file level without reading data.
    """

    def __init__(self, base_path: str | Path, region: str | list[str] | None = None):
        """
        Initialize CaravanDataSource.

        Args:
            base_path: Root directory containing hive-partitioned data
            region: Optional region specification:
                   - str: Single region name (e.g., 'camels')
                   - list[str]: Multiple region names (e.g., ['camels', 'hysets'])
                   - None: All regions are accessible
        """
        self.base_path = Path(base_path)
        self.region = region

        # Build glob patterns for reuse
        if region is None:
            # Access all regions
            region_pattern = "REGION_NAME=*"
            self._ts_glob = str(self.base_path / region_pattern / "data_type=timeseries" / "gauge_id=*" / "data.parquet")
            self._attr_glob = str(self.base_path / region_pattern / "data_type=attributes" / "data.parquet")
            self._shapefile_patterns = [self.base_path / region_pattern / "data_type=shapefiles"]
        elif isinstance(region, str):
            # Single region
            region_pattern = f"REGION_NAME={region}"
            self._ts_glob = str(self.base_path / region_pattern / "data_type=timeseries" / "gauge_id=*" / "data.parquet")
            self._attr_glob = str(self.base_path / region_pattern / "data_type=attributes" / "data.parquet")
            self._shapefile_patterns = [self.base_path / region_pattern / "data_type=shapefiles"]
        elif isinstance(region, list):
            # Multiple regions - create list of glob patterns
            if not region:
                # Empty list - no data accessible
                self._ts_glob = []
                self._attr_glob = []
                self._shapefile_patterns = []
            else:
                # Create glob patterns for each region
                self._ts_glob = [
                    str(self.base_path / f"REGION_NAME={r}" / "data_type=timeseries" / "gauge_id=*" / "data.parquet")
                    for r in region
                ]
                self._attr_glob = [
                    str(self.base_path / f"REGION_NAME={r}" / "data_type=attributes" / "data.parquet")
                    for r in region
                ]
                self._shapefile_patterns = [
                    self.base_path / f"REGION_NAME={r}" / "data_type=shapefiles"
                    for r in region
                ]
        else:
            raise TypeError(f"region must be str, list[str], or None, got {type(region)}")

    def list_regions(self) -> list[str]:
        """
        List available regions based on the region filter.

        Returns:
            List of region names (filtered if region was specified during init)
        """
        if isinstance(self.region, str):
            # Single region specified - return it if it exists
            region_path = self.base_path / f"REGION_NAME={self.region}"
            return [self.region] if region_path.exists() else []
        elif isinstance(self.region, list):
            # Multiple regions specified - return those that exist
            existing_regions = []
            for r in self.region:
                region_path = self.base_path / f"REGION_NAME={r}"
                if region_path.exists():
                    existing_regions.append(r)
            return sorted(existing_regions)
        else:
            # No filter - list all available regions
            region_dirs = self.base_path.glob("REGION_NAME=*")
            regions = [d.name.split("=")[1] for d in region_dirs if d.is_dir()]
            return sorted(regions)

    def list_gauge_ids(self) -> list[str]:
        """
        List all available gauge IDs using lazy schema inspection.

        Returns:
            List of unique gauge IDs
        """
        if isinstance(self._ts_glob, list) and not self._ts_glob:
            # Empty list - no data
            return []

        try:
            # Use union_by_name to handle schema differences
            lf = pl.scan_parquet(self._ts_glob, hive_partitioning=True, rechunk=False, low_memory=True)
            # Use lazy execution to get unique gauge_ids and ensure they're strings
            gauge_ids = lf.select("gauge_id").unique().collect()["gauge_id"]
            # Convert to strings in case hive partitioning parsed them as integers
            gauge_ids = [str(gid) for gid in gauge_ids.to_list()]
            return sorted(gauge_ids)
        except Exception:
            # Fallback to filesystem if scan fails
            if isinstance(self._ts_glob, list):
                # Multiple patterns - gather from all
                all_gauge_ids = set()
                for glob_pattern in self._ts_glob:
                    gauge_dirs = Path(glob_pattern).parent.glob("gauge_id=*")
                    all_gauge_ids.update(d.name.split("=")[1] for d in gauge_dirs if d.is_dir())
                return sorted(all_gauge_ids)
            else:
                gauge_dirs = Path(self._ts_glob).parent.glob("gauge_id=*")
                gauge_ids = [d.name.split("=")[1] for d in gauge_dirs if d.is_dir()]
                return sorted(gauge_ids)

    def list_timeseries_variables(self) -> list[str]:
        """
        List all available timeseries variables using schema inspection.

        Returns:
            List of variable names (excluding metadata columns)
        """
        if isinstance(self._ts_glob, list) and not self._ts_glob:
            # Empty list - no data
            return []

        try:
            lf = pl.scan_parquet(self._ts_glob, hive_partitioning=True, n_rows=0, rechunk=False, low_memory=True)
            schema = lf.collect_schema()

            # Exclude partition columns and date column
            exclude_cols = {"REGION_NAME", "gauge_id", "data_type", "date"}
            variables = [col for col in schema if col not in exclude_cols]
            return sorted(variables)
        except Exception:
            # Return empty list if no files found
            return []

    def list_static_attributes(self) -> list[str]:
        """
        List all available static attributes using schema inspection.

        Returns:
            List of attribute column names
        """
        if isinstance(self._attr_glob, list) and not self._attr_glob:
            # Empty list - no data
            return []

        try:
            lf = pl.scan_parquet(self._attr_glob, hive_partitioning=True, n_rows=0, rechunk=False, low_memory=True)
            schema = lf.collect_schema()

            # Exclude partition columns and gauge_id
            exclude_cols = {"REGION_NAME", "data_type", "gauge_id"}
            attributes = [col for col in schema if col not in exclude_cols]
            return sorted(attributes)
        except Exception:
            # Return empty list if no files found
            return []

    def get_date_ranges(self, gauge_ids: list[str] | None = None) -> pl.LazyFrame:
        """
        Get date ranges for gauges as a LazyFrame.

        Args:
            gauge_ids: Optional list of gauge IDs to filter

        Returns:
            LazyFrame with columns: REGION_NAME, gauge_id, min_date, max_date
        """
        if isinstance(self._ts_glob, list) and not self._ts_glob:
            # Empty list - return empty LazyFrame
            return pl.DataFrame({
                "REGION_NAME": pl.Series([], dtype=pl.Utf8),
                "gauge_id": pl.Series([], dtype=pl.Utf8),
                "min_date": pl.Series([], dtype=pl.Date),
                "max_date": pl.Series([], dtype=pl.Date),
            }).lazy()

        lf = pl.scan_parquet(self._ts_glob, hive_partitioning=True, rechunk=False, low_memory=True)

        # Handle date dtype normalization if stored as string
        schema = lf.collect_schema()
        if schema["date"] == pl.Utf8:
            lf = lf.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        elif schema["date"] == pl.Datetime:
            lf = lf.with_columns(pl.col("date").cast(pl.Date))

        # Apply gauge filter if specified
        if gauge_ids:
            lf = lf.filter(pl.col("gauge_id").is_in(gauge_ids))

        # Group by region and gauge to get date ranges
        return lf.group_by(["REGION_NAME", "gauge_id"]).agg(
            pl.col("date").min().alias("min_date"), pl.col("date").max().alias("max_date")
        )

    def get_timeseries(
        self,
        gauge_ids: list[str] | None = None,
        columns: list[str] | None = None,
        date_range: tuple[str, str] | None = None,
    ) -> pl.LazyFrame:
        """
        Get timeseries data as a LazyFrame.

        Args:
            gauge_ids: Optional list of gauge IDs to filter (uses partition pruning)
            columns: Optional list of columns to select
            date_range: Optional tuple of (start_date, end_date) as strings

        Returns:
            LazyFrame with timeseries data
        """
        import warnings

        if isinstance(self._ts_glob, list) and not self._ts_glob:
            # Empty list - return empty LazyFrame
            empty_df = pl.DataFrame({
                "REGION_NAME": pl.Series([], dtype=pl.Utf8),
                "gauge_id": pl.Series([], dtype=pl.Utf8),
                "date": pl.Series([], dtype=pl.Date),
            })
            if columns:
                for col in columns:
                    empty_df = empty_df.with_columns(pl.lit(None).alias(col))
            return empty_df.lazy()

        lf = pl.scan_parquet(self._ts_glob, hive_partitioning=True, rechunk=False, low_memory=True)

        # Handle date dtype normalization if stored as string
        schema = lf.collect_schema()
        if schema["date"] == pl.Utf8:
            lf = lf.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        elif schema["date"] == pl.Datetime:
            lf = lf.with_columns(pl.col("date").cast(pl.Date))

        # Apply filters - Polars optimizes partition pruning automatically
        if gauge_ids:
            lf = lf.filter(pl.col("gauge_id").is_in(gauge_ids))

        if date_range:
            start_date, end_date = date_range
            # Convert string dates to Date type for comparison
            lf = lf.filter(
                pl.col("date").is_between(
                    pl.lit(start_date).str.strptime(pl.Date, "%Y-%m-%d"),
                    pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d"),
                )
            )

        if columns:
            # Keep metadata columns plus requested columns
            keep_cols = {"REGION_NAME", "gauge_id", "date"} | set(columns)
            available_cols = set(lf.collect_schema().names())

            # Check for missing columns and warn
            requested_cols = set(columns)
            missing_cols = requested_cols - available_cols
            if missing_cols:
                warnings.warn(
                    f"Requested timeseries columns not found in data: {sorted(missing_cols)}. "
                    f"Available columns: {sorted(available_cols - {'REGION_NAME', 'gauge_id', 'date', 'data_type'})}",
                    UserWarning,
                    stacklevel=2
                )

            cols_to_select = list(keep_cols & available_cols)
            lf = lf.select(cols_to_select)

        return lf

    def get_static_attributes(
        self,
        gauge_ids: list[str] | None = None,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """
        Get static attributes as a LazyFrame.

        Args:
            gauge_ids: Optional list of gauge IDs to filter
            columns: Optional list of attribute columns to select

        Returns:
            LazyFrame with static attributes
        """
        import warnings

        if isinstance(self._attr_glob, list) and not self._attr_glob:
            # Empty list - return empty LazyFrame
            empty_df = pl.DataFrame(
                {
                    "REGION_NAME": pl.Series([], dtype=pl.Utf8),
                    "gauge_id": pl.Series([], dtype=pl.Utf8),
                }
            )
            if columns:
                for col in columns:
                    empty_df = empty_df.with_columns(pl.lit(None).alias(col))
            return empty_df.lazy()

        # Check if files exist first
        from glob import glob

        if isinstance(self._attr_glob, list):
            files = []
            for pattern in self._attr_glob:
                files.extend(glob(pattern))
        else:
            files = glob(self._attr_glob)

        if not files:
            # Return empty LazyFrame with expected schema when no files found
            empty_df = pl.DataFrame(
                {
                    "REGION_NAME": pl.Series([], dtype=pl.Utf8),
                    "gauge_id": pl.Series([], dtype=pl.Utf8),
                }
            )
            if columns:
                for col in columns:
                    empty_df = empty_df.with_columns(pl.lit(None).alias(col))
            return empty_df.lazy()

        lf = pl.scan_parquet(self._attr_glob, hive_partitioning=True, rechunk=False, low_memory=True)

        # Apply gauge filter if needed
        if gauge_ids:
            lf = lf.filter(pl.col("gauge_id").is_in(gauge_ids))

        # Select columns if specified
        if columns:
            # Keep metadata columns plus requested attributes
            keep_cols = {"REGION_NAME", "gauge_id"} | set(columns)
            available_cols = set(lf.collect_schema().names())

            # Check for missing columns and warn
            requested_cols = set(columns)
            missing_cols = requested_cols - available_cols
            if missing_cols:
                warnings.warn(
                    f"Requested attribute columns not found in data: {sorted(missing_cols)}. "
                    f"Available columns: {sorted(available_cols - {'REGION_NAME', 'gauge_id', 'data_type'})}",
                    UserWarning,
                    stacklevel=2
                )

            cols_to_select = list(keep_cols & available_cols)
            lf = lf.select(cols_to_select)

        return lf

    def get_geometries(self, gauge_ids: list[str] | None = None) -> gpd.GeoDataFrame:
        """
        Get watershed geometries as a GeoDataFrame.

        Args:
            gauge_ids: Optional list of gauge IDs to filter

        Returns:
            GeoDataFrame with watershed geometries
        """
        gdfs = []

        if isinstance(self.region, str):
            # Single region
            shapefile_path = self._shapefile_patterns[0] / f"{self.region}_shapes.shp"
            if not shapefile_path.exists():
                raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

            gdf = gpd.read_file(shapefile_path)
            gdf["REGION_NAME"] = self.region  # Add region column for consistency
            gdfs.append(gdf)

        elif isinstance(self.region, list):
            # Multiple regions specified
            if not self.region:
                # Empty list
                raise FileNotFoundError("No regions specified")

            for region_name in self.region:
                shapefile_path = self.base_path / f"REGION_NAME={region_name}" / "data_type=shapefiles" / f"{region_name}_shapes.shp"
                if shapefile_path.exists():
                    gdf = gpd.read_file(shapefile_path)
                    gdf["REGION_NAME"] = region_name
                    gdfs.append(gdf)

        else:
            # No region specified - load all available
            for pattern in self._shapefile_patterns:
                for region_dir in pattern.parent.glob("REGION_NAME=*"):
                    region_name = region_dir.name.split("=")[1]
                    shapefile_path = region_dir / "data_type=shapefiles" / f"{region_name}_shapes.shp"

                    if shapefile_path.exists():
                        gdf = gpd.read_file(shapefile_path)
                        gdf["REGION_NAME"] = region_name
                        gdfs.append(gdf)

        if not gdfs:
            raise FileNotFoundError("No shapefiles found")

        # Combine all GeoDataFrames
        combined_gdf = pd.concat(gdfs, ignore_index=True) if len(gdfs) > 1 else gdfs[0]

        # Apply gauge filter if specified
        if gauge_ids:
            combined_gdf = combined_gdf[combined_gdf["gauge_id"].isin(gauge_ids)]

        return combined_gdf

    def write_timeseries(
        self, df: pl.DataFrame | pl.LazyFrame, output_base_path: str | Path, overwrite: bool = False
    ) -> None:
        """
        Write timeseries data to hive-partitioned parquet files.

        Creates structure: REGION_NAME={region}/data_type=timeseries/gauge_id={id}/data.parquet

        Args:
            df: DataFrame or LazyFrame with timeseries data. Must contain 'gauge_id' column.
            output_base_path: Root directory for output
            overwrite: If False, raise error if gauge partitions exist. If True, overwrite.

        Raises:
            ValueError: If region not set, gauge_id column missing, or existing data when overwrite=False
        """
        import logging

        logger = logging.getLogger(__name__)

        # Validate region is set and is a single string
        if self.region is None:
            raise ValueError("Region must be set to write timeseries. Initialize with a specific region.")
        if isinstance(self.region, list):
            raise ValueError("Cannot write timeseries with multiple regions. Initialize with a single region string.")

        # Convert LazyFrame to DataFrame if needed
        if isinstance(df, pl.LazyFrame):
            logger.warning(
                "LazyFrame passed to write_timeseries - collecting to DataFrame. This may use significant memory."
            )
            df = df.collect()

        # Validate required columns
        if "gauge_id" not in df.columns:
            raise ValueError("DataFrame must contain 'gauge_id' column for partitioning")

        # Warn if date column missing (expected but not required)
        if "date" not in df.columns:
            logger.warning("No 'date' column found in timeseries data")

        # Build output path
        output_path = Path(output_base_path) / f"REGION_NAME={self.region}" / "data_type=timeseries"

        # Get unique gauge IDs
        unique_gauges = df["gauge_id"].unique().to_list()
        existing_gauges = []

        # Check for existing partitions
        if output_path.exists():
            for gauge_id in unique_gauges:
                gauge_path = output_path / f"gauge_id={gauge_id}"
                if gauge_path.exists():
                    existing_gauges.append(str(gauge_id))

            if existing_gauges:
                if not overwrite:
                    n_more = len(existing_gauges) - 5
                    raise ValueError(
                        f"Gauge partitions already exist: {', '.join(existing_gauges[:5])}"
                        f"{f' and {n_more} more' if n_more > 0 else ''}"
                        "\nSet overwrite=True to replace existing data."
                    )
                else:
                    logger.warning(f"Overwriting {len(existing_gauges)} existing gauge partition(s)")

        # Write each gauge_id to its own partition with data.parquet filename
        # Group by gauge_id and write each group separately
        for gauge_id in unique_gauges:
            gauge_df = df.filter(pl.col("gauge_id") == gauge_id)
            gauge_path = output_path / f"gauge_id={gauge_id}"
            gauge_path.mkdir(parents=True, exist_ok=True)

            # Remove the gauge_id column since it's in the partition path
            gauge_df = gauge_df.drop("gauge_id")
            gauge_df.write_parquet(gauge_path / "data.parquet", use_pyarrow=True, statistics=True)

        logger.info(f"Wrote timeseries for {len(unique_gauges)} gauges to {output_path}")

    def write_static_attributes(
        self, df: pl.DataFrame | pl.LazyFrame, output_base_path: str | Path, overwrite: bool = False
    ) -> None:
        """
        Write static attributes to hive-partitioned parquet file.

        Creates structure: REGION_NAME={region}/data_type=attributes/data.parquet

        Args:
            df: DataFrame or LazyFrame with attributes. Must contain 'gauge_id' column.
            output_base_path: Root directory for output
            overwrite: If False, raise error if file exists. If True, overwrite.

        Raises:
            ValueError: If region not set, gauge_id column missing, or existing data when overwrite=False
        """
        import logging

        logger = logging.getLogger(__name__)

        # Validate region is set and is a single string
        if self.region is None:
            raise ValueError("Region must be set to write attributes. Initialize with a specific region.")
        if isinstance(self.region, list):
            raise ValueError("Cannot write attributes with multiple regions. Initialize with a single region string.")

        # Convert LazyFrame to DataFrame if needed
        if isinstance(df, pl.LazyFrame):
            logger.warning(
                "LazyFrame passed to write_static_attributes - collecting to DataFrame. This may use significant memory."
            )
            df = df.collect()

        # Validate required columns
        if "gauge_id" not in df.columns:
            raise ValueError("DataFrame must contain 'gauge_id' column")

        # Build output path
        output_path = Path(output_base_path) / f"REGION_NAME={self.region}" / "data_type=attributes"
        output_file = output_path / "data.parquet"

        # Check for existing file
        if output_file.exists():
            if not overwrite:
                raise ValueError(
                    f"Attributes file already exists: {output_file}\nSet overwrite=True to replace existing data."
                )
            else:
                logger.warning(f"Overwriting existing attributes file: {output_file}")

        # Create directory if needed
        output_path.mkdir(parents=True, exist_ok=True)

        # Write the dataframe directly
        df.write_parquet(output_file, use_pyarrow=True, statistics=True)

        n_gauges = df["gauge_id"].n_unique()
        n_attributes = len([col for col in df.columns if col != "gauge_id"])
        logger.info(f"Wrote {n_gauges} gauges with {n_attributes} attributes to {output_file}")

    def to_time_series_collection(self, lf: pl.LazyFrame) -> TimeSeriesCollection:
        """
        Convert a LazyFrame to TimeSeriesCollection.

        Expects LazyFrame from get_timeseries() with columns:
        - 'gauge_id': Group identifier (string)
        - 'date': Date column
        - feature columns: All other numeric columns
        - 'REGION_NAME': Optional metadata (ignored)

        Args:
            lf: LazyFrame from get_timeseries()

        Returns:
            TimeSeriesCollection with data loaded into memory

        Raises:
            ValueError: If data validation fails (indicates upstream processing error)
        """
        df = lf.collect()

        if df.is_empty():
            # Return empty collection
            return TimeSeriesCollection(tensors=[], feature_names=[], date_ranges=[], group_identifiers=[])

        # Validate required columns
        required_columns = {"gauge_id", "date"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Auto-detect feature columns (exclude metadata)
        metadata_columns = {"gauge_id", "date", "REGION_NAME", "data_type"}
        feature_columns = [col for col in df.columns if col not in metadata_columns]

        if not feature_columns:
            raise ValueError("No feature columns found after excluding metadata")

        # Validate data types of feature columns
        for col in feature_columns:
            dtype = df[col].dtype
            if dtype not in [
                pl.Float32,
                pl.Float64,
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
            ]:
                raise ValueError(
                    f"Feature column '{col}' has non-numeric dtype: {dtype}. "
                    "All feature columns must be numeric. This indicates an upstream processing error."
                )

        # Process each gauge
        tensors = []
        date_ranges = []

        # Get unique gauge_ids and sort them for consistent ordering
        gauge_ids = sorted(df["gauge_id"].unique().to_list())

        for gauge_id in gauge_ids:
            # Filter data for this gauge
            gauge_df = df.filter(pl.col("gauge_id") == gauge_id)

            # Sort by date to ensure chronological order
            gauge_df = gauge_df.sort("date")

            # Validate no duplicate dates
            dates = gauge_df["date"]
            if dates.n_unique() != len(dates):
                raise ValueError(
                    f"Gauge '{gauge_id}' has duplicate dates. This indicates an upstream processing error."
                )

            # Extract feature data and convert to tensor
            feature_data = gauge_df.select(feature_columns)

            # Check for nulls before conversion
            if feature_data.null_count().sum_horizontal()[0] > 0:
                null_cols = [col for col in feature_columns if gauge_df[col].null_count() > 0]
                raise ValueError(
                    f"Gauge '{gauge_id}' has null values in columns: {null_cols}. "
                    "This indicates an upstream processing error."
                )

            # Convert to torch tensor using Polars' to_torch method
            tensor = feature_data.to_torch(dtype=pl.Float32)
            tensors.append(tensor)

            # Extract date range
            date_series = gauge_df["date"]
            min_date = date_series.min()
            max_date = date_series.max()

            # Convert to datetime if needed (Polars dates to Python datetime)
            if isinstance(min_date, pl.datatypes.Date):
                min_date = min_date.to_py()
            if isinstance(max_date, pl.datatypes.Date):
                max_date = max_date.to_py()

            from datetime import date, datetime

            if isinstance(min_date, date) and not isinstance(min_date, datetime):
                min_date = datetime.combine(min_date, datetime.min.time())
            if isinstance(max_date, date) and not isinstance(max_date, datetime):
                max_date = datetime.combine(max_date, datetime.min.time())

            date_ranges.append((min_date, max_date))

        # Validate all gauges have the same features in the same order
        if len(gauge_ids) > 1:
            first_shape = tensors[0].shape[1]

            for idx in range(1, len(gauge_ids)):
                if tensors[idx].shape[1] != first_shape:
                    raise ValueError(
                        f"Gauge '{gauge_ids[idx]}' has {tensors[idx].shape[1]} features, "
                        f"but gauge '{gauge_ids[0]}' has {first_shape} features. "
                        "All gauges must have the same features. "
                        "This indicates an upstream processing error."
                    )

        return TimeSeriesCollection(
            tensors=tensors,
            feature_names=feature_columns,
            date_ranges=date_ranges,
            group_identifiers=gauge_ids,
            validate=True,  # Always validate to catch any remaining issues
        )

    def to_static_attribute_collection(self, lf: pl.LazyFrame) -> StaticAttributeCollection:
        """
        Convert a LazyFrame to StaticAttributeCollection.

        Expects LazyFrame from get_static_attributes() with columns:
        - 'gauge_id': Group identifier (string)
        - attribute columns: All numeric attribute columns
        - 'REGION_NAME', 'data_type': Optional metadata (ignored)

        Args:
            lf: LazyFrame from get_static_attributes()

        Returns:
            StaticAttributeCollection with data loaded into memory

        Raises:
            ValueError: If data validation fails (indicates upstream processing error)
        """
        df = lf.collect()

        if df.is_empty():
            # Return empty collection
            return StaticAttributeCollection(tensors=[], attribute_names=[], group_identifiers=[])

        # Validate required columns
        if "gauge_id" not in df.columns:
            raise ValueError("Missing required column: gauge_id")

        # Auto-detect attribute columns (exclude metadata)
        metadata_columns = {"gauge_id", "REGION_NAME", "data_type"}
        attribute_columns = [col for col in df.columns if col not in metadata_columns]

        if not attribute_columns:
            raise ValueError("No attribute columns found after excluding metadata")

        # Validate data types of attribute columns
        for col in attribute_columns:
            dtype = df[col].dtype
            if dtype not in [
                pl.Float32,
                pl.Float64,
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
            ]:
                raise ValueError(
                    f"Attribute column '{col}' has non-numeric dtype: {dtype}. "
                    "All attribute columns must be numeric. This indicates an upstream processing error."
                )

        # Process each gauge
        tensors = []

        # Get unique gauge_ids and sort them for consistent ordering
        gauge_ids = sorted(df["gauge_id"].unique().to_list())

        for gauge_id in gauge_ids:
            # Filter data for this gauge
            gauge_df = df.filter(pl.col("gauge_id") == gauge_id)

            # Should be exactly one row per gauge since we have merged attributes
            if len(gauge_df) != 1:
                raise ValueError(
                    f"Gauge '{gauge_id}' has {len(gauge_df)} rows, expected 1. "
                    "This indicates an upstream processing error."
                )

            # Extract attribute data
            attribute_data = gauge_df.select(attribute_columns)

            # Check for nulls before conversion
            if attribute_data.null_count().sum_horizontal()[0] > 0:
                null_cols = [col for col in attribute_columns if gauge_df[col].null_count() > 0]
                raise ValueError(
                    f"Gauge '{gauge_id}' has null values in columns: {null_cols}. "
                    "This indicates an upstream processing error."
                )

            # Convert to torch tensor - should be 1D
            tensor = attribute_data.to_torch(dtype=pl.Float32)

            # Ensure it's 1D (should be since we have 1 row)
            if tensor.dim() != 2 or tensor.shape[0] != 1:
                raise ValueError(f"Expected tensor shape (1, n_attributes) for gauge '{gauge_id}', got {tensor.shape}")

            # Squeeze to get 1D tensor
            tensor = tensor.squeeze(0)
            tensors.append(tensor)

        # Validate all gauges have the same attributes in the same order
        if len(gauge_ids) > 1:
            first_shape = tensors[0].shape[0]

            for idx in range(1, len(gauge_ids)):
                if tensors[idx].shape[0] != first_shape:
                    raise ValueError(
                        f"Gauge '{gauge_ids[idx]}' has {tensors[idx].shape[0]} attributes, "
                        f"but gauge '{gauge_ids[0]}' has {first_shape} attributes. "
                        "All gauges must have the same attributes. "
                        "This indicates an upstream processing error."
                    )

        return StaticAttributeCollection(
            tensors=tensors,
            attribute_names=attribute_columns,
            group_identifiers=gauge_ids,
            validate=True,  # Always validate to catch any remaining issues
        )
