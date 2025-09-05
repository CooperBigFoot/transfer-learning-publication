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

    def __init__(self, base_path: str | Path, region: str | None = None):
        """
        Initialize CaravanDataSource.

        Args:
            base_path: Root directory containing hive-partitioned data
            region: Optional specific region name (e.g., 'camels').
                   If None, all regions are accessible.
        """
        self.base_path = Path(base_path)
        self.region = region

        # Build glob patterns for reuse
        region_pattern = f"REGION_NAME={region}" if region else "REGION_NAME=*"

        self._ts_glob = str(self.base_path / region_pattern / "data_type=timeseries" / "gauge_id=*" / "data.parquet")
        self._attr_glob = str(
            self.base_path / region_pattern / "data_type=attributes" / "attribute_type=*" / "data.parquet"
        )

        # For shapefiles, we need a slightly different approach
        self._shapefile_pattern = self.base_path / region_pattern / "data_type=shapefiles"

    def list_regions(self) -> list[str]:
        """
        List all available regions using filesystem glob.

        Returns:
            List of region names
        """
        region_dirs = self.base_path.glob("REGION_NAME=*")
        regions = [d.name.split("=")[1] for d in region_dirs if d.is_dir()]
        return sorted(regions)

    def list_gauge_ids(self) -> list[str]:
        """
        List all available gauge IDs using lazy schema inspection.

        Returns:
            List of unique gauge IDs
        """
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
            gauge_dirs = Path(self._ts_glob).parent.glob("gauge_id=*")
            gauge_ids = [d.name.split("=")[1] for d in gauge_dirs if d.is_dir()]
            return sorted(gauge_ids)

    def list_timeseries_variables(self) -> list[str]:
        """
        List all available timeseries variables using schema inspection.

        Returns:
            List of variable names (excluding metadata columns)
        """
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

    def list_static_attributes(self, attribute_types: list[str] | None = None) -> list[str]:
        """
        List all available static attributes using schema inspection.

        Args:
            attribute_types: Optional list of attribute types to include
                           (e.g., ['caravan', 'hydroatlas'])

        Returns:
            List of attribute column names
        """
        # First get all parquet files to union their schemas
        from glob import glob

        files = glob(self._attr_glob)

        if not files:
            return []

        # Collect all unique columns from all files
        all_columns = set()
        for file in files:
            try:
                lf = pl.scan_parquet(file, hive_partitioning=True, n_rows=0)
                schema = lf.collect_schema()
                all_columns.update(schema.names())
            except Exception:
                continue

        # Exclude partition columns and gauge_id
        exclude_cols = {"REGION_NAME", "attribute_type", "data_type", "gauge_id"}
        attributes = [col for col in all_columns if col not in exclude_cols]
        return sorted(attributes)

    def get_date_ranges(self, gauge_ids: list[str] | None = None) -> pl.LazyFrame:
        """
        Get date ranges for gauges as a LazyFrame.

        Args:
            gauge_ids: Optional list of gauge IDs to filter

        Returns:
            LazyFrame with columns: REGION_NAME, gauge_id, min_date, max_date
        """
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
        variables: list[str] | None = None,
        date_range: tuple[str, str] | None = None,
    ) -> pl.LazyFrame:
        """
        Get timeseries data as a LazyFrame.

        Args:
            gauge_ids: Optional list of gauge IDs to filter (uses partition pruning)
            variables: Optional list of variables to select
            date_range: Optional tuple of (start_date, end_date) as strings

        Returns:
            LazyFrame with timeseries data
        """
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

        if variables:
            # Keep metadata columns plus requested variables
            keep_cols = {"REGION_NAME", "gauge_id", "date"} | set(variables)
            available_cols = set(lf.collect_schema().names())
            cols_to_select = list(keep_cols & available_cols)
            lf = lf.select(cols_to_select)

        return lf

    def get_static_attributes(
        self,
        gauge_ids: list[str] | None = None,
        columns: list[str] | None = None,
        attribute_types: list[str] | None = None,
    ) -> pl.LazyFrame:
        """
        Get static attributes as a LazyFrame.

        Args:
            gauge_ids: Optional list of gauge IDs to filter
            columns: Optional list of attribute columns to select
            attribute_types: Optional list of attribute types (e.g., ['caravan', 'hydroatlas'])

        Returns:
            LazyFrame with static attributes (long format by default)
        """
        from glob import glob

        # Get list of files that match the pattern
        files = glob(self._attr_glob)

        if not files:
            # Return empty LazyFrame with expected schema when no files found
            empty_df = pl.DataFrame(
                {
                    "REGION_NAME": pl.Series([], dtype=pl.Utf8),
                    "gauge_id": pl.Series([], dtype=pl.Utf8),
                    "attribute_type": pl.Series([], dtype=pl.Utf8),
                }
            )
            if columns:
                for col in columns:
                    empty_df = empty_df.with_columns(pl.lit(None).alias(col))
            return empty_df.lazy()

        # Read each file separately and union them
        # This approach handles schema differences better
        lazy_frames = []

        for file in files:
            try:
                # Scan with hive partitioning to get partition columns
                lf = pl.scan_parquet(file, hive_partitioning=True, rechunk=False, low_memory=True)

                # Force early validation by checking schema - this will fail for corrupted files
                _ = lf.collect_schema()

                # Apply attribute_type filter if needed
                if attribute_types and "attribute_type" in lf.collect_schema().names():
                    lf = lf.filter(pl.col("attribute_type").is_in(attribute_types))

                # Apply gauge filter if needed
                if gauge_ids:
                    lf = lf.filter(pl.col("gauge_id").is_in(gauge_ids))

                # Select columns if specified
                if columns:
                    # Keep metadata columns plus requested attributes
                    keep_cols = {"REGION_NAME", "gauge_id", "attribute_type"} | set(columns)
                    available_cols = set(lf.collect_schema().names())
                    cols_to_select = list(keep_cols & available_cols)

                    # Only add if we have columns beyond just metadata
                    if len(cols_to_select) > 3 or not columns:
                        lf = lf.select(cols_to_select)
                        lazy_frames.append(lf)
                else:
                    lazy_frames.append(lf)

            except Exception as e:
                # Skip files that can't be read
                print(f"Warning: Could not read {file}: {e}")
                continue

        if not lazy_frames:
            # Return empty LazyFrame with expected schema if no valid files were read
            empty_df = pl.DataFrame(
                {
                    "REGION_NAME": pl.Series([], dtype=pl.Utf8),
                    "gauge_id": pl.Series([], dtype=pl.Utf8),
                    "attribute_type": pl.Series([], dtype=pl.Utf8),
                }
            )
            if columns:
                for col in columns:
                    empty_df = empty_df.with_columns(pl.lit(None).alias(col))
            return empty_df.lazy()

        # Union all frames, handling different schemas
        result = lazy_frames[0]
        for lf in lazy_frames[1:]:
            # Use diagonal concatenation to handle different schemas
            result = pl.concat([result, lf], how="diagonal_relaxed")

        return result

    def get_geometries(self, gauge_ids: list[str] | None = None) -> gpd.GeoDataFrame:
        """
        Get watershed geometries as a GeoDataFrame.

        Args:
            gauge_ids: Optional list of gauge IDs to filter

        Returns:
            GeoDataFrame with watershed geometries
        """
        # For shapefiles, we need to handle differently since they're not partitioned
        if self.region:
            shapefile_path = self._shapefile_pattern / f"{self.region}_shapes.shp"
            if not shapefile_path.exists():
                raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

            gdf = gpd.read_file(shapefile_path)

            if gauge_ids:
                # Assuming shapefile has a gauge_id column
                gdf = gdf[gdf["gauge_id"].isin(gauge_ids)]

            return gdf
        else:
            # If no region specified, need to load from all regions
            gdfs = []
            for region_dir in self._shapefile_pattern.parent.glob("REGION_NAME=*"):
                region_name = region_dir.name.split("=")[1]
                shapefile_path = region_dir / "data_type=shapefiles" / f"{region_name}_shapes.shp"

                if shapefile_path.exists():
                    gdf = gpd.read_file(shapefile_path)
                    gdf["REGION_NAME"] = region_name  # Add region column
                    gdfs.append(gdf)

            if not gdfs:
                raise FileNotFoundError("No shapefiles found")

            combined_gdf = pd.concat(gdfs, ignore_index=True)

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

        # Validate region is set
        if self.region is None:
            raise ValueError("Region must be set to write timeseries. Initialize with a specific region.")

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
        Write static attributes to hive-partitioned parquet files.

        Creates structure: REGION_NAME={region}/data_type=attributes/attribute_type={type}/data.parquet

        Args:
            df: DataFrame or LazyFrame with attributes. Must contain 'gauge_id' and 'attribute_type' columns.
            output_base_path: Root directory for output
            overwrite: If False, raise error if attribute type partitions exist. If True, overwrite.

        Raises:
            ValueError: If region not set, required columns missing, or existing data when overwrite=False
        """
        import logging

        logger = logging.getLogger(__name__)

        # Validate region is set
        if self.region is None:
            raise ValueError("Region must be set to write attributes. Initialize with a specific region.")

        # Convert LazyFrame to DataFrame if needed
        if isinstance(df, pl.LazyFrame):
            logger.warning(
                "LazyFrame passed to write_static_attributes - collecting to DataFrame. This may use significant memory."
            )
            df = df.collect()

        # Validate required columns
        if "gauge_id" not in df.columns:
            raise ValueError("DataFrame must contain 'gauge_id' column")
        if "attribute_type" not in df.columns:
            raise ValueError("DataFrame must contain 'attribute_type' column for partitioning")

        # Build output path
        output_path = Path(output_base_path) / f"REGION_NAME={self.region}" / "data_type=attributes"

        # Get unique attribute types
        unique_types = df["attribute_type"].unique().to_list()
        existing_types = []

        # Check for existing attribute type partitions
        if output_path.exists():
            for attr_type in unique_types:
                type_path = output_path / f"attribute_type={attr_type}"
                if type_path.exists():
                    existing_types.append(str(attr_type))

            if existing_types:
                if not overwrite:
                    raise ValueError(
                        f"Attribute type partitions already exist: {', '.join(existing_types)}"
                        "\nSet overwrite=True to replace existing data."
                    )
                else:
                    logger.warning(f"Overwriting existing attribute type(s): {', '.join(existing_types)}")

        # Write each attribute_type to its own partition with data.parquet filename
        # Group by attribute_type and write each group separately
        for attr_type in unique_types:
            type_df = df.filter(pl.col("attribute_type") == attr_type)
            type_path = output_path / f"attribute_type={attr_type}"
            type_path.mkdir(parents=True, exist_ok=True)

            # Remove the attribute_type column since it's in the partition path
            type_df = type_df.drop("attribute_type")
            type_df.write_parquet(type_path / "data.parquet", use_pyarrow=True, statistics=True)

        n_gauges = df["gauge_id"].n_unique()
        logger.info(f"Wrote {len(unique_types)} attribute type(s) for {n_gauges} gauges to {output_path}")

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
            return TimeSeriesCollection(group_tensors={}, feature_names=[], date_ranges={})

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
        group_tensors = {}
        date_ranges = {}

        # Get unique gauge_ids
        gauge_ids = df["gauge_id"].unique().to_list()

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
            group_tensors[gauge_id] = tensor

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

            date_ranges[gauge_id] = (min_date, max_date)

        # Validate all gauges have the same features in the same order
        if len(gauge_ids) > 1:
            first_gauge = gauge_ids[0]
            first_shape = group_tensors[first_gauge].shape[1]

            for gauge_id in gauge_ids[1:]:
                if group_tensors[gauge_id].shape[1] != first_shape:
                    raise ValueError(
                        f"Gauge '{gauge_id}' has {group_tensors[gauge_id].shape[1]} features, "
                        f"but gauge '{first_gauge}' has {first_shape} features. "
                        "All gauges must have the same features. "
                        "This indicates an upstream processing error."
                    )

        return TimeSeriesCollection(
            group_tensors=group_tensors,
            feature_names=feature_columns,
            date_ranges=date_ranges,
            validate=True,  # Always validate to catch any remaining issues
        )

    def to_static_attribute_collection(self, lf: pl.LazyFrame) -> StaticAttributeCollection:
        """
        Convert a LazyFrame to StaticAttributeCollection.

        Expects LazyFrame from get_static_attributes() with columns:
        - 'gauge_id': Group identifier (string)
        - attribute columns: All numeric attribute columns
        - 'REGION_NAME', 'attribute_type', 'data_type': Optional metadata (ignored)

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
            return StaticAttributeCollection(group_tensors={}, attribute_names=[])

        # Validate required columns
        if "gauge_id" not in df.columns:
            raise ValueError("Missing required column: gauge_id")

        # Auto-detect attribute columns (exclude metadata)
        metadata_columns = {"gauge_id", "REGION_NAME", "attribute_type", "data_type"}
        attribute_columns = [col for col in df.columns if col not in metadata_columns]

        if not attribute_columns:
            raise ValueError("No attribute columns found after excluding metadata")

        # Handle potential long format data - pivot if needed
        # Check if we have multiple rows per gauge (indicating long format)
        gauge_counts = df.group_by("gauge_id").len()
        max_count = gauge_counts["len"].max() if len(gauge_counts) > 0 else 0

        if max_count > 1:
            # Long format detected - need to pivot to wide format
            # This handles cases where attributes come from different attribute_types
            if "attribute_type" not in df.columns:
                raise ValueError(
                    "Multiple rows per gauge detected but no 'attribute_type' column found. "
                    "This indicates an upstream processing error."
                )

            # Get a single representative row per gauge to check structure
            first_rows = df.group_by("gauge_id").first()
            if len(first_rows) != df["gauge_id"].n_unique():
                raise ValueError("Inconsistent data structure - cannot reliably pivot to wide format")

            # Use the first row as our base and assume all attribute columns are present
            df = first_rows

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
        group_tensors = {}

        # Get unique gauge_ids
        gauge_ids = df["gauge_id"].unique().to_list()

        for gauge_id in gauge_ids:
            # Filter data for this gauge
            gauge_df = df.filter(pl.col("gauge_id") == gauge_id)

            # Should be exactly one row per gauge after handling long format
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
            group_tensors[gauge_id] = tensor

        # Validate all gauges have the same attributes in the same order
        if len(gauge_ids) > 1:
            first_gauge = gauge_ids[0]
            first_shape = group_tensors[first_gauge].shape[0]

            for gauge_id in gauge_ids[1:]:
                if group_tensors[gauge_id].shape[0] != first_shape:
                    raise ValueError(
                        f"Gauge '{gauge_id}' has {group_tensors[gauge_id].shape[0]} attributes, "
                        f"but gauge '{first_gauge}' has {first_shape} attributes. "
                        "All gauges must have the same attributes. "
                        "This indicates an upstream processing error."
                    )

        return StaticAttributeCollection(
            group_tensors=group_tensors,
            attribute_names=attribute_columns,
            validate=True,  # Always validate to catch any remaining issues
        )
