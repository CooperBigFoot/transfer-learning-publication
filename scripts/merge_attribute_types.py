#!/usr/bin/env python3
"""
Merge CARAVAN attribute types from separate partitions into unified files.

Problem: The CARAVAN dataset has attributes split across three types (caravan, hydroatlas, other),
each containing the same columns but with different subsets populated. This script merges them
into a single file per region/split by coalescing the non-null values.

Structure before:
    data_type=attributes/attribute_type={caravan,hydroatlas,other}/data.parquet

Structure after:
    data_type=attributes/data.parquet
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path

import polars as pl


def setup_logging(log_dir: Path = Path("logs")) -> logging.Logger:
    """Configure logging with file and console output."""
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"merge_attributes_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return logging.getLogger(__name__)


def read_attribute_partitions(attributes_path: Path) -> tuple[list[pl.DataFrame], list[str]]:
    """
    Read all attribute type partitions from a region.

    Args:
        attributes_path: Path to data_type=attributes directory

    Returns:
        Tuple of (list of dataframes, list of attribute type names)
    """
    dataframes = []
    attr_types = []

    # Find all attribute_type=* directories
    attr_dirs = sorted(attributes_path.glob("attribute_type=*"))

    for attr_dir in attr_dirs:
        attr_type = attr_dir.name.split("=")[1]
        parquet_file = attr_dir / "data.parquet"

        if parquet_file.exists():
            df = pl.read_parquet(parquet_file)
            dataframes.append(df)
            attr_types.append(attr_type)
            logging.info(f"    Loaded {attr_type}: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            logging.warning(f"    Missing data.parquet in {attr_dir}")

    return dataframes, attr_types


def merge_sparse_attributes(dataframes: list[pl.DataFrame]) -> pl.DataFrame:
    """
    Merge multiple dataframes with same schema but different non-null values.

    Strategy: Stack vertically, then group by gauge_id and take first non-null
    value for each column.

    Args:
        dataframes: List of dataframes with identical schemas

    Returns:
        Merged dataframe with one row per gauge_id
    """
    if len(dataframes) == 1:
        return dataframes[0]

    # Stack all dataframes vertically
    combined = pl.concat(dataframes, how="vertical")

    # Get all columns except gauge_id
    value_cols = [col for col in combined.columns if col != "gauge_id"]

    # Create aggregation to take first non-null value per column
    agg_exprs = [pl.col(col).drop_nulls().first().alias(col) for col in value_cols]

    # Group by gauge_id and coalesce
    merged = combined.group_by("gauge_id").agg(agg_exprs).sort("gauge_id")

    return merged


def backup_existing(attributes_path: Path, attr_dirs: list[Path]) -> None:
    """Create backup of existing attribute_type directories."""
    backup_dir = attributes_path / "backup_attribute_types"

    if not backup_dir.exists():
        backup_dir.mkdir(parents=True)

        for attr_dir in attr_dirs:
            backup_target = backup_dir / attr_dir.name
            if not backup_target.exists():
                shutil.copytree(attr_dir, backup_target)

        logging.info(f"    Created backup in {backup_dir}")


def process_region(base_path: Path, region: str, split: str, backup: bool = True, delete_old: bool = True) -> bool:
    """
    Process a single region/split combination.

    Args:
        base_path: Root CARAVAN_CLEAN directory
        region: Region name (e.g., 'camels')
        split: Split name ('train', 'val', or 'test')
        backup: Whether to backup existing directories
        delete_old: Whether to delete old attribute_type directories

    Returns:
        True if successful, False otherwise
    """
    attributes_path = base_path / split / f"REGION_NAME={region}" / "data_type=attributes"

    # Check if path exists
    if not attributes_path.exists():
        logging.warning(f"  Path not found: {attributes_path}")
        return False

    # Check if already merged
    merged_file = attributes_path / "data.parquet"
    attr_dirs = list(attributes_path.glob("attribute_type=*"))

    if merged_file.exists() and not attr_dirs:
        logging.info(f"  Already merged: {region}/{split}")
        return True

    if not attr_dirs:
        logging.warning(f"  No attribute_type directories found in {region}/{split}")
        return False

    try:
        # Read all partitions
        dataframes, attr_types = read_attribute_partitions(attributes_path)

        if not dataframes:
            logging.warning(f"  No valid data found in {region}/{split}")
            return False

        logging.info(f"  Merging {len(dataframes)} attribute types: {attr_types}")

        # Merge the dataframes
        merged_df = merge_sparse_attributes(dataframes)

        # Calculate statistics
        n_gauges = merged_df.shape[0]
        n_attrs = merged_df.shape[1] - 1  # Excluding gauge_id

        # Calculate sparsity
        null_counts = sum(merged_df[col].null_count() for col in merged_df.columns if col != "gauge_id")
        total_cells = n_gauges * n_attrs
        sparsity = (null_counts / total_cells * 100) if total_cells > 0 else 0

        # Backup if requested
        if backup:
            backup_existing(attributes_path, attr_dirs)

        # Write merged file
        merged_df.write_parquet(merged_file, use_pyarrow=True, statistics=True)

        logging.info(f"  ✓ Merged {n_gauges} gauges × {n_attrs} attributes")
        logging.info(f"    Sparsity: {sparsity:.1f}% null values")
        logging.info(f"    Output: {merged_file}")

        # Delete old directories if requested
        if delete_old:
            for attr_dir in attr_dirs:
                shutil.rmtree(attr_dir)
            logging.info(f"    Removed {len(attr_dirs)} old directories")

        return True

    except Exception as e:
        logging.error(f"  ✗ Failed to process {region}/{split}: {e}")
        import traceback

        logging.error(traceback.format_exc())
        return False


def verify_merge(base_path: Path, region: str, split: str) -> dict | None:
    """
    Verify merged attributes file and return statistics.

    Args:
        base_path: Root CARAVAN_CLEAN directory
        region: Region name
        split: Split name

    Returns:
        Dictionary with statistics or None if file not found
    """
    merged_file = base_path / split / f"REGION_NAME={region}" / "data_type=attributes" / "data.parquet"

    if not merged_file.exists():
        return None

    try:
        df = pl.read_parquet(merged_file)

        # Calculate statistics
        n_gauges = df.shape[0]
        n_attrs = df.shape[1] - 1

        # Find columns with most/least data
        completeness = []
        for col in df.columns:
            if col != "gauge_id":
                non_null = df.shape[0] - df[col].null_count()
                completeness.append((col, non_null))

        completeness.sort(key=lambda x: x[1], reverse=True)

        # Overall sparsity
        total_nulls = sum(df[col].null_count() for col in df.columns if col != "gauge_id")
        total_cells = n_gauges * n_attrs
        sparsity = (total_nulls / total_cells * 100) if total_cells > 0 else 0

        return {
            "n_gauges": n_gauges,
            "n_attributes": n_attrs,
            "sparsity_pct": sparsity,
            "most_complete": completeness[:5],
            "least_complete": completeness[-5:],
            "has_duplicates": df["gauge_id"].n_unique() != n_gauges,
        }

    except Exception as e:
        logging.error(f"Error verifying {region}/{split}: {e}")
        return None


def main():
    """Main execution function."""
    # Configuration
    CLEAN_DATA_PATH = Path("/Users/cooper/Desktop/CARAVAN_CLEAN/")

    REGIONS = ["camels", "camelsaus", "camelsbr", "camelscl", "camelsch", "camelsgb", "hysets", "lamah", "tajikkyrgyz"]

    SPLITS = ["train", "val", "test"]

    # Setup logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("CARAVAN Attribute Type Merge")
    logger.info("=" * 60)

    # Process each split and region
    success_count = 0
    fail_count = 0

    for split in SPLITS:
        logger.info(f"\nProcessing split: {split}")
        logger.info("-" * 40)

        for region in REGIONS:
            logger.info(f"\n{region}:")
            success = process_region(
                base_path=CLEAN_DATA_PATH,
                region=region,
                split=split,
                backup=True,
                delete_old=True,  # Set to True after verifying results
            )

            if success:
                success_count += 1
            else:
                fail_count += 1

    # Verification phase
    logger.info("\n" + "=" * 60)
    logger.info("Verification")
    logger.info("=" * 60)

    for split in SPLITS:
        logger.info(f"\n{split}:")

        for region in REGIONS:
            stats = verify_merge(CLEAN_DATA_PATH, region, split)

            if stats:
                logger.info(
                    f"  {region}: {stats['n_gauges']} gauges, "
                    f"{stats['n_attributes']} attrs, "
                    f"{stats['sparsity_pct']:.1f}% sparse"
                )

                if stats["has_duplicates"]:
                    logger.warning("    ⚠️ Duplicate gauge_ids detected!")
            else:
                logger.warning(f"  {region}: Not found or error")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Failed: {fail_count}")

    if fail_count == 0:
        logger.info("\n✅ All regions processed successfully!")
        logger.info("To clean up old directories, run again with delete_old=True")
    else:
        logger.warning(f"\n⚠️ {fail_count} regions failed. Check logs for details.")


if __name__ == "__main__":
    main()
