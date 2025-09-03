#!/usr/bin/env python3
# /// script
# dependencies = [
#   "pandas>=2.0.0",
#   "pyarrow>=10.0.0",
# ]
# ///
"""
Large Sample Hydrology Dataset - Hive Partitioning Transformation

Transform LSH dataset from nested CSV structure to clean, hive-partitioned Parquet format.

Usage:
    uv run scripts/caravan_to_hive.py [input_path]

Example:
    uv run scripts/caravan_to_hive.py /Users/cooper/Desktop/data_to_hive
"""

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd


def setup_logging() -> None:
    """Configure logging for progress tracking."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("lsh_transformation.log")],
    )


def validate_input_path(input_path: Path) -> None:
    """Validate input directory structure."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    required_dirs = ["attributes", "timeseries", "shapefiles"]
    for dir_name in required_dirs:
        dir_path = input_path / dir_name
        if not dir_path.exists():
            logging.warning(f"Expected directory missing: {dir_path}")


def handle_ca_renaming(df: pd.DataFrame, filename: str) -> tuple[pd.DataFrame, str]:
    """
    Rename CA references to tajikkyrgyz in data content and filenames.

    Args:
        df: DataFrame to process
        filename: Original filename

    Returns:
        Tuple of (processed_dataframe, new_filename)
    """
    # Rename columns containing CA
    df = df.rename(
        columns={
            col: col.replace("CA_", "tajikkyrgyz_").replace("_CA", "_tajikkyrgyz") for col in df.columns if "CA" in col
        }
    )

    # Replace CA values in string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.replace("CA_", "tajikkyrgyz_").str.replace("_CA", "_tajikkyrgyz")

    # Update filename
    new_filename = filename.replace("CA_", "tajikkyrgyz_").replace("_CA", "_tajikkyrgyz")

    return df, new_filename


def process_attributes(region: str, input_dir: Path, output_dir: Path) -> dict[str, int]:
    """
    Process attribute files for a region.

    Args:
        region: Region name (e.g., 'camels', 'tajikkyrgyz')
        input_dir: Input data directory
        output_dir: Output hive directory

    Returns:
        Dictionary with processing statistics
    """
    stats = {"processed": 0, "errors": 0}
    attributes_input = input_dir / "attributes" / region

    if not attributes_input.exists():
        logging.warning(f"No attributes directory for region: {region}")
        return stats

    # Create output directory structure
    region_output = output_dir / f"REGION_NAME={region}" / "data_type=attributes"
    region_output.mkdir(parents=True, exist_ok=True)

    # Map attribute file patterns to types
    attribute_types = ["caravan", "hydroatlas", "other"]

    for attr_type in attribute_types:
        # Handle different file patterns
        pattern = f"attributes_{attr_type}_CA.*" if region == "tajikkyrgyz" else f"attributes_{attr_type}_{region}.*"

        input_files = list(attributes_input.glob(pattern))

        if not input_files:
            logging.warning(f"No {attr_type} attributes found for {region}")
            continue

        input_file = input_files[0]  # Should be exactly one file per type
        output_path = region_output / f"attribute_type={attr_type}" / "data.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Read file (CSV or Parquet)
            if input_file.suffix.lower() == ".csv":
                # Use chunking for large CSV files
                chunks = []
                chunk_size = 10000
                for chunk in pd.read_csv(input_file, chunksize=chunk_size):
                    if region == "tajikkyrgyz":
                        chunk, _ = handle_ca_renaming(chunk, input_file.name)
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_parquet(input_file)
                if region == "tajikkyrgyz":
                    df, _ = handle_ca_renaming(df, input_file.name)

            # Write to Parquet
            df.to_parquet(output_path, compression="snappy")
            stats["processed"] += 1
            logging.info(f"Processed {region} {attr_type} attributes: {len(df)} rows")

        except Exception as e:
            logging.error(f"Error processing {input_file}: {e}")
            stats["errors"] += 1

    return stats


def process_timeseries(region: str, input_dir: Path, output_dir: Path) -> dict[str, int]:
    """
    Process timeseries files for a region.

    Args:
        region: Region name
        input_dir: Input data directory
        output_dir: Output hive directory

    Returns:
        Dictionary with processing statistics
    """
    stats = {"processed": 0, "errors": 0}
    timeseries_input = input_dir / "timeseries" / "csv" / region

    if not timeseries_input.exists():
        logging.warning(f"No timeseries directory for region: {region}")
        return stats

    # Create output directory structure
    region_output = output_dir / f"REGION_NAME={region}" / "data_type=timeseries"
    region_output.mkdir(parents=True, exist_ok=True)

    # Get all timeseries files
    if region == "tajikkyrgyz":
        timeseries_files = list(timeseries_input.glob("CA_*.parquet"))
    else:
        timeseries_files = list(timeseries_input.glob(f"{region}_*.csv"))

    for input_file in timeseries_files:
        try:
            # Extract gauge ID from filename
            gauge_id = input_file.stem

            # Read file
            df = pd.read_csv(input_file) if input_file.suffix.lower() == ".csv" else pd.read_parquet(input_file)

            # Handle CA renaming for tajikkyrgyz
            if region == "tajikkyrgyz":
                df, gauge_id = handle_ca_renaming(df, gauge_id)

            # Create gauge partition directory and write data
            gauge_output = region_output / f"gauge_id={gauge_id}"
            gauge_output.mkdir(parents=True, exist_ok=True)
            output_path = gauge_output / "data.parquet"

            df.to_parquet(output_path, compression="snappy")
            stats["processed"] += 1

            if stats["processed"] % 100 == 0:
                logging.info(f"Processed {stats['processed']} {region} timeseries files")

        except Exception as e:
            logging.error(f"Error processing {input_file}: {e}")
            stats["errors"] += 1

    logging.info(f"Completed {region} timeseries: {stats['processed']} files processed")
    return stats


def copy_shapefiles(region: str, input_dir: Path, output_dir: Path) -> dict[str, int]:
    """
    Copy shapefile components for a region.

    Args:
        region: Region name
        input_dir: Input data directory
        output_dir: Output hive directory

    Returns:
        Dictionary with processing statistics
    """
    stats = {"processed": 0, "errors": 0}
    shapefiles_input = input_dir / "shapefiles" / region

    if not shapefiles_input.exists():
        logging.warning(f"No shapefiles directory for region: {region}")
        return stats

    # Create output directory
    shapefiles_output = output_dir / f"REGION_NAME={region}" / "data_type=shapefiles"
    shapefiles_output.mkdir(parents=True, exist_ok=True)

    # Copy all shapefile components
    for input_file in shapefiles_input.iterdir():
        if input_file.is_file():
            try:
                output_file = shapefiles_output / input_file.name
                shutil.copy2(input_file, output_file)
                stats["processed"] += 1
            except Exception as e:
                logging.error(f"Error copying {input_file}: {e}")
                stats["errors"] += 1

    logging.info(f"Copied {stats['processed']} shapefile components for {region}")
    return stats


def print_summary(all_stats: dict[str, dict[str, dict[str, int]]]) -> None:
    """Print final processing summary."""
    print("\n" + "=" * 60)
    print("LARGE SAMPLE HYDROLOGY TRANSFORMATION SUMMARY")
    print("=" * 60)

    for region, region_stats in all_stats.items():
        print(f"\nRegion: {region.upper()}")
        print("-" * 40)

        for data_type, stats in region_stats.items():
            processed = stats.get("processed", 0)
            errors = stats.get("errors", 0)
            print(f"  {data_type:12}: {processed:4} processed, {errors:2} errors")

    # Overall totals
    total_processed = sum(
        stats.get("processed", 0) for region_stats in all_stats.values() for stats in region_stats.values()
    )
    total_errors = sum(stats.get("errors", 0) for region_stats in all_stats.values() for stats in region_stats.values())

    print("\n" + "-" * 40)
    print(f"TOTAL: {total_processed} processed, {total_errors} errors")
    print("=" * 60)


def transform_lsh_to_hive(input_path: str) -> None:
    """
    Main transformation function.

    Args:
        input_path: Path to 'data_to_hive' folder
    """
    setup_logging()
    logging.info("Starting Large Sample Hydrology dataset transformation")

    # Setup paths
    input_dir = Path(input_path)
    output_dir = input_dir.parent / "LSH_hive_data"

    # Validate input
    validate_input_path(input_dir)

    # Create output directory
    output_dir.mkdir(exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # List of regions to process
    regions = ["camels", "camelsaus", "camelsbr", "camelsch", "camelscl", "camelsgb", "hysets", "lamah", "tajikkyrgyz"]

    # Track statistics for all processing
    all_stats = {}

    # Process each region
    for region in regions:
        logging.info(f"\nProcessing region: {region}")
        region_stats = {}

        # Process attributes
        region_stats["attributes"] = process_attributes(region, input_dir, output_dir)

        # Process timeseries
        region_stats["timeseries"] = process_timeseries(region, input_dir, output_dir)

        # Copy shapefiles (skip tajikkyrgyz - no shapefiles in input)
        if region != "tajikkyrgyz":
            region_stats["shapefiles"] = copy_shapefiles(region, input_dir, output_dir)

        all_stats[region] = region_stats

    # Print final summary
    print_summary(all_stats)
    logging.info("Transformation completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform Large Sample Hydrology dataset to hive-partitioned format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run scripts/caravan_to_hive.py
    uv run scripts/caravan_to_hive.py /path/to/data_to_hive
        """,
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="/Users/cooper/Desktop/data_to_hive",
        help="Path to data_to_hive folder (default: /Users/cooper/Desktop/data_to_hive)",
    )

    args = parser.parse_args()
    transform_lsh_to_hive(args.input_path)
