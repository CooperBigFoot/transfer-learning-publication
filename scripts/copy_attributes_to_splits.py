import logging
from datetime import datetime
from pathlib import Path

from transfer_learning_publication.data import CaravanDataSource

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/attributes_copy_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),  # Also log to console
    ],
)


def copy_attributes_to_splits(
    raw_data_path: str, clean_data_path: str, regions: list[str], splits: list[str] = ["train", "val", "test"]
):
    """
    Copy static attributes from raw data to all train/val/test splits.

    Args:
        raw_data_path: Path to raw LSH_hive_data
        clean_data_path: Path to CARAVAN_CLEAN directory
        regions: List of regions to process
        splits: List of split names (default: train, val, test)
    """

    for region in regions:
        logging.info(f"Processing attributes for region: {region}")

        try:
            # Initialize data source for reading raw attributes
            caravan_reader = CaravanDataSource(raw_data_path, region=region)

            # Get all static attributes (lazy frame)
            # This will read all attribute types (caravan, hydroatlas, other)
            attributes_lf = caravan_reader.get_static_attributes()

            # Collect the attributes once
            attributes_df = attributes_lf.collect()
            logging.info(f"  Loaded {attributes_df.shape[0]} gauges with {attributes_df.shape[1]} columns")

            # Write the same attributes to each split
            for split in splits:
                output_path = Path(clean_data_path) / split

                # Initialize data source for writing
                caravan_writer = CaravanDataSource(output_path, region=region)

                # Write the attributes
                caravan_writer.write_static_attributes(
                    attributes_df,
                    output_path,
                    overwrite=True,  # Overwrite if exists
                )

                logging.info(f"  ✓ Written attributes to {split}")

            logging.info(f"✓ Completed region {region}")

        except Exception as e:
            logging.error(f"✗ Failed to process region {region}: {str(e)}")
            continue

    logging.info("All regions processed!")


def verify_attributes(clean_data_path: str, regions: list[str], splits: list[str] = None):
    """
    Verify that attributes were copied correctly by checking a sample.

    Args:
        clean_data_path: Path to CARAVAN_CLEAN directory
        regions: List of regions to verify
        splits: List of split names to check
    """
    if splits is None:
        splits = ["train", "val", "test"]

    logging.info("\n=== Verifying copied attributes ===")

    for region in regions:
        logging.info(f"Verifying region: {region}")

        try:
            # Check each split
            for split in splits:
                caravan = CaravanDataSource(Path(clean_data_path) / split, region=region)

                # Try to list attributes
                attrs = caravan.list_static_attributes()

                if attrs:
                    logging.info(f"  {split}: Found {len(attrs)} attribute columns")
                else:
                    logging.warning(f"  {split}: No attributes found!")

        except Exception as e:
            logging.error(f"  Error verifying {region}: {str(e)}")


if __name__ == "__main__":
    # Configuration
    RAW_DATA_PATH = "/Users/cooper/Desktop/LSH_hive_data/"
    CLEAN_DATA_PATH = "/Users/cooper/Desktop/CARAVAN_CLEAN/"

    REGIONS = ["camels", "camelsaus", "camelsbr", "camelsch", "camelscl", "camelsgb", "hysets", "lamah", "tajikkyrgyz"]

    SPLITS = ["train", "val", "test"]

    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Copy attributes to all splits
    logging.info("Starting attribute copy process...")
    copy_attributes_to_splits(
        raw_data_path=RAW_DATA_PATH, clean_data_path=CLEAN_DATA_PATH, regions=REGIONS, splits=SPLITS
    )

    # Verify the copy was successful
    verify_attributes(clean_data_path=CLEAN_DATA_PATH, regions=REGIONS, splits=SPLITS)

    logging.info("\n✨ Process complete! Check log for details.")
