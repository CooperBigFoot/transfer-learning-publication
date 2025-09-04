import logging
from datetime import datetime

from transfer_learning_publication.cleaners import GaugeCleaner, train_val_test_split
from transfer_learning_publication.data import CaravanDataSource

# Setup logging
logging.basicConfig(filename=f"logs/cleaning_{datetime.now():%Y%m%d_%H%M%S}.log", level=logging.INFO)


def clean_caravan_data(base_path: str, output_path: str, region: str, variables: list[str]):
    """Main cleaning function for a single region"""

    # Initialize data source
    caravan = CaravanDataSource(base_path, region=region)

    # Define cleaning pipeline once
    cleaner = (
        GaugeCleaner()
        .ensure_temporal_consistency()
        .trim_to_column("streamflow")
        .fill_na_columns(["streamflow"], fill_value=0.0, add_binary_flag=True)
        .clip_columns(["streamflow"], min_value=0.0)
        .add_cyclical_date_encoding()
    )

    gauge_ids = caravan.list_gauge_ids()

    logging.info(f"Processing {len(gauge_ids)} gauges in region {region}")

    for i, gauge_id in enumerate(gauge_ids, 1):
        try:
            lf = caravan.get_timeseries(gauge_ids=[gauge_id], variables=variables)

            lf_clean = cleaner.apply(lf)

            train, val, test = train_val_test_split(lf_clean, 0.5, 0.25)

            caravan.write_timeseries(train.collect(), f"{output_path}/train/")
            caravan.write_timeseries(val.collect(), f"{output_path}/val/")
            caravan.write_timeseries(test.collect(), f"{output_path}/test/")

            logging.info(f"  [{i}/{len(gauge_ids)}] ✓ {gauge_id}")

        except Exception as e:
            logging.error(f"  [{i}/{len(gauge_ids)}] ✗ {gauge_id}: {str(e)}")
            continue

    logging.info(f"Completed region {region}")


if __name__ == "__main__":
    regions = ["camels", "camelsaus", "camelsbr", "camelsch", "camelscl", "camelsgb", "hysets", "lamah", "tajikkyrgyz"]

    variables = [
        "snow_depth_water_equivalent_mean",
        "surface_net_solar_radiation_mean",
        "surface_net_thermal_radiation_mean",
        "potential_evaporation_sum_ERA5_LAND",
        "potential_evaporation_sum_FAO_PENMAN_MONTEITH",
        "temperature_2m_mean",
        "temperature_2m_min",
        "temperature_2m_max",
        "total_precipitation_sum",
        "streamflow",
    ]

    for region in regions:
        clean_caravan_data(
            base_path="/Users/cooper/Desktop/LSH_hive_data/",
            output_path="/Users/cooper/Desktop/CARAVAN_CLEAN/",
            region=region,
            variables=variables,
        )

    print("Cleaning complete. Check log for details.")
