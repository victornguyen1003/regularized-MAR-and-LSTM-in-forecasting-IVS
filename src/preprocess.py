import pandas as pd
import numpy as np

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from util import split_center_data

from pathlib import Path

import logging
logger = logging.getLogger(__name__)


def load_raw_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        logger.error(f"File {file_path} does not exist.")
        raise FileNotFoundError(f"File {file_path} does not exist.")

    df = pd.read_csv(file_path)
    logger.info(f"Successfully loaded raw data from {file_path} with shape {df.shape}")

    return df


def transform_raw_data(df) -> pd.DataFrame:
    """transform the data from a CSV file."""

    cols = df.columns.drop('Dates').astype(str)
    meta = cols.to_series().str.extract(
        r"(?P<tenor>\d+)(?P<unit>DAY|MTH)_IMPVOL_(?P<moneyness>[0-9.]+)%MNY_DF"
    )
    meta['tenor'] = meta['tenor'].replace({'30': '1', '60': '2'}).astype(int)
    meta['moneyness'] = meta['moneyness'].astype(float)
    columns = pd.MultiIndex.from_frame(meta[['tenor', 'moneyness']])
    columns.names = ['tenor', 'moneyness']

    transformed = df.copy().set_index('Dates')
    transformed.index = pd.to_datetime(transformed.index, format='%m/%d/%Y')
    transformed.columns = columns
    transformed = transformed.sort_index(axis=0).sort_index(axis=1)

    # Save and return transformed data
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(PROCESSED_DATA_DIR / "transformed_data.csv")
    logger.info(f"Successfully saved transformed data to {PROCESSED_DATA_DIR / 'transformed_data.csv'} with shape {transformed.shape}")

    return transformed


def split_center_data(df: pd.DataFrame, train_proportion: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into training and testing sets based on the given proportion.
    """
    n = len(df)
    cutoff_index = int(n * train_proportion)
    cutoff_date = df.index[cutoff_index]
    logger.info(f"Cutoff date from which forecasts for backtesting are made: {cutoff_date}")

    df_train, df_test = df.iloc[:cutoff_index], df.iloc[cutoff_index:]
    train_mean = np.mean(df_train, axis=0)

    datasets = [df, df_train, df_test]
    names = ['centered_full_data', 'centered_train_data', 'centered_test_data']
    for dataset, name in zip(datasets, names):
        dataset_centered = dataset - train_mean

        out_path = PROCESSED_DATA_DIR / f"{name}.csv"
        dataset_centered.to_csv(out_path)
        logger.info(f"Successfully saved centered {name} to {out_path} with shape {dataset_centered.shape}")

    return df_train, df_test


def main():
    logger.info("\n=== Data Preprocessing ===")

    logger.info("Loading raw data...")
    file_path = Path(RAW_DATA_DIR) / "ivs_spx_16_26.csv"
    df = load_raw_data(file_path)

    logger.info("Transforming raw data...")
    df = transform_raw_data(df)

    logger.info("Splitting and centering data...")
    df_train, df_test = split_center_data(df, train_proportion=0.8)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                        format="%(name)s - %(levelname)s - %(message)s",)
    main()