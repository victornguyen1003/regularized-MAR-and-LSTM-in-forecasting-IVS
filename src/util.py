import pandas as pd

from pathlib import Path
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

from config import PROCESSED_DATA_DIR, TRAIN_PROPORTION


def _extract_metadata(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Extract and return tenor and moneyness series from dataframe columns."""
    cols = df.columns.drop('Dates').astype(str)
    logger.debug(f"Extracted column names with dates: {cols}")

    meta = pd.Series(cols).str.extract(r"(?P<tenor>\d+)(?P<unit>DAY|MTH)_IMPVOL_(?P<moneyness>[0-9.]+)%MNY_DF")
    logger.debug(f"Extracted metadata: {meta} with shape {meta.shape}")

    tenor = meta['tenor'].astype('str').replace({'30': '1', '60': '2'}).drop_duplicates().astype(int).sort_values()
    moneyness = meta['moneyness'].astype(float).drop_duplicates().sort_values()
    logger.info(f"Extracted metadata: tenor={tenor}, moneyness={moneyness}")

    return tenor, moneyness


def load_raw_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        logger.error(f"File {file_path} does not exist.")
        raise FileNotFoundError(f"File {file_path} does not exist.")

    df = pd.read_csv(file_path)
    logger.info(f"Successfully loaded raw data from {file_path} with shape {df.shape}")

    return df


def transform_raw_data(df) -> pd.DataFrame:
    """transform the data from a CSV file."""

    # Extract metadata
    tenor, moneyness = _extract_metadata(df)

    # Set index and columns
    cols_MultiIndex = pd.MultiIndex.from_product([tenor, moneyness], names=['tenor', 'moneyness'])

    transformed = df.copy().set_index('Dates')
    transformed.index = pd.to_datetime(transformed.index, format='%m/%d/%Y')
    transformed.columns = cols_MultiIndex
    transformed = transformed.sort_index()

    # Save and return transformed data
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(PROCESSED_DATA_DIR / "transformed_data.csv")
    logger.info(f"Successfully saved transformed data to {PROCESSED_DATA_DIR / 'transformed_data.csv'} with shape {transformed.shape}")

    return transformed


def load_transformed_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        logger.error(f"File {file_path} does not exist.")
        raise FileNotFoundError(f"File {file_path} does not exist.")

    df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
    logger.info(f"Successfully loaded transformed data from {file_path} with shape {df.shape}")

    return df


def extract_iv_matrix(df: pd.DataFrame, date: datetime, save: bool = False) -> pd.DataFrame:
    """Extract the implied volatility matrix for a given date."""

    if date not in df.index:
        logger.error(f"Date {date} not found in the dataframe index.")
        raise ValueError(f"Date {date} not found in the dataframe index.")

    iv_matrix = df.loc[date].unstack(level='moneyness')

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DATA_DIR / f"iv_matrix_{date.strftime('%Y%m%d')}.csv"
        iv_matrix.to_csv(out_path)
        logger.info(f"Successfully saved implied volatility matrix for {date} to {out_path} with shape {iv_matrix.shape}")

    return iv_matrix


def split_train_test_data(df: pd.DataFrame, train_proportion: float = TRAIN_PROPORTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into training and testing sets based on the given proportion.
    """
    n = len(df)
    cutoff_index = int(n * train_proportion)
    cutoff_date = df.index[cutoff_index]

    train_df = df.iloc[:cutoff_index]
    test_df = df.iloc[cutoff_index:]

    logger.info(f"Data split into training and testing sets with cutoff date {cutoff_date}.")
    logger.info(f"Training set shape: {train_df.shape}, Testing set shape: {test_df.shape}")

    return train_df, test_df