import pandas as pd

from pathlib import Path
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

from config import PROCESSED_DATA_DIR


def extract_metadata(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Extract metadata from a dataframe columns

    Returns:
        tuple[pd.Series, pd.Series]: tenor and moneyness series
    """
    cols = df.columns.drop('Dates').astype(str)
    logger.debug(f"Extracted column names with dates: {cols}")

    meta = pd.Series(cols).str.extract(r"(?P<tenor>\d+)(?P<unit>DAY|MTH)_IMPVOL_(?P<moneyness>[0-9.]+)%MNY_DF")
    logger.debug(f"Extracted metadata: {meta} with shape {meta.shape}")

    tenor = meta['tenor'].astype('str').replace({'30': '1', '60': '2'}).drop_duplicates().astype(int).sort_values()
    moneyness = meta['moneyness'].astype(float).drop_duplicates().sort_values()
    logger.info(f"Extracted metadata: tenor={tenor}, moneyness={moneyness}")

    return tenor, moneyness


def load_transform_data(file_path: Path) -> pd.DataFrame:
    """Load and transform the data from a CSV file.
    """
    # Load data
    if not file_path.exists():
        logger.error(f"File {file_path} does not exist.")
        raise FileNotFoundError(f"File {file_path} does not exist.")

    df = pd.read_csv(file_path)
    logger.info(f"Successfully loaded data from {file_path} with shape {df.shape}")

    # Extract metadata
    tenor, moneyness = extract_metadata(df)

    # Set index and columns
    cols_MultiIndex = pd.MultiIndex.from_product([tenor, moneyness], names=['tenor', 'moneyness'])

    transformed = df.copy().set_index('Dates')
    transformed.index = pd.to_datetime(transformed.index, format='%m/%d/%Y')
    transformed.columns = cols_MultiIndex

    # Save and return transformed data
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(PROCESSED_DATA_DIR / "transformed_data.csv")
    logger.info(f"Successfully saved transformed data to {PROCESSED_DATA_DIR / 'transformed_data.csv'} with shape {transformed.shape}")

    return transformed


def extract_iv_matrix(df: pd.DataFrame, date: datetime) -> pd.DataFrame:
    """Extract the implied volatility matrix for a given date
    """
    if date not in df.index:
        logger.error(f"Date {date} not found in the dataframe index.")
        raise ValueError(f"Date {date} not found in the dataframe index.")

    iv_matrix = df.loc[date].unstack(level='moneyness')

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / f"iv_matrix_{date.strftime('%Y%m%d')}.csv"
    iv_matrix.to_csv(out_path)
    logger.info(f"Successfully saved implied volatility matrix for {date} to {out_path} with shape {iv_matrix.shape}")

    return iv_matrix