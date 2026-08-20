import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

from config import PROCESSED_DATA_DIR


def load_transformed_data(file_path: Path = PROCESSED_DATA_DIR / "transformed_data.csv") -> pd.DataFrame:
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


def save_csv(df: pd.DataFrame, file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    logger.info(f"Successfully saved to {file_path} with shape {df.shape}")