import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

import logging
logger = logging.getLogger(__name__)

from util import extract_iv_matrix
from config import FIG_SIZE_3D, FIG_DIR


def plot_ivs(df: pd.DataFrame, date: datetime) -> None:
    """Plot the implied volatility surface for a given date."
    """
    logger.info(f"Plotting implied volatility surface for date={date}.")

    iv_matrix = extract_iv_matrix(df, date)

    tenor, moneyness = iv_matrix.index.unique().sort_values(), iv_matrix.columns.unique().sort_values()
    nx, ny = len(tenor), len(moneyness)

    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    logger.info(f"X={tenor} with shape {X.shape}, Y={moneyness} with shape {Y.shape}")

    Z = iv_matrix.values.reshape(nx, ny)
    logger.info(f"Z={Z} with shape {Z.shape}")

    fig, ax = plt.subplots(figsize=FIG_SIZE_3D, subplot_kw={"projection": "3d"})
    ax.plot_surface(Y, X, Z=Z, cmap='viridis')

    ax.set_xticks(np.arange(nx))
    ax.set_xticklabels(tenor, fontsize=8, rotation=30)
    ax.set_xlabel('Tenor (M)', fontsize=10, labelpad=10)

    ax.set_yticks(np.arange(ny))
    ax.set_yticklabels(moneyness, fontsize=8)
    ax.set_ylabel('Moneyness (%)', fontsize=10, labelpad=10)

    ax.set_title(f"IVS for {date.strftime('%Y-%m-%d')}", fontsize=12)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"ivs_{date.strftime('%Y%m%d')}.png"

    plt.savefig(out_path)    
    logger.info(f"Successfully saved implied volatility surface for {date.strftime('%Y-%m-%d')} to {out_path}.")

    plt.close(fig)

