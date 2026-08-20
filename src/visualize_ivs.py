import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

import logging
logger = logging.getLogger(__name__)

from util import extract_iv_matrix, load_transformed_data
from config import FIG_SIZE_3D, FIG_DIR, PROCESSED_DATA_DIR


def plot_ivs(df: pd.DataFrame, date: datetime) -> None:
    """Plot the implied volatility surface for a given date."""
    logger.info(f"Plotting implied volatility surface for date={date}.")

    iv_matrix = extract_iv_matrix(df, date)

    tenor, moneyness = iv_matrix.index.to_numpy(), iv_matrix.columns.to_numpy()
    nx, ny = len(tenor), len(moneyness)

    X, Y = np.meshgrid(np.arange(ny), np.arange(nx))
    logger.info(f"X={moneyness} with shape {X.shape}, Y={tenor} with shape {Y.shape}")

    Z = iv_matrix.values.reshape(nx, ny)
    logger.info(f"Z={Z} with shape {Z.shape}")

    fig, ax = plt.subplots(figsize=FIG_SIZE_3D, subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xticks(np.arange(ny))
    ax.set_xticklabels(moneyness, fontsize=8)
    ax.set_xlabel('Moneyness (%)', fontsize=10, labelpad=10)

    ax.set_yticks(np.arange(nx))
    ax.set_yticklabels(tenor, fontsize=8, rotation=30)
    ax.set_ylabel('Tenor (M)', fontsize=10, labelpad=10)

    ax.set_title(f"IVS for {date.strftime('%Y-%m-%d')}", fontsize=12)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"ivs_{date.strftime('%Y%m%d')}.png"

    plt.savefig(out_path)    
    logger.info(f"Successfully saved implied volatility surface for {date.strftime('%Y-%m-%d')} to {out_path}.")

    plt.close(fig)


def main():
    logger.info("\n=== Visualization ===")

    logger.info("Loading data for visualization...")
    file_path = PROCESSED_DATA_DIR / "transformed_data.csv"
    df = load_transformed_data(file_path)

    date = datetime(2026,5,25)
    plot_ivs(df, date)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                        format="%(name)s - %(levelname)s - %(message)s",)
    main()

