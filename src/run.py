import config
import util
import visualize

from pathlib import Path
from datetime import datetime

from run_var import train_and_test as run_var

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("\n=== Data Preprocessing ===")
    # Load and transform data
    file_path = Path(config.RAW_DATA_DIR) / "ivs_spx_16_26.csv"
    df = util.load_raw_data(file_path)
    df = util.transform_raw_data(df)

    logger.info("\n=== Visualization ===")
    # Plot implied volatility surface for a specific date
    date = datetime(2016,4,1)
    visualize.plot_ivs(df, date)

    logger.info("\n=== Data Split ===")
    n = len(df)
    cutoff_index = int(n*config.TRAIN_PROPORTION)
    cutoff_date = df.index.get_loc(cutoff_index)
    logger.info(f"Cutoff date from which forecasts for backtesting are made: {cutoff_date}")

    logger.info("\n=== VAR Models ===")
    run_var(horizons=config.FORECAST_HORIZONS, y=df, cutoff_date=cutoff_date)

    logger.info("\n=== MAR Model ===")

if __name__ == "__main__":
    main()