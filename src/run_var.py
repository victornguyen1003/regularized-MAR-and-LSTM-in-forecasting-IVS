from statsmodels.tsa.api import VAR
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from typing import List
from collections import defaultdict

from config import PROCESSED_DATA_DIR, RES_DIR, FORECAST_HORIZONS
from util import load_transformed_data, save_csv

res_dir = RES_DIR / "VAR"
res_dir.mkdir(parents=True, exist_ok=True)

import logging
logger = logging.getLogger(__name__)


class VAR_Results:
    def __init__(self, res: "VARResults"):
        self.res = res
        self.p = res.k_ar
        self.phi_hat = res.params
        self.name = f"VAR({self.p})"

        self.forecasts = defaultdict(list)
        self.squared_errors = defaultdict(list)
        self.mse = defaultdict(int)

    def compute_test_mse(self, y: pd.DataFrame, horizons: List[int], cutoff_date: datetime) -> float:
        """Compute and store forecast error."""

        dates_to_test = y.loc[cutoff_date:].index

        for h in horizons:
            for d in dates_to_test:
                start_index, end_index = y.index.get_loc(d) - h - self.p + 1, y.index.get_loc(d) - h + 1
                if start_index < 0:
                    logger.warning(f"Not enough data to backtest for date {d} - horizon {h}.")
                    continue
                inputs = y.iloc[start_index:end_index].to_numpy()

                actual = y.loc[d]
                predicted = self.res.forecast(inputs, steps=h)[-1]
                self.squared_errors[h].append((actual - predicted)**2)

            self.mse[h] = np.mean(self.squared_errors[h])
            logger.debug(f"MSE of {self.name} for horizon {h}: {self.mse[h]}.")


def train_test(y: pd.DataFrame, horizons: List[int] = FORECAST_HORIZONS, train_proportion: float = 0.8) -> pd.DataFrame:
    """Train and test VAR(1), VAR(p) with p selected by AIC and by BIC"""

    cutoff_index = int(len(y) * train_proportion)
    cutoff_date = y.index[cutoff_index]
    y_train, y_test = y.iloc[:cutoff_index], y.iloc[cutoff_index:]
    logger.info(f"Cutoff date from which forecasts for backtesting are made: {cutoff_date}")

    logger.info("Starting training...")
    var = VAR(y_train.to_numpy().astype(float))

    logger.info("Fitting VAR(1)...")
    res_var_1 = VAR_Results(var.fit(1, trend='n'))

    logger.info("Fitting VAR(p) with p selected by AIC...")
    res_var_aic = VAR_Results(var.fit(maxlags=15, ic='aic', trend='n'))
    logger.info(f"p selected by AIC: {res_var_aic.p}")

    logger.info("Fitting VAR(p) with p selected by BIC...")
    res_var_bic = VAR_Results(var.fit(maxlags=15, ic='bic', trend='n'))
    logger.info(f"p selected by BIC: {res_var_bic.p}")

    # Filter out models with distinct p values
    var_models = [res_var_1]
    if res_var_aic.p != 1:
        var_models.append(res_var_aic)
    if res_var_bic.p != 1 and res_var_bic.p != res_var_aic.p:
        var_models.append(res_var_bic)

    # Save phi_hat
    for model in var_models:
        np.savetxt(res_dir / f"{model.name}_phi_hat.csv", model.phi_hat, delimiter=",")
        logger.info(f"Successfully saved to {res_dir / f'{model.name}_phi_hat.csv'} with shape {model.phi_hat.shape}")

    logger.info("Starting testing...")

    mse_map = {'h': horizons}

    for model in var_models:
        model.compute_test_mse(y, horizons, cutoff_date)
        mse_map[model.name] = list(model.mse.values())

    mse_df = pd.DataFrame(mse_map)
    save_csv(mse_df, res_dir / "VAR_mse.csv")


def main():
    logger.info("\n=== VAR Models ===")

    logger.info("Loading data...")
    df = load_transformed_data(PROCESSED_DATA_DIR / 'centered_full_data.csv')
    train_test(y=df, horizons=FORECAST_HORIZONS)


if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO,
                        format = "%(name)s - %(levelname)s - %(message)s")
    main()