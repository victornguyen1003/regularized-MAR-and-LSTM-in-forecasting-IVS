from statsmodels.tsa.api import VAR

import numpy as np
import pandas as pd

from datetime import datetime
from collections import defaultdict
from pathlib import Path

from config import PROCESSED_DATA_DIR, TRAIN_PROPORTION, FORECAST_HORIZONS, RES_DIR

import logging
logger = logging.getLogger(__name__)

class VAR_Results:
    def __init__(self, res: "VARResults"):
        self.res = res
        self.p = res.k_ar
        self.forecasts = defaultdict(list)
        self.squared_errors = defaultdict(list)
        self.mse = defaultdict(int)
        self.name = f"VAR({self.p})"

    def compute_squared_error(self, h: int, d: datetime, y: pd.DataFrame) -> float:
        """Compute forecast error for a given horizon and date, and store it in the errors dictionary.
        
        Args:
            h: forecast horizon
            d: date to forecast
            y: data to backtest
        """
        if not d in y.index:
            logger.error(f"Date {d} not found in backtesting data.")
            raise ValueError(f"Date {d} not found in backtesting data.")

        actual_y = y.loc[d]

        d = y.index.get_loc(d)

        start_index, end_index = d-h-self.p+1, d-h+1
        if start_index < 0:
            logger.warning(f"Not enough data to backtest for horizon {h}.")
            return
        
        inputs = y.iloc[start_index:end_index].to_numpy()
        prediction = self.res.forecast(inputs, steps=h)[-1]

        self.forecasts[h].append(prediction)
        self.squared_errors[h].append((actual_y - prediction)**2)

    def compute_mse(self, h: int) -> float:
        """Compute the mean squared error from the current sum of squared errors for a given forecast horizon.

        Args:
            h: forecast horizon
        """
        if not h in list(self.forecasts.keys()):
            logger.error(f"Forecast horizon {h} not found.")
            raise KeyError(f"Forecast horizon {h} not found.")

        self.mse[h] = np.mean(self.squared_errors[h])
        return self.mse[h]


def train_and_test(horizons: list[int], y: pd.DataFrame, cutoff_date: datetime) -> pd.DataFrame:
    """Compute the MSE table for VAR models for a given list of forecast horizons.

    Args:
        models: list of VAR model results
        horizons: list of forecast horizons
        y: date-indexed data for backtesting
        cutoff_date: date from which forecasts are made for backtesting
    Returns:
        MSE table
    """
    # Split dataset
    cutoff_index = y.index.get_loc(cutoff_date)
    y_train, y_test = y.iloc[:cutoff_index], y.iloc[cutoff_index:]
    logger.info(f"# observations: total={len(y)}, train={len(y_train)}, test={len(y_test)}.")


    logger.info("Starting training...")
    var = VAR(y_train.to_numpy().astype(float))

    logger.info("Fitting VAR(1)...")
    res_var_1 = VAR_Results(var.fit(1))

    logger.info("Fitting VAR(p) with p selected by AIC...")
    res_var_aic = VAR_Results(var.fit(maxlags=15, ic='aic'))
    logger.info(f"p selected by AIC: {res_var_aic.p}")

    logger.info("Fitting VAR(p) with p selected by BIC...")
    res_var_bic = VAR_Results(var.fit(maxlags=15, ic='bic'))
    logger.info(f"p selected by BIC: {res_var_bic.p}")


    # Make a list of models with distinct numbers of lags
    var_models = [res_var_1]
    if res_var_aic.p != 1:
        var_models.append(res_var_aic)
    if res_var_bic.p != 1 and res_var_bic.p != res_var_aic.p:
        var_models.append(res_var_bic)


    # Extract forecast dates for backtesting
    dates_to_test = y_test.index

    df_mse = pd.DataFrame({'h': horizons})

    logger.info("Starting testing...")
    for model in var_models:
        for h in horizons:
            for d in dates_to_test:
                model.compute_squared_error(h, d, y)

            model.compute_mse(h)
            logger.info(f"MSE of {model.name} for horizon {h}: {model.mse[h]}.")

        df_single_mse = pd.DataFrame(model.mse.items(), columns=['h', model.name])
        df_mse = df_mse.merge(df_single_mse, on='h')

    RES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RES_DIR / "var_mse.csv"

    df_mse.to_csv(out_path, index=False)
    logger.info(f"Successfully saved MSE table to {out_path}.")

    return df_mse


def main():
    logging.basicConfig(
        level = logging.INFO,
        format = "%(name)s - %(levelname)s - %(message)s",
        )
    logger = logging.getLogger(__name__)

    file_path = Path(PROCESSED_DATA_DIR) / "transformed_data.csv"

    df = pd.read_csv(file_path, header=[0,1], index_col=0, parse_dates=True)

    # Check if the first column contains all missing values just to hold the index name
    if df.iloc[0].isna().all():
        df = df.iloc[1:]

    df.index = pd.to_datetime(df.index)

    logger.info("\n=== Data Split ===")
    n = len(df)
    cutoff_index = int(n*TRAIN_PROPORTION)
    cutoff_date = df.index[cutoff_index]
    logger.info(f"Cutoff date from which forecasts for backtesting are made: {cutoff_date}")

    logger.info("\n=== VAR Models ===")
    train_and_test(horizons=FORECAST_HORIZONS, y=df, cutoff_date=cutoff_date)


if __name__ == "__main__":
    main()