import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime

from config import PROCESSED_DATA_DIR, RES_DIR, MAX_ITERATIONS, CONVERGENCE_THRESHOLD, FORECAST_HORIZONS
from util import load_transformed_data, split_train_test_data

import logging
logger = logging.getLogger(__name__)


def _take_inverse(M):
    try:
        return np.linalg.inv(M)
    except:
        return np.linalg.pinv(M)


def _rearrange(P: np.ndarray, m: int, n: int) -> np.ndarray:
    """Rearrangement operator G: R^{mn x mn} -> R^{m^2 x n^2}, G(B kron A) = vec(A)vec(B)'.
    Args:
        P: B kron A matrix
        m: dimension of square matrix A
        n: dimension of square matrix B

    Returns:
        Rearranged matrix.
    """
    return P.reshape(n,m,n,m).transpose(3,1,2,0).reshape(m*m,n*n)


def _project(Phi_hat: np.ndarray, m: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Use projection method to obtain two coefficient matrix estimators A and B from the VAR coefficient matrix estimator Phi_hat.
    Args:
        Phi_hat: VAR coefficient matrix estimator
        m: dimension of square matrix A
        n: dimension of square matrix B

    Returns:
        Tuple of coefficient matrix estimators A and B.
    """
    Phi_tilda = _rearrange(Phi_hat, m, n)

    U, S, Vt = np.linalg.svd(Phi_tilda)
    u1, d1, v1 = U[:,0], S[0], Vt[0,:]
    logger.debug(f"Singular value: {d1}, u1: {u1}, v1: {v1}")

    vec_A_hat = np.sqrt(d1) * u1
    vec_B_hat = np.sqrt(d1) * v1
    logger.debug(f"vec_A_hat: {vec_A_hat}, vec_B_hat: {vec_B_hat}")

    A_proj = vec_A_hat.reshape((m,m), order='F')
    B_proj = vec_B_hat.reshape((n,n), order='F')
    logger.debug(f"A_proj: {A_proj}, B_proj: {B_proj}")

    scale = np.linalg.norm(A_proj, ord='fro')
    A_proj = A_proj / scale
    B_proj = B_proj * scale
    logger.debug(f"Scaled A_proj: {A_proj}, Scaled B_proj: {B_proj}")

    return A_proj, B_proj    


def _iterate(X_train_centered: pd.DataFrame, A_proj: np.ndarray, B_proj: np.ndarray, max_iterations: int = MAX_ITERATIONS, convergence_threshold: float = CONVERGENCE_THRESHOLD):
    """Iteratively update the coefficient matrix estimators A and B until convergence.
    Args:
        X_train_centered: training data with mean subtracted
        A_proj: initial guess for A from projection method
        B_proj: initial guess for B from projection method
        max_iterations: maximum number of iterations
        convergence_threshold: convergence threshold

    Returns:
        Tuple of coefficient matrix estimators A and B.
    """
    data_centered = np.array([X_train_centered.iloc[i].unstack('tenor').values for i in range(len(X_train_centered))])

    X = data_centered[1:]
    X_lagged = data_centered[:-1]

    A_prev, B_prev = A_proj, B_proj

    does_converge = False

    for iteration in range(max_iterations):
        # Update A and B
        A = np.sum(X @ B_prev @ X_lagged.transpose(0,2,1), axis=0) @ _take_inverse(np.sum(X_lagged @ B_prev.T @ B_prev @ X_lagged.transpose(0,2,1), axis=0))
        B = np.sum(X.transpose(0,2,1) @ A @ X_lagged, axis=0) @ _take_inverse(np.sum(X_lagged.transpose(0,2,1) @ A.T @ A @ X_lagged, axis=0))

        scale = np.linalg.norm(A, ord='fro')
        A = A / scale
        B = B * scale

        delta_A = np.linalg.norm(A_prev - A, ord='fro')
        delta_B = np.linalg.norm(B_prev - B, ord='fro')
        logger.debug(f"Iteration {iteration + 1} - delta_A: {delta_A}, delta_B: {delta_B}")

        if delta_A < convergence_threshold and delta_B < convergence_threshold:
            logger.info(f"Converged after {iteration + 1} iterations with delta_A: {delta_A}, delta_B: {delta_B}")
            does_converge = True
            break

        A_prev, B_prev = A, B

    if not does_converge:
        logger.warning(f"Did not converge after {max_iterations} iterations with delta_A: {delta_A}, delta_B: {delta_B}")

    return A, B


class MAR_Results:
    def __init__(self, A: np.ndarray | None = None, B: np.ndarray | None = None):
        self.A = A
        self.B = B
        self.name = "MAR(1)"
        self.forecasts: dict[int, list[float]] = {h: [] for h in FORECAST_HORIZONS}
        self.squared_errors: dict[int, list[float]] = {h: [] for h in FORECAST_HORIZONS}
        self.mse: dict[int, float] = {}

    def set_params(self, A: np.ndarray | None = None, B: np.ndarray | None = None):
        self.A = A
        self.B = B

    def train(self, X_train_centered: pd.DataFrame, Phi_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dim_moneyness = X_train_centered.columns.get_level_values('moneyness').nunique()
        dim_tenor = X_train_centered.columns.get_level_values('tenor').nunique()
        
        A_proj, B_proj = _project(Phi_hat, dim_moneyness, dim_tenor)
        A_lse, B_lse = _iterate(X_train_centered, A_proj, B_proj)
        self.A, self.B = A_lse, B_lse

        return A_lse, B_lse

    def forecast(self, X: np.ndarray, steps: int) -> np.ndarray:
        """Forecast the next steps"""

        forecast = X
        for step in range(steps):
            forecast = self.A @ forecast @ self.B.T

        return forecast

    def compute_squared_error(self, h: int, d: datetime, X_centered: pd.DataFrame):
        """Compute the squared error for a given forecast horizon and date.

        Args:
            h: forecast horizon
            d: date to forecast
            y: data to backtest
        """
        if not d in X_centered.index:
            logger.error(f"Date {d} not found in backtesting data.")
            raise ValueError(f"Date {d} not found in backtesting data.")

        actual_X = X_centered.loc[d].unstack('tenor').values

        start_index = X_centered.index.get_loc(d)-h
        if start_index < 0:
            logger.warning(f"Not enough data to backtest for horizon {h}.")
            return
        start_date = X_centered.index[start_index]
        
        input = X_centered.loc[start_date].unstack('tenor').values
        # print(f"input: {input}, A: {self.A}, B: {self.B}") # Debugging line to check values
        # print(f"input shape: {input.shape}, A shape: {self.A.shape}, B shape: {self.B.shape}") # Debugging line to check shapes
        prediction = self.forecast(input, steps=h)

        self.forecasts[h].append(prediction)
        self.squared_errors[h].append(np.linalg.norm(actual_X - prediction, ord='fro')**2)

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
    

    def test(self, dates_to_test: pd.DatetimeIndex, X_centered: pd.DataFrame, training_set: bool = False) -> pd.DataFrame:
        """Compute the MSE table for MAR models for a given list of forecast horizons."""

        logger.info("Starting testing...")
        for h in FORECAST_HORIZONS:
            for d in dates_to_test:
                self.compute_squared_error(h, d, X_centered)

            self.compute_mse(h)
            logger.debug(f"MSE of {self.name} for horizon {h}: {self.mse[h]}.")

        df_mse = pd.DataFrame(self.mse.items(), columns=['h', self.name])
        RES_DIR.mkdir(parents=True, exist_ok=True)
        out_name = f"{self.name}_mse.csv" if not training_set else f"{self.name}_training_mse.csv"
        out_path = RES_DIR / out_name
        df_mse.to_csv(out_path, index=False)
        logger.info(f"Successfully saved MSE table to {out_path}.")

        return df_mse


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)


    # Load and split data
    logger.info("Loading and splitting data...")

    Phi_hat = np.loadtxt(RES_DIR / 'VAR(1)_phi_hat.csv', delimiter=',')

    transformed_df = load_transformed_data(PROCESSED_DATA_DIR / 'transformed_data.csv')
    X_train, X_test = split_train_test_data(transformed_df)
    X_train_mean = np.mean(X_train, axis=0)
    X_train_centered, X_test_centered, X_centered = X_train - X_train_mean, X_test - X_train_mean, transformed_df - X_train_mean

    # Train MAR model
    logger.info("Training MAR model...")
    mar_res = MAR_Results()
    A_lse, B_lse = mar_res.train(X_train_centered, Phi_hat)

    # Save coefficient matrix estimators
    np.savetxt(RES_DIR / 'A_lse.csv', A_lse, delimiter=',')
    logger.info(f"Successfully saved A_lse to {RES_DIR / 'A_lse.csv'} with shape {A_lse.shape}")

    np.savetxt(RES_DIR / 'B_lse.csv', B_lse, delimiter=',')
    logger.info(f"Successfully saved B_lse to {RES_DIR / 'B_lse.csv'} with shape {B_lse.shape}")


    # Test MAR model
    logger.info("Testing MAR model...")
    dates_to_test = X_test_centered.index
    mse_df = mar_res.test(dates_to_test, X_centered)


if __name__ == "__main__":
    main()




