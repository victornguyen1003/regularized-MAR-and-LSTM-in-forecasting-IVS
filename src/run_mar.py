import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime

from config import PROCESSED_DATA_DIR, RES_DIR, FORECAST_HORIZONS
from util import load_transformed_data, save_csv

import logging
logger = logging.getLogger(__name__)

res_dir = RES_DIR / "MAR"
res_dir.mkdir(parents=True, exist_ok=True)


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


def _iterate(X_train_centered: pd.DataFrame, A_proj: np.ndarray, B_proj: np.ndarray, max_iterations: int = 10000, convergence_threshold: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
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

    def set_params(self, A: np.ndarray, B: np.ndarray):
        self.A = A
        self.B = B

    def train(self, X_train_centered: pd.DataFrame, Phi_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dim_moneyness = X_train_centered.columns.get_level_values('moneyness').nunique()
        dim_tenor = X_train_centered.columns.get_level_values('tenor').nunique()
        
        A_proj, B_proj = _project(Phi_hat, dim_moneyness, dim_tenor)
        A_lse, B_lse = _iterate(X_train_centered, A_proj, B_proj)
        self.A, self.B = A_lse, B_lse

        np.savetxt(res_dir / 'A_lse.csv', A_lse, delimiter=',')
        logger.info(f"Successfully saved to {res_dir / 'A_lse.csv'} with shape {A_lse.shape}")
        np.savetxt(res_dir / 'B_lse.csv', B_lse, delimiter=',')
        logger.info(f"Successfully saved to {res_dir / 'B_lse.csv'} with shape {B_lse.shape}")

        return A_lse, B_lse

    def forecast(self, X: np.ndarray, steps: int) -> np.ndarray:
        """Forecast the next steps"""

        forecast = X
        for step in range(steps):
            forecast = self.A @ forecast @ self.B.T

        return forecast

    def compute_squared_error(self, horizon: int, date: datetime, X_centered: pd.DataFrame):
        """Compute the squared error for a given forecast horizon and date"""
        if not date in X_centered.index:
            logger.error(f"Date {date} not found in backtesting data")
            raise ValueError(f"Date {date} not found in backtesting data")

        start_index = X_centered.index.get_loc(date)-horizon
        if start_index < 0:
            logger.warning(f"Not enough data to backtest for date {date} - horizon {horizon}")
            return
        start_date = X_centered.index[start_index]
        
        input = X_centered.loc[start_date].unstack('tenor').values
        prediction = self.forecast(input, steps=horizon)
        actual_X = X_centered.loc[date].unstack('tenor').values

        self.forecasts[horizon].append(prediction)
        self.squared_errors[horizon].append((actual_X - prediction).flatten()**2)

    def compute_mse(self, horizon: int) -> float:
        """Compute the MSE from current SSE for a given forecast horizon"""
        if not horizon in list(self.forecasts.keys()):
            logger.error(f"Forecast horizon {horizon} not found.")
            raise KeyError(f"Forecast horizon {horizon} not found.")

        self.mse[horizon] = np.mean(self.squared_errors[horizon])
        return self.mse[horizon]
    
    def test(self, dates_to_test: pd.DatetimeIndex, X_centered: pd.DataFrame, training_set: bool = False):
        for h in FORECAST_HORIZONS:
            for d in dates_to_test:
                self.compute_squared_error(h, d, X_centered)

            self.compute_mse(h)
            logger.debug(f"MSE of {self.name} for horizon {h}: {self.mse[h]}.")

        df_mse = pd.DataFrame(self.mse.items(), columns=['h', self.name])
        out_name = f"{self.name}_mse.csv" if not training_set else f"{self.name}_training_mse.csv"
        save_csv(df_mse, res_dir / out_name)

        return df_mse


def main():
    logger.info("Loading data...")
    Phi_hat = np.loadtxt(RES_DIR / 'VAR' / 'VAR(1)_phi_hat.csv', delimiter=',')
    X_centered = load_transformed_data(PROCESSED_DATA_DIR / 'centered_full_data.csv')
    X_train_centered = load_transformed_data(PROCESSED_DATA_DIR / 'centered_train_data.csv')
    X_test_centered = load_transformed_data(PROCESSED_DATA_DIR / 'centered_test_data.csv')
    
    logger.info("Training MAR model...")
    mar_res = MAR_Results()
    A_lse, B_lse = mar_res.train(X_train_centered, Phi_hat)

    logger.info("Testing MAR model...")
    dates_to_test = X_test_centered.index
    mse_df = mar_res.test(dates_to_test, X_centered)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(name)s - %(levelname)s - %(message)s')
    main()




