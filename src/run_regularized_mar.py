from pyexpat import errors

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit

from pathlib import Path
from datetime import datetime

from config import PROCESSED_DATA_DIR, RES_DIR, MAX_ITERATIONS, CONVERGENCE_THRESHOLD, FORECAST_HORIZONS
from util import load_transformed_data, split_train_test_data

import logging
logger = logging.getLogger(__name__)

from run_mar import MAR_Results


def _iterate(A_0: np.ndarray, B_0: np.ndarray, alpha_A: float, alpha_B: float, max_iter: int, max_iter_elastic_net: int, Y_train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    A, B = A_0, B_0
    p1, p2 = A.shape[0], B.shape[0]

    elastic_net_A = ElasticNet(alpha=alpha_A, fit_intercept=False, max_iter=max_iter_elastic_net, tol=1e-2, warm_start=True, l1_ratio=0.95)
    elastic_net_B = ElasticNet(alpha=alpha_B, fit_intercept=False, max_iter=max_iter_elastic_net, tol=1e-2, warm_start=True, l1_ratio=0.95)

    y_A = np.array([Y_train.iloc[i].unstack(level='tenor').values.flatten(order='F') for i in range(1, len(Y_train))])
    y_B = np.array([Y_train.iloc[i].unstack(level='tenor').values.T.flatten(order='F') for i in range(1, len(Y_train))])

    delta_A, delta_B = np.inf, np.inf

    does_converge = False
    for iteration in range(max_iter):
        logger.info(f"Iteration {iteration+1}")

        # Fit Elastic Net for A
        Z_lagged_A = np.array([np.kron(B @ Y_train.iloc[i].unstack(level='tenor').values.T, np.eye(p1)) for i in range(len(Y_train[:-1]))])
        elastic_net_A.fit(Z_lagged_A.reshape(-1, p1*p1), y_A.reshape(-1))
        A_new = elastic_net_A.coef_.reshape((p1, p1), order='F')

        # Fit Elastic Net for B
        Z_lagged_B = np.array([np.kron(A_new @ Y_train.iloc[i].unstack(level='tenor').values, np.eye(p2)) for i in range(len(Y_train[:-1]))])
        elastic_net_B.fit(Z_lagged_B.reshape(-1, p2*p2), y_B.reshape(-1))
        B_new = elastic_net_B.coef_.reshape((p2, p2), order='F')

        scale = np.linalg.norm(A_new, ord='fro')
        if scale > 0:
            A_new, B_new = A_new / scale, B_new * scale

        # Check convergence
        delta_A, delta_B = np.linalg.norm(A - A_new), np.linalg.norm(B - B_new)
        if delta_A < CONVERGENCE_THRESHOLD and delta_B < CONVERGENCE_THRESHOLD:
            does_converge = True
            logger.info(f"Converged after {iteration+1} iterations.")
            A, B = A_new, B_new
            break

        A, B = A_new, B_new

    if not does_converge:
        logger.warning(f"Did not converge after {max_iter} iterations with delta_A={delta_A}, delta_B={delta_B}.")

    return A, B


class Regularized_MAR_Results(MAR_Results):
    def __init__(self, alpha_A: float | None = None, alpha_B: float | None = None):
        super().__init__()
        self.alpha_A = alpha_A
        self.alpha_B = alpha_B
        self.name = f"regularized_MAR"

    def search_alpha(self, Y_train_centered: pd.DataFrame, alpha_grid_A: np.ndarray, alpha_grid_B: np.ndarray, A_0: np.ndarray, B_0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tscv = list(TimeSeriesSplit(n_splits=5).split(Y_train_centered))

        mse_results: dict[tuple[float, float], float] = {}

        for alpha_A in alpha_grid_A:
            for alpha_B in alpha_grid_B:
                logger.info(f"Training Regularized MAR with alpha_A={alpha_A}, alpha_B={alpha_B}...")

                mse_all_folds = []

                for i, (train_index, test_index) in enumerate(tscv):
                    logger.info(f"TimeSeriesSplit fold {i+1}")
                    logger.debug(f"train index: {train_index}, test index: {test_index}")

                    # Extract training data
                    Y_train_fold, Y_test_fold = Y_train_centered.iloc[train_index], Y_train_centered.iloc[test_index]

                    A, B = _iterate(A_0, B_0, alpha_A, alpha_B, max_iter=20, max_iter_elastic_net=200, Y_train=Y_train_fold)

                    forecast = Y_train_fold.iloc[-1].unstack('tenor').values
                    mse_one_fold = []
                    for i in range(len(Y_test_fold)):
                        actual = Y_test_fold.iloc[i].unstack('tenor').values
                        logger.debug(f"A shape: {A.shape}, B shape: {B.shape}, forecast shape: {forecast.shape}, actual shape: {actual.shape}")

                        forecast = A @ forecast @ B.T
                        mse_one_fold.append(np.linalg.norm(forecast - actual, ord='fro')**2)

                    mse_all_folds.append(np.mean(mse_one_fold))

                mse_results[(alpha_A, alpha_B)] = np.mean(mse_all_folds)
                logger.debug(f"MSE of Regularized MAR for alpha_A={alpha_A}, alpha_B={alpha_B}: {mse_results[(alpha_A, alpha_B)]}.")

        best_alpha_A, best_alpha_B = min(mse_results, key=mse_results.get)
        logger.info(f"Best alpha_A: {best_alpha_A}, Best alpha_B: {best_alpha_B} with MSE: {mse_results[(best_alpha_A, best_alpha_B)]}")

        self.alpha_A = best_alpha_A
        self.alpha_B = best_alpha_B

        return best_alpha_A, best_alpha_B

    def train(self, Y_train_centered: pd.DataFrame, A_0: np.ndarray, B_0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logger.info(f"Training Regularized MAR with alpha_A={self.alpha_A}, alpha_B={self.alpha_B}...")

        A, B = _iterate(A_0, B_0, self.alpha_A, self.alpha_B, max_iter=1000, max_iter_elastic_net=1000, Y_train=Y_train_centered)
        self.A, self.B = A, B

        return A, B


def main():
    print(f"Starting regularized MAR training and testing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")

    logging.basicConfig(
        filename = RES_DIR / f"regularized_MAR.log",
        filemode = 'a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    logging.captureWarnings(True)
    logger = logging.getLogger(__name__)


    # Load and split data
    logger.info("Loading and splitting data...")

    transformed_df = load_transformed_data(PROCESSED_DATA_DIR / 'transformed_data.csv')
    X_train, X_test = split_train_test_data(transformed_df)
    X_train_mean = np.mean(X_train, axis=0)
    X_train_centered, X_test_centered, X_centered = X_train - X_train_mean, X_test - X_train_mean, transformed_df - X_train_mean


    # Fine-tune hyperparameters 
    logger.info("Fine-tuning Regularized MAR hyperparameters alpha_A and alpha_B...")
    mar_res = Regularized_MAR_Results(0.01, 0.01)
    A_lse = np.loadtxt(RES_DIR / 'A_lse.csv', delimiter=',')
    B_lse = np.loadtxt(RES_DIR / 'B_lse.csv', delimiter=',')

    # Search for alphas from coare grid
    alpha_parse_grid_A = np.logspace(-5, 5, 11)
    alpha_parse_grid_B = np.logspace(-5, 5, 11)

    logger.info(f"Searching for optimal alpha values for Regularized MAR from coarse grid {alpha_parse_grid_A} for alpha_A and {alpha_parse_grid_B} for alpha_B...")
    alpha_A_parse, alpha_B_parse = mar_res.search_alpha(X_train_centered, alpha_parse_grid_A, alpha_parse_grid_B, A_lse, B_lse)
    logger.info(f"Optimal alpha values found from coarse grid: alpha_A={alpha_A_parse}, alpha_B={alpha_B_parse}")

    # Search for alphas from fine grid
    alpha_fine_grid_A = np.logspace(np.log10(alpha_A_parse)-1, np.log10(alpha_A_parse)+1, 5)
    alpha_fine_grid_B = np.logspace(np.log10(alpha_B_parse)-1, np.log10(alpha_B_parse)+1, 5)

    logger.info(f"Searching for optimal alpha values for Regularized MAR from fine grid {alpha_fine_grid_A} for alpha_A and {alpha_fine_grid_B} for alpha_B...")
    alpha_A_fine, alpha_B_fine = mar_res.search_alpha(X_train_centered, alpha_fine_grid_A, alpha_fine_grid_B, A_lse, B_lse)
    logger.info(f"Optimal alpha values found from fine grid: alpha_A={alpha_A_fine}, alpha_B={alpha_B_fine}")


    # Train MAR model
    logger.info("Training Regularized MAR model...")
    regularized_A, regularized_B = mar_res.train(X_train_centered, A_lse, B_lse)

    np.savetxt(RES_DIR / 'regularized_A.csv', regularized_A, delimiter=',')
    logger.info(f"Successfully saved regularized A to {RES_DIR / 'regularized_A.csv'} with shape {regularized_A.shape}")

    np.savetxt(RES_DIR / 'regularized_B.csv', regularized_B, delimiter=',')
    logger.info(f"Successfully saved regularized B to {RES_DIR / 'regularized_B.csv'} with shape {regularized_B.shape}")


    # Test MAR model
    logger.info("Testing Regularized MAR model...")
    dates_to_test = X_test_centered.index
    mse_df = mar_res.test(dates_to_test, X_centered)

    print(f"Finished regularized MAR training and testing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")

if __name__ == "__main__":
    main()




                





