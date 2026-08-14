from pyexpat import errors

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit

from pathlib import Path
from datetime import datetime

from config import PROCESSED_DATA_DIR, RES_DIR, MAX_ITERATIONS, CONVERGENCE_THRESHOLD, FORECAST_HORIZONS
from util import load_transformed_data, split_train_test_data

import logging
logger = logging.getLogger(__name__)

from run_mar import MAR_Results


class MAR_Lasso_Results(MAR_Results):
    def __init__(self, lambda_A: float | None = None, lambda_B: float | None = None):
        super().__init__()
        self.name = f"MAR_Lasso_{self.lambda_A}_{self.lambda_B}"
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B

    def search_alpha(self, Y_train_centered: pd.DataFrame, alpha_grid: list[float], A_0: np.ndarray, B_0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tscv = TimeSeriesSplit(n_splits=5).split(Y_train_centered)

        mse_results: dict[tuple[float, float], float] = {}

        for alpha_A in alpha_grid:
            for alpha_B in alpha_grid:
                logger.info(f"Training MAR Lasso with alpha_A={alpha_A}, alpha_B={alpha_B}...")

                mse_all_folds = []

                A, B = A_0, B_0
                p1, p2 = A.shape[0], B.shape[0]

                # Initialize Lasso model with the current alpha value
                lasso_A = Lasso(alpha=alpha_A, fit_intercept=False)
                lasso_B = Lasso(alpha=alpha_B, fit_intercept=False)

                for i, (train_index, test_index) in enumerate(tscv):
                    logger.info(f"TimeSeriesSplit fold {i+1}, - train index: {train_index}, test index: {test_index}")

                    # Extract training data
                    Y_train_fold, Y_test_fold = Y_train_centered.iloc[train_index], Y_train_centered.iloc[test_index]

                    y_A = np.array([Y.unstack(level='tenor').values.flatten(order='F') for Y in Y_train_fold[1:]])
                    y_B = np.array([Y.unstack(level='tenor').values.T.flatten(order='F') for Y in Y_train_fold[1:]])

                    does_converge = False
                    max_iterations = 50
                    for iteration in range(max_iterations):
                        logger.info(f"Iteration {iteration+1}")

                        # Fit Lasso for A
                        Z_lagged_A = np.array([np.kron(B @ Y.values.T, np.eye(p1)) for Y in Y_train_fold[:-1]])
                        lasso_A.fit(Z_lagged_A, y_A)
                        A_new = lasso_A.coef_.reshape(p1, p1)

                        # Fit Lasso for B
                        Z_lagged_B = np.array([np.kron(A_new @ Y.values, np.eye(p2)) for Y in Y_train_fold[:-1]])
                        lasso_B.fit(Z_lagged_B, y_B)
                        B_new = lasso_B.coef_.reshape(p2, p2)

                        # Check convergence
                        if np.linalg.norm(A - A_new) < CONVERGENCE_THRESHOLD and np.linalg.norm(B - B_new) < CONVERGENCE_THRESHOLD:
                            does_converge = True
                            logger.info(f"Converged after {iteration+1} iterations.")
                            A, B = A_new, B_new
                            break

                        A, B = A_new, B_new

                    forecast = Y_train_fold.iloc[-1].unstack('tenor').values.flatten(order='F')
                    mse_one_fold = []
                    for actual in Y_test_fold:
                        forecast = A @ forecast @ B.T
                        mse_one_fold.append(np.linalg.norm((forecast - actual.unstack('tenor').values.flatten(order='F')) ** 2))

                    mse_all_folds.append(np.mean(mse_one_fold))

                mse_results[(alpha_A, alpha_B)] = np.mean(mse_all_folds)
                logger.debug(f"MSE of MAR Lasso for alpha_A={alpha_A}, alpha_B={alpha_B}: {mse_results[(alpha_A, alpha_B)]}.")

        best_alpha_A, best_alpha_B = min(mse_results, key=mse_results.get)
        logger.info(f"Best alpha_A: {best_alpha_A}, Best alpha_B: {best_alpha_B} with MSE: {mse_results[(best_alpha_A, best_alpha_B)]}")

        self.lambda_A = best_alpha_A
        self.lambda_B = best_alpha_B

    def train(self, Y_train_centered: pd.DataFrame, A_0: np.ndarray, B_0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logger.info(f"Training MAR Lasso with lambda_A={self.lambda_A}, lambda_B={self.lambda_B}...")

        A, B = A_0, B_0
        p1, p2 = A.shape[0], B.shape[0]

        lasso_A = Lasso(alpha=self.lambda_A, fit_intercept=False)
        lasso_B = Lasso(alpha=self.lambda_B, fit_intercept=False)

        does_converge = False
        max_iterations = MAX_ITERATIONS
        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration+1}")

            # Fit Lasso for A
            Z_lagged_A = np.array([np.kron(B @ Y.values.T, np.eye(p1)) for Y in Y_train_centered[:-1]])
            y_A = np.array([Y.unstack(level='tenor').values.flatten(order='F') for Y in Y_train_centered[1:]])
            lasso_A.fit(Z_lagged_A, y_A)
            A_new = lasso_A.coef_.reshape(p1, p1)

            # Fit Lasso for B
            Z_lagged_B = np.array([np.kron(A_new @ Y.values, np.eye(p2)) for Y in Y_train_centered[:-1]])
            y_B = np.array([Y.unstack(level='tenor').values.T.flatten(order='F') for Y in Y_train_centered[1:]])
            lasso_B.fit(Z_lagged_B, y_B)
            B_new = lasso_B.coef_.reshape(p2, p2)

            # Check convergence
            if np.linalg.norm(A - A_new) < CONVERGENCE_THRESHOLD and np.linalg.norm(B - B_new) < CONVERGENCE_THRESHOLD:
                does_converge = True
                logger.info(f"Converged after {iteration+1} iterations.")
                A, B = A_new, B_new
                break

            A, B = A_new, B_new

        if not does_converge:
            logger.warning(f"Did not converge after {max_iterations} iterations.")

        self.A, self.B = A, B
    
        return A, B


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)


    # Load and split data
    logger.info("Loading and splitting data...")

    transformed_df = load_transformed_data(PROCESSED_DATA_DIR / 'transformed_data.csv')
    X_train, X_test = split_train_test_data(transformed_df)
    X_train_mean = np.mean(X_train, axis=0)
    X_train_centered, X_test_centered, X_centered = X_train - X_train_mean, X_test - X_train_mean, transformed_df - X_train_mean


    # Train MAR LASSO models
    logger.info("Training MAR LASSO model...")
    mar_res = MAR_Lasso_Results()
    A_lse = np.loadtxt(RES_DIR / 'A_lse.csv', delimiter=',')
    B_lse = np.loadtxt(RES_DIR / 'B_lse.csv', delimiter=',')

    alpha_grid = np.

    # Test MAR model
    logger.info("Testing MAR model...")
    dates_to_test = X_test_centered.index
    mse_df = mar_res.test(dates_to_test, X_centered)


if __name__ == "__main__":
    main()




                





