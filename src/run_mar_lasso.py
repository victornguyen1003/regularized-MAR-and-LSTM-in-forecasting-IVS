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
    def __init__(self, alpha_A: float | None = None, alpha_B: float | None = None):
        super().__init__()
        self.alpha_A = alpha_A
        self.alpha_B = alpha_B
        self.name = f"MAR_Lasso_{self.alpha_A}_{self.alpha_B}"
        self.n_splits = 5

    def search_alpha(self, Y_train_centered: pd.DataFrame, alpha_grid_A: np.ndarray, alpha_grid_B: np.ndarray, A_0: np.ndarray, B_0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tscv = TimeSeriesSplit(n_splits=self.n_splits).split(Y_train_centered)

        mse_results: dict[tuple[float, float], float] = {}

        for alpha_A in alpha_grid_A:
            for alpha_B in alpha_grid_B:
                logger.info(f"Training MAR Lasso with alpha_A={alpha_A}, alpha_B={alpha_B}...")

                mse_all_folds = []

                A, B = A_0, B_0
                p1, p2 = A.shape[0], B.shape[0]

                # Initialize Lasso model with the current alpha value
                lasso_A = Lasso(alpha=alpha_A, fit_intercept=False, max_iter=200, warm_start=True)
                lasso_B = Lasso(alpha=alpha_B, fit_intercept=False, max_iter=200, warm_start=True)

                for i, (train_index, test_index) in enumerate(tscv):
                    logger.info(f"TimeSeriesSplit fold {i+1}")
                    logger.debug(f"train index: {train_index}, test index: {test_index}")

                    # Extract training data
                    Y_train_fold, Y_test_fold = Y_train_centered.iloc[train_index], Y_train_centered.iloc[test_index]

                    y_A = np.array([Y_train_fold.iloc[i].unstack(level='tenor').values.flatten(order='F') for i in range(1, len(Y_train_fold))])
                    y_B = np.array([Y_train_fold.iloc[i].unstack(level='tenor').values.T.flatten(order='F') for i in range(1, len(Y_train_fold))])

                    does_converge = False
                    for iteration in range(5):
                        logger.info(f"Iteration {iteration+1}")

                        # Fit Lasso for A
                        Z_lagged_A = np.array([np.kron(B @ Y_train_fold.iloc[i].unstack(level='tenor').values.T, np.eye(p1)) for i in range(len(Y_train_fold[:-1]))])
                        lasso_A.fit(Z_lagged_A.reshape(-1, p1*p1), y_A.reshape(-1))
                        A_new = lasso_A.coef_.reshape((p1, p1), order='F')

                        # Fit Lasso for B
                        Z_lagged_B = np.array([np.kron(A_new @ Y_train_fold.iloc[i].unstack(level='tenor').values, np.eye(p2)) for i in range(len(Y_train_fold[:-1]))])
                        lasso_B.fit(Z_lagged_B.reshape(-1, p2*p2), y_B.reshape(-1))
                        B_new = lasso_B.coef_.reshape((p2, p2), order='F')

                        scale = np.linalg.norm(A_new, ord='fro')
                        if scale > 0:
                            A_new, B_new = A_new / scale, B_new * scale

                        # Check convergence
                        if np.linalg.norm(A - A_new) < CONVERGENCE_THRESHOLD and np.linalg.norm(B - B_new) < CONVERGENCE_THRESHOLD:
                            does_converge = True
                            logger.info(f"Converged after {iteration+1} iterations.")
                            A, B = A_new, B_new
                            break

                        A, B = A_new, B_new

                    forecast = Y_train_fold.iloc[-1].unstack('tenor').values
                    mse_one_fold = []
                    for i in range(len(Y_test_fold)):
                        actual = Y_test_fold.iloc[i].unstack('tenor').values
                        logger.debug(f"A shape: {A.shape}, B shape: {B.shape}, forecast shape: {forecast.shape}, actual shape: {actual.shape}")

                        forecast = A @ forecast @ B.T
                        mse_one_fold.append(np.linalg.norm(forecast - actual, ord='fro')**2)

                    mse_all_folds.append(np.mean(mse_one_fold))

                mse_results[(alpha_A, alpha_B)] = np.mean(mse_all_folds)
                logger.debug(f"MSE of MAR Lasso for alpha_A={alpha_A}, alpha_B={alpha_B}: {mse_results[(alpha_A, alpha_B)]}.")

        best_alpha_A, best_alpha_B = min(mse_results, key=mse_results.get)
        logger.info(f"Best alpha_A: {best_alpha_A}, Best alpha_B: {best_alpha_B} with MSE: {mse_results[(best_alpha_A, best_alpha_B)]}")

        self.alpha_A = best_alpha_A
        self.alpha_B = best_alpha_B

        return best_alpha_A, best_alpha_B

    def train(self, Y_train_centered: pd.DataFrame, A_0: np.ndarray, B_0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logger.info(f"Training MAR Lasso with alpha_A={self.alpha_A}, alpha_B={self.alpha_B}...")

        A, B = A_0, B_0
        p1, p2 = A.shape[0], B.shape[0]

        lasso_A = Lasso(alpha=self.alpha_A, fit_intercept=False, max_iter=3000, warm_start=True)
        lasso_B = Lasso(alpha=self.alpha_B, fit_intercept=False, max_iter=3000, warm_start=True)

        y_A = np.array([Y_train_centered.iloc[i].unstack(level='tenor').values.flatten(order='F') for i in range(1, len(Y_train_centered))])
        y_B = np.array([Y_train_centered.iloc[i].unstack(level='tenor').values.T.flatten(order='F') for i in range(1, len(Y_train_centered))])

        does_converge = False
        for iteration in range(3000):
            logger.info(f"Iteration {iteration+1}")

            # Fit Lasso for A
            Z_lagged_A = np.array([np.kron(B @ Y_train_centered.iloc[i].unstack(level='tenor').values.T, np.eye(p1)) for i in range(len(Y_train_centered)-1)])
            lasso_A.fit(Z_lagged_A.reshape(-1, p1*p1), y_A.reshape(-1))
            A_new = lasso_A.coef_.reshape((p1, p1), order='F')

            # Fit Lasso for B
            Z_lagged_B = np.array([np.kron(A_new @ Y_train_centered.iloc[i].unstack(level='tenor').values, np.eye(p2)) for i in range(len(Y_train_centered)-1)])
            lasso_B.fit(Z_lagged_B.reshape(-1, p2*p2), y_B.reshape(-1))
            B_new = lasso_B.coef_.reshape((p2, p2), order='F')

            scale = np.linalg.norm(A_new, ord='fro')
            if scale > 0:
                A_new, B_new = A_new / scale, B_new * scale

            # Check convergence
            if np.linalg.norm(A - A_new) < CONVERGENCE_THRESHOLD and np.linalg.norm(B - B_new) < CONVERGENCE_THRESHOLD:
                does_converge = True
                logger.info(f"Converged after {iteration+1} iterations.")
                A, B = A_new, B_new
                break

            A, B = A_new, B_new

        if not does_converge:
            logger.warning(f"Did not converge after 3000 iterations.")

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


    # Fine-tune hyperparameters 
    logger.info("Fine-tuning MAR LASSO hyperparameters alpha_A and alpha_B...")
    mar_res = MAR_Lasso_Results()
    A_lse = np.loadtxt(RES_DIR / 'A_lse.csv', delimiter=',')
    B_lse = np.loadtxt(RES_DIR / 'B_lse.csv', delimiter=',')

    # Search for alphas from coare grid
    alpha_parse_grid_A = np.logspace(-5, 5, 11)
    alpha_parse_grid_B = np.logspace(-5, 5, 11)

    logger.info(f"Searching for optimal alpha values for MAR LASSO from coarse grid {alpha_parse_grid_A} for alpha_A and {alpha_parse_grid_B} for alpha_B...")
    alpha_A_parse, alpha_B_parse = mar_res.search_alpha(X_train_centered, alpha_parse_grid_A, alpha_parse_grid_B, A_lse, B_lse)
    logger.info(f"Optimal alpha values found from coarse grid: alpha_A={alpha_A_parse}, alpha_B={alpha_B_parse}")

    # Search for alphas from fine grid
    alpha_fine_grid_A = np.logspace(np.log10(alpha_A_parse)-1, np.log10(alpha_A_parse)+1, 5)
    alpha_fine_grid_B = np.logspace(np.log10(alpha_B_parse)-1, np.log10(alpha_B_parse)+1, 5)

    logger.info(f"Searching for optimal alpha values for MAR LASSO from fine grid {alpha_fine_grid_A} for alpha_A and {alpha_fine_grid_B} for alpha_B...")
    alpha_A_fine, alpha_B_fine = mar_res.search_alpha(X_train_centered, alpha_fine_grid_A, alpha_fine_grid_B, A_lse, B_lse)
    logger.info(f"Optimal alpha values found from fine grid: alpha_A={alpha_A_fine}, alpha_B={alpha_B_fine}")


    # Train MAR model
    logger.info("Training MAR LASSO model...")
    A_lasso, B_lasso = mar_res.train(X_train_centered, A_lse, B_lse)

    np.savetxt(RES_DIR / 'A_lasso.csv', A_lasso, delimiter=',')
    logger.info(f"Successfully saved A_lasso to {RES_DIR / 'A_lasso.csv'} with shape {A_lasso.shape}")

    np.savetxt(RES_DIR / 'B_lasso.csv', B_lasso, delimiter=',')
    logger.info(f"Successfully saved B_lasso to {RES_DIR / 'B_lasso.csv'} with shape {B_lasso.shape}")


    # Test MAR model
    logger.info("Testing MAR model...")
    dates_to_test = X_test_centered.index
    mse_df = mar_res.test(dates_to_test, X_centered)


if __name__ == "__main__":
    main()




                





