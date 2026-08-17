import pandas as pd
import numpy as np

from config import RES_DIR, PROCESSED_DATA_DIR
from util import load_transformed_data, split_train_test_data

from run_mar import MAR_Results
from run_regularized_mar import Regularized_MAR_Results

import logging
logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s',
        filename=RES_DIR/'MAR_examination.log',
        filemode='w',
    )
    logger = logging.getLogger(__name__)

    print(f"Examining MAR models...")

    # Load and split data
    logger.info("Loading and splitting data...")

    transformed_df = load_transformed_data(PROCESSED_DATA_DIR / 'transformed_data.csv')
    X_train, X_test = split_train_test_data(transformed_df)
    X_train_mean = np.mean(X_train, axis=0)
    X_train_centered, X_test_centered, X_centered = X_train - X_train_mean, X_test - X_train_mean, transformed_df - X_train_mean
    logger.info(f"Shape of X_train_centered: {X_train_centered.shape}, X_test_centered: {X_test_centered.shape}, X_centered: {X_centered.shape}")
    

    # Load and examine parameters
    logger.info("Loading and examining parameters...")

    params: dict[str, np.ndarray] = {}
    eigenvalues: dict[str, float] = {}
    distances: dict[str, float] = {}
    param_names = ['A_lse', 'B_lse', 'regularized_A', 'regularized_B']

    for param_name in param_names:
        param = np.loadtxt(RES_DIR / (param_name + '.csv'), delimiter=',')
        params[param_name] = param
        logger.info(f"{param_name}: {param}")

        param_eigenvalues = np.linalg.eigvals(param)
        eigenvalues[param_name] = param_eigenvalues
        logger.info(f"Eigenvalues of {param_name}: {param_eigenvalues}")

        id_matrix = np.eye(*param.shape)
        dist = np.linalg.norm(param - id_matrix, ord='fro')
        distances[param_name] = dist
        logger.info(f"Distance from {param_name} to identity matrix of shape {id_matrix.shape}: {dist}")

    eigenvalues_df = pd.DataFrame({k: pd.Series(v).sort_values(ascending=False) for k, v in eigenvalues.items()})
    eigenvalues_df.to_csv(RES_DIR / "MAR_eigenvalues.csv", index=False)

    distances_df = pd.DataFrame([distances])
    distances_df.to_csv(RES_DIR / "MAR_coefficient_matrix_distance_from_identity.csv", index=False)

    # Compute MAR training errors
    dates_to_test = X_train_centered.index

    logger.info("Computing training MSE for MAR(1)...")
    mar_res = MAR_Results()
    mar_res.set_params(params['A_lse'], params['B_lse'])
    mar_mse_df = mar_res.test(dates_to_test, X_train_centered, training_set=True)

    # Compute regularized MAR training errors
    regularized_mar_res = Regularized_MAR_Results(0.01, 0.01)
    regularized_mar_res.set_params(params['regularized_A'], params['regularized_B'])
    regularized_mar_mse_df = regularized_mar_res.test(dates_to_test, X_train_centered, training_set=True)

    print(f"Examination ends.")

if __name__ == "__main__":
    main()