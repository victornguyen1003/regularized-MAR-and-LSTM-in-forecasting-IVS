import pandas as pd
import numpy as np

from config import RES_DIR, PROCESSED_DATA_DIR
from util import load_transformed_data, save_csv

res_dir = RES_DIR / "MAR"
res_dir.mkdir(parents=True, exist_ok=True)

from run_mar import MAR_Results
from run_regularized_mar import Regularized_MAR_Results

import logging
logger = logging.getLogger(__name__)


def main():
    print(f"Examining MAR models...")

    # Load and split data
    logger.info("Loading data...")

    X_centered = load_transformed_data(PROCESSED_DATA_DIR / 'centered_full_data.csv')
    X_train_centered = load_transformed_data(PROCESSED_DATA_DIR / 'centered_train_data.csv')
    X_test_centered = load_transformed_data(PROCESSED_DATA_DIR / 'centered_test_data.csv')
    

    # Load and examine parameters
    logger.info("Loading and examining parameters...")

    params: dict[str, np.ndarray] = {}
    eigenvalues: dict[str, float] = {}
    distances: dict[str, float] = {}
    param_names = ['A_lse', 'B_lse', 'regularized_A', 'regularized_B']

    for param_name in param_names:
        param = np.loadtxt(res_dir / (param_name + '.csv'), delimiter=',')
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
    save_csv(eigenvalues_df, res_dir / "MAR_eigenvalues.csv")

    distances_df = pd.DataFrame([distances])
    save_csv(distances_df, res_dir / "MAR_coefficient_matrix_distance_from_identity.csv")


    # Compute MAR training errors
    dates_to_test = X_train_centered.index

    logger.info("Computing training MSE for MAR(1)...")
    mar_res = MAR_Results()
    mar_res.set_params(params['A_lse'], params['B_lse'])
    mar_mse_df = mar_res.test(dates_to_test, X_train_centered, training_set=True)

    logger.info("Computing training MSE for Regularized MAR...")
    regularized_mar_res = Regularized_MAR_Results()
    regularized_mar_res.set_params(params['regularized_A'], params['regularized_B'])
    regularized_mar_mse_df = regularized_mar_res.test(dates_to_test, X_train_centered, training_set=True)

    print(f"Finished examining MAR models.")
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename= res_dir /'examining_MAR.log',
                        filemode= 'w',)
    main()