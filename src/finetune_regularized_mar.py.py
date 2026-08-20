import numpy as np
import pandas as pd

from run_regularized_mar import Regularized_MAR_Results

from config import PROCESSED_DATA_DIR, RES_DIR
from util import load_transformed_data

res_dir = RES_DIR / "MAR"
res_dir.mkdir(parents=True, exist_ok=True)

import json
from datetime import datetime

import logging
logger = logging.getLogger(__name__)


def main():
    print(f"Starting finetuning regularized MAR at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")

    # Load and split data
    logger.info("Loading data...")
    X_centered = load_transformed_data(PROCESSED_DATA_DIR / 'centered_full_data.csv')
    X_train_centered = load_transformed_data(PROCESSED_DATA_DIR / 'centered_train_data.csv')
    X_test_centered = load_transformed_data(PROCESSED_DATA_DIR / 'centered_test_data.csv')

        
    # Fine-tune hyperparameters 
    logger.info("Fine-tuning Regularized MAR hyperparameters alpha_A and alpha_B...")
    mar_res = Regularized_MAR_Results()
    A_lse = np.loadtxt(RES_DIR / 'MAR' / 'A_lse.csv', delimiter=',')
    B_lse = np.loadtxt(RES_DIR / 'MAR' / 'B_lse.csv', delimiter=',')

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

    best_alpha = {'best_alpha_A': alpha_A_fine, 'best_alpha_B': alpha_B_fine}
    with open(res_dir / 'best_alpha.json', 'w') as f:
        json.dump(best_alpha, f)

if __name__ == "__main__":
    logging.basicConfig(filename = res_dir / f"finetuning_regularized_MAR.log",
                        filemode = 'w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',)
    logging.captureWarnings(True)
    main()