import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime

from config import PROCESSED_DATA_DIR, RES_DIR, MAX_ITERATIONS, CONVERGENCE_THRESHOLD, FORECAST_HORIZONS
from util import load_transformed_data, split_train_test_data

import logging
logger = logging.getLogger(__name__)

from run_mar import MAR_Results

class MAR_Lasso_Results(MAR_Results):
    def __init__(self, A: np.ndarray | None = None, B: np.ndarray | None = None, lambda_A: float | None = None, lambda_B: float | None = None):
        super().__init__(A, B)
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
    