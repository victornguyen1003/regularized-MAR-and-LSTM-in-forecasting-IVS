from pathlib import Path

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

FIG_DIR = Path(__file__).parent.parent / "asset/figure"
FIG_SIZE_3D = (10,8)

RES_DIR = Path(__file__).parent.parent / "data/result"

TRAIN_PROPORTION = 0.8

FORECAST_HORIZONS = [1, 14, 30, 90, 180, 365]

# For MAR model
MAX_ITERATIONS = 1000
CONVERGENCE_THRESHOLD = 1e-6
