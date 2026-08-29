from run_lstm import VanillaLSTM, ResidualLSTM, LSTM_Results
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import json

from util import load_transformed_data, save_csv
from config import PROCESSED_DATA_DIR, RES_DIR
res_dir = RES_DIR / "LSTM"
res_dir.mkdir(parents=True, exist_ok=True)

import logging
logger = logging.getLogger(__name__)


def main():
    rng = np.random.default_rng()

    logger.info(f"Loading training data...")
    train_data = load_transformed_data(PROCESSED_DATA_DIR / "uncentered_train_data.csv")
    k_vars = train_data.shape[1]

    model_names = ["VanillaLSTM", "ResidualLSTM"]

    for model_name in model_names:
        logger.info(f"Starting finetuning {model_name}...")

        if model_name == "VanillaLSTM":
            model_class = VanillaLSTM
        elif model_name == "ResidualLSTM":
            model_class = ResidualLSTM

        best_params = {}
        best_mse = np.inf
        for iter in range(20):
            logger.info(f"Iteration {iter+1}/20...")

            params = {
                'hidden_dim': int(rng.choice([16, 32, 64, 128, 256])),
                'lr': float(rng.uniform(1e-5, 1e-2)),
                'weight_decay': float(rng.uniform(1e-5, 1e-2)),
                'dropout': float(rng.uniform(0.2, 0.5)),
                'num_layers': int(rng.choice([1, 2])),
                "batch_size": int(rng.choice([16, 32, 64, 128, 256])),
                "seq_len": int(rng.choice([10, 20, 30, 40, 50])),
                "max_epochs": int(rng.choice([10, 20, 30, 40, 50])),
            }
            logger.info(f"Current hyperparameters: {params}")

            model = model_class(
                input_dim=k_vars, 
                hidden_dim=params['hidden_dim'], 
                output_dim=k_vars, 
                lr=params['lr'], 
                weight_decay=params['weight_decay'], 
                dropout=params['dropout'], 
                num_layers=params['num_layers']
            )

            res = LSTM_Results(
                model=model, 
                name=model_name, 
                input_dim=k_vars, 
                output_dim=k_vars, 
                **params
            )
            
            tscv = TimeSeriesSplit(n_splits=5)
            mse_all_fold = []
            for fold_index, (train_index, val_index) in enumerate(tscv.split(train_data)):
                logger.info(f"Processing fold {fold_index}")

                train_fold = train_data.iloc[train_index]
                val_fold = train_data.iloc[val_index]

                res.train(train_fold)
                train_fold_scaled = res.scaler.transform(train_fold.values)
                val_fold_scaled = res.scaler.transform(val_fold.values)
                combined_scaled = np.concatenate([train_fold_scaled, val_fold_scaled], axis=0)
                
                val_start_idx = len(train_fold_scaled)

                mse_one_fold = []
                for i in range(len(val_fold)):
                    actual = val_fold.iloc[i].values
                    
                    input_seq = combined_scaled[val_start_idx + i - res.seq_len : val_start_idx + i]
                    
                    pred = res.forecast(input_seq, steps=1)
                    mse_one_fold.append(np.mean((pred - actual)**2))

                mse_all_fold.append(np.mean(mse_one_fold))

            current_mse = np.mean(mse_all_fold)
            logger.info(f"Current MSE: {current_mse}")
            if current_mse < best_mse:
                best_mse = current_mse
                best_params = params

        logger.info(f"Best parameters for {model_name}: {best_params}")
        logger.info(f"Best MSE for {model_name}: {best_mse}")

        out_path = res_dir / f"best_hyperparameters_{model_name}.json"
        with open(out_path, 'w') as f:
            json.dump(best_params, f)
        logger.info(f"Successfully saved to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO,
                        format ='%(asctime)s - %(levelname)s - %(message)s',
                        filename = res_dir / "finetuning_lstm.log",
                        filemode = 'w')
    main()