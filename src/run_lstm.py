import numpy as np
import pandas as pd

import lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler

from datetime import datetime
from pathlib import Path
import json

from util import load_transformed_data, save_csv
from config import PROCESSED_DATA_DIR, RES_DIR, FORECAST_HORIZONS
res_dir = RES_DIR / "LSTM"
res_dir.mkdir(parents=True, exist_ok=True)

import logging
logger = logging.getLogger(__name__)


#Hyperparameter: hidden_dim, learning_rate, dropout, weightdecay
class VanillaLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, lr, weight_decay, dropout, num_layers):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        last_hidden_state = self.lstm(x)[0][:,-1,:]
        last_hidden_state = self.dropout(last_hidden_state)
        return self.fc(last_hidden_state)

    def training_step(self, batch: tuple):
        x, y = batch
        y_hat = self(x)
        return torch.nn.functional.mse_loss(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class ResidualLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, lr, weight_decay, dropout, num_layers):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        last_input = x[:,-1,:]
        last_hidden_state = self.dropout(self.lstm(x)[0][:,-1,:])
        delta = self.fc(last_hidden_state)  
        return delta + last_input

    def training_step(self, batch: tuple):
        x, y = batch
        y_hat = self(x)
        return torch.nn.functional.mse_loss(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


#Hyperparameter: batch_size, seq_len, max_epochs
class LSTM_Results:
    def __init__(self, model: pl.LightningModule, name, batch_size, seq_len, max_epochs):
        self.model = model
        self.name = name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.max_epochs = max_epochs

        self.scaler = StandardScaler()
        
        self.forecasts: dict[int, list[float]] = {h: [] for h in FORECAST_HORIZONS}
        self.squared_errors: dict[int, list[float]] = {h: [] for h in FORECAST_HORIZONS}
        self.mse: dict[int, float] = {}

    def train(self, X_train: pd.DataFrame):
        X_train_scaled = self.scaler.fit_transform(X_train.values)

        X_seq, y_seq = [], []
        for t in range(self.seq_len, len(X_train_scaled)):
            X_seq.append(X_train_scaled[t-self.seq_len:t])
            y_seq.append(X_train_scaled[t])

        train_loader = DataLoader(
            TensorDataset(torch.tensor(np.array(X_seq), dtype=torch.float32),
                          torch.tensor(np.array(y_seq), dtype=torch.float32)),
            batch_size=self.batch_size,
            shuffle=True
        )

        trainer = pl.Trainer(max_epochs=self.max_epochs)
        trainer.fit(self.model, train_loader)

    def forecast(self, input: np.ndarray, steps: int) -> np.ndarray:
        self.model.eval()
        cur = np.expand_dims(input, axis=0)

        with torch.no_grad():
            for step in range(steps):
                pred = self.model(torch.tensor(cur, dtype=torch.float32)).numpy()
                cur = np.concatenate([cur[:,1:,:], np.expand_dims(pred, axis=1)], axis=1)

        return self.scaler.inverse_transform(pred).flatten()
    
    def test(self, dates_to_test: pd.DatetimeIndex, X: pd.DataFrame, training_set: bool = False):
        self.forecasts = {h: [] for h in FORECAST_HORIZONS}
        self.squared_errors = {h: [] for h in FORECAST_HORIZONS}
        self.mse = {}

        X_scaled = self.scaler.transform(X.values)

        for horizon in FORECAST_HORIZONS:
            if not horizon in list(self.forecasts.keys()):
                logger.error(f"Forecast horizon {horizon} not found.")
                raise KeyError(f"Forecast horizon {horizon} not found.")

            for date in dates_to_test:
                if not date in X.index:
                    logger.error(f"Date {date} not found in backtesting data")
                    raise ValueError(f"Date {date} not found in backtesting data")

                date_index = X.index.get_loc(date)
                end_index = date_index-horizon
                start_index = end_index - self.seq_len + 1
                if start_index < 0:
                    logger.warning(f"Not enough data to backtest for date {date} - horizon {horizon}")
                    continue
                
                input = X_scaled[start_index:end_index+1]
                prediction = self.forecast(input, steps=horizon)
                actual = X.loc[date].values
        
                self.forecasts[horizon].append(prediction)
                self.squared_errors[horizon].append((actual - prediction).flatten()**2)
                
            self.mse[horizon] = np.mean(self.squared_errors[horizon])
            logger.debug(f"MSE of {self.name} for horizon {horizon}: {self.mse[horizon]}.")

        df_mse = pd.DataFrame(self.mse.items(), columns=['h', self.name])
        out_name = f"{self.name}_mse.csv" if not training_set else f"{self.name}_training_mse.csv"
        save_csv(df_mse, res_dir / out_name)

        return df_mse


def main():
    print(f"Starting training and testing LSTM models at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    logger.info(f"Loading data...")
    train_data = load_transformed_data(PROCESSED_DATA_DIR / "uncentered_train_data.csv")
    test_data = load_transformed_data(PROCESSED_DATA_DIR / "uncentered_test_data.csv")
    full_data = load_transformed_data(PROCESSED_DATA_DIR / "uncentered_full_data.csv")
    k_vars = train_data.shape[1]
    model_names = ["VanillaLSTM", "ResidualLSTM"]

    for model_name in model_names:
        logger.info(f"Model: {model_name}")

        logger.info(f"Loading hyperparameters...")
        with open(res_dir / f"best_hyperparameters_{model_name}.json") as f:
            params = json.load(f)
            logger.info(f"Hyperparameters: {params}")

        if model_name == "VanillaLSTM":
            model_class = VanillaLSTM
        elif model_name == "ResidualLSTM":
            model_class = ResidualLSTM

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

        logger.info(f"Starting training {model_name}...")
        res.train(train_data)

        logger.info(f"Computing training MSE for {model_name}...")
        dates_to_train = train_data.index
        res.test(dates_to_train, train_data,training_set=True)

        logger.info(f"Starting testing {model_name}...")
        dates_to_test = test_data.index
        res.test(dates_to_test, full_data)

    print(f"Finished training and testing LSTM models at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO,
                        format ='%(asctime)s - %(levelname)s - %(message)s',
                        filename = res_dir / "modeling_lstm.log",
                        filemode = 'w')
    main()