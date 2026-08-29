# Regularized MAR and Residual LSTM in Forecasting Implied Volatility Surfaces

This project compares the predictive performance of deep-learning architectures LSTM and LSTM with residual connections against econometric baselines Vetor Autoregressive (VAR), Matrix Autoregressive (MAR), and MAR with ElasticNet regularization in forecasting Implied Volatility (IV) Surfaces.

## Project structure

```text
.
├── README.md
├── config/
│   └── environment.yml
├── data/
│   ├── raw/
│   ├── processed/
│   └── result/
├── asset/
│   ├── figure/
│   ├── reference/
│   └── report.typ
├── notebook/
└── src/
```


## How to run locally

### 1) Set up a Conda environment

```bash
conda env create -f config/environment.yml -l ./.env
conda activate ./.env
```

### 2) Run the full pipeline

```bash
python src/run.py
```