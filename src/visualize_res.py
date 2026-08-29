import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

import logging
logger = logging.getLogger(__name__)

from util import load_transformed_data
from config import FIG_DIR, PROCESSED_DATA_DIR, RES_DIR  


def visualize_mse(is_train: bool):
    mse_paths = ["VAR/VAR_mse.csv", 
                 "MAR/MAR(1)_mse.csv", 
                 "MAR/MAR(1)_training_mse.csv",
                 "MAR/regularized_MAR_mse.csv", 
                 "MAR/regularized_MAR_training_mse.csv",
                 "LSTM/VanillaLSTM_mse.csv",
                 "LSTM/VanillaLSTM_training_mse.csv",
                 "LSTM/ResidualLSTM_mse.csv", 
                 "LSTM/ResidualLSTM_training_mse.csv",]

    if is_train:
        mse_dfs = [pd.read_csv(f"{RES_DIR}/{path}") for path in mse_paths if "training" in path]
        fname = "train_mse"
        data = load_transformed_data(PROCESSED_DATA_DIR / "uncentered_train_data.csv")
    else:
        mse_dfs = [pd.read_csv(f"{RES_DIR}/{path}") for path in mse_paths if not "training" in path]
        fname = "test_mse"
        data = load_transformed_data(PROCESSED_DATA_DIR / "uncentered_test_data.csv")

    base = mse_dfs[0]
    for df in mse_dfs[1:]:
        base = base.merge(df, how="outer", on='h')
    base.to_csv(f"{RES_DIR}/{fname}.csv", index=False)
    logger.info(f"Successfully saved to {RES_DIR}/{fname}.csv")

    base = base.sort_values('h').set_index('h')
    variance = np.var(data.values)

    plt.figure()
    ax = base.reset_index(drop=True).plot(kind="line")
    ax.set_xticks(range(len(base.index)))
    ax.set_xticklabels(base.index)
    ax.set_xlabel('h')
    ax.axhline(variance, color='black', linestyle='--', label='Sample variance')
    ax.legend()
    plt.ylim(0,variance*1.5)
    plt.title(f"{'Training' if is_train else 'Testing'} MSE vs {'Training' if is_train else 'Testing'} Sample Variance")
    plt.savefig(f"{FIG_DIR}/{fname}.png")
    plt.close()
    logger.info(f"Successfully saved to {FIG_DIR}/{fname}.png")


def visualize_mar_coef():
    fnames = ["A_lse",
             "B_lse",
             "regularized_A",
             "regularized_B"]

    for fname in fnames:
        path = RES_DIR / "MAR" / f"{fname}.csv"
        df = pd.read_csv(path, header=None)
        plt.figure()
        sns.heatmap(df)
        plt.title(f"{fname} coefficient matrix")
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"{FIG_DIR}/MAR_{fname}.png")
        plt.close()
        logger.info(f"Successfully saved to {FIG_DIR}/MAR_{fname}.png")


def visualize_eigenvalues():
    df = pd.read_csv(RES_DIR / "MAR/MAR_eigenvalues.csv")
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip('()').astype(complex).abs()

    plt.figure()
    ax = sns.stripplot(data=df, jitter=True, alpha=0.7)
    plt.title("Eigenvalues of MAR coefficient matrices")
    plt.savefig(f"{FIG_DIR}/MAR_eigenvalues.png")
    plt.close()
    logger.info(f"Successfully saved to {FIG_DIR}/MAR_eigenvalues.png")


def main():
    logger.info("Visualizing training and testing MSE...")
    visualize_mse(is_train = False)
    visualize_mse(is_train = True)

    logger.info("Visualizing coeffient matrices of MAR models")
    visualize_mar_coef()

    logger.info("Visualizing eigenvalues of MAR models")
    visualize_eigenvalues()


    



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                        format="%(name)s - %(levelname)s - %(message)s",)
    main()

