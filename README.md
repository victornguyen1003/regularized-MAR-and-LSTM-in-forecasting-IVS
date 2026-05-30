# Regularized MAR and LSTM in Forecasting IV Surfaces

## Abstract
This project investigates the forecasting the Implied Volatility (IV) surface for SPX options by comparing traditional Vector Autoregressive (VAR) models with modern Matrix Autoregressive (MAR) models, Regularized MAR (LASSO), and Long Short-term Memory (LSTM). For a comprehensive breakdown of the methodology, please refer to the report.pdf document.  


## Data Collection
* **Source:** Bloomberg.  
* **Time Period:** 4/1/2016 to 5/25/2026.  
* **Matrix Structure:** Each IV surface is represented as a 7x9 matrix where Moneyness (S/K) ranges from 80% to 120%, and Time to Maturity ranges from 1M to 24M.  

## Methodology
* **VAR vs. MAR:** The baseline VAR(1) and VAR(15) models struggle to capture structural relationships between rows and columns while requiring a massive parameter space of $O(m^2n^2)$. The MAR(1) model preserves the natural $m \times n$ matrix structure, reducing the parameter space to $O(m^2+n^2)$ through the equation $X_t = A X_{t-1} B^\top + E_t$.  
* **MAR Estimation:** Optimal parameters are found utilizing Projection (PROJ) via Singular Value Decomposition and Iterated Least Squares (ILS).  
* **Regularized MAR (LASSO):** L1 penalties are applied to further reduce the number of parameters and prevent predicton explosions for long-horizon forecasts of MAR due to the ILS method, which easily yields an eigenvalue greater than 1.
* **LSTM:** Both Vanilla and Residual LSTM architectures are tested where the Residual LSTM predicts the daily change $\Delta_t = y_t - y_{t-1}$ rather than raw levels as IV has shown to follow a random walk. Dropout and weight decay are implemented while the hidden state dimension is reduced to prevent overfitting, with an increase in the number of epochs to offset the effect on learning.

## Results and Performance
* **Random Walk:** VAR(15) showed the worst performance due to overfitting, which suggests IV is a martingale
* **Short-Term vs. Long-Term:** MAR(1) slightly outperformed VAR(1) for 1-day and 5-day horizons, but experienced exponentially exploding MSFE on longer horizons.
* **Penalty Trade-offs:** Regularized MAR (LASSO) with smaller penalties improved short-term forecasts but failed to avoid long-term explosion. Conversely, larger penalties successfully shrank the spectral radius to mitigate long-term explosion, but caused underfitting in the short term[cite: 1, 2].  
* **Best Overall Model:** Residual LSTM achieved similar short-term results to the lightly penalized MAR LASSO but remained significantly more stable across extended multi-day horizons.

### References

* Jiang, H., Shen, B., Li, Y., & Gao, Z. (2024). Regularized estimation of high-dimensional matrix-variate autoregressive models
* Chen, R., Xiao, H., & Yang, D. (2018). Autoregressive models for matrix-valued time series
* Li, F.-F., Johnson, J., & Yeung, S. (2017). Lecture 10: Recurrent Neural Networks. CS231n: Convolutional Neural Networks for Visual Recognition (Spring 2017), Stanford University