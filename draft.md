# Regularized MAR and Residual LSTM in Forecasting Implied Volatility Surfaces

## Structure

---
marp: true
theme: gaia
paginate: true
style: |
  section.table-slide {
    display: flex;
    flex-direction: column;
  }
  section.table-slide .table-middle {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  section.table-slide .table-middle table {
    margin: 0;
  }
  section.table-slide .table-footnote {
    flex-shrink: 0;
    margin-top: 0.5em;
    font-size: 0.8em;
    text-align: left;
    opacity: 0.85;
  }
  section.image-slide {
    display: flex;
    flex-direction: column;
  }
  section.image-slide h2 {
    flex-shrink: 0;
    margin-bottom: 0;
  }
  section.image-slide .image-middle {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    min-height: 0;
  }
  section.image-slide .image-middle p {
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
  }
  section.image-slide .image-middle img {
    display: block;
    margin: 0 auto;
  }
  section.bs-slide .bs-legend {
    display: flex;
    justify-content: space-evenly;
    gap: 0.5rem;
    padding: 0 2rem;
  }
  section.bs-slide .bs-legend > div {
    flex: 0 1 auto;
    min-width: 8em;
  }
---
<!-- _class: lead -->

# Regularized MAR and LSTM in forecasting IV surfaces

---

### References

Jiang, H., Shen, B., Li, Y., & Gao, Z. (2024). Regularized estimation of high-dimensional matrix-variate autoregressive models

Chen, R., Xiao, H., & Yang, D. (2018). Autoregressive models for matrix-valued time series

Li, F.-F., Johnson, J., & Yeung, S. (2017). Lecture 10: Recurrent Neural Networks. CS231n: Convolutional Neural Networks for Visual Recognition (Spring 2017), Stanford University

---

### Options Pricing Overview

- Option: a financial instrument that gives the owner the right, but not the obligation, to buy/sell an underlying asset at a strike price

  - Strike price
  - Time to maturity
- Two types: call options (right to buy) and put options (right to sell)
- European-style options: can only be exercised at expiration

E.g. A European-style, $90, 30-day expiration, call option gives you the right to buy the underlying stock at $90 regardless of its market price at expiration

---

<!-- _class: bs-slide -->

### Black-Scholes's Model

$$
C = S_0 N(d_1) - K e^{-rT} N(d_2)
$$

$$
P = K e^{-rT} N(-d_2) - S_0N(-d_1)
$$

where

$$
d_1 = \frac{\ln(S_0/K) + (r + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}},
\qquad
d_2 = d_1 - \sigma\sqrt{T}
$$

and

<div class="bs-legend">

```text
├── config/
│   └── environment.yml        # Conda environment dependencies
├── data/raw/                  # Original, untouched yearly datasets
├── demo/                      # Demo code and report
│   ├── pics/                  # Figures 
│   ├── demo.ipynb             # Prototype script
│   └── report.md              # Findings
├── notebooks                  # Notebooks for EDA and modeling
├── src/                       # Production-level code
├── reports/                   # Figures and finding reports
└── README.md                  # Project overview
```

## Run locally

Prerequisites: Conda

1. Create conda environment:
   ```bash
   conda env create -p ./.env -f config/environment.yml
   ```
2. Activate environment:
   ```bash
   conda activate ./.env
   ```

## References

* Jiang, H., Shen, B., Li, Y., & Gao, Z. (2024). Regularized estimation of high-dimensional matrix-variate autoregressive models
* Chen, R., Xiao, H., & Yang, D. (2018). Autoregressive models for matrix-valued time series
* Li, F.-F., Johnson, J., & Yeung, S. (2017). Lecture 10: Recurrent Neural Networks. CS231n: Convolutional Neural Networks for Visual Recognition (Spring 2017), Stanford University
