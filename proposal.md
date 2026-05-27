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
# Regularized MAR and LSTM
# in forecasting IV surfaces

---

<!-- _class: table-slide -->

## Trading and Big Money

<div class="table-middle">

| Company | Headcount | Compensation |
| --- | --- | --- |
| Jane Street | 3K | Mean 2.68M |
| Google | 190K | Median 331K |

</div>

<p class="table-footnote">
  Source: Yahoo Finance
</p>

---

## Option Pricing Over(simplified)view

- **Equity option**: equity derivative that gives the owner the right, but not the obligation, to execute a trade.

- **Two types**: call & put options 
  - Strike price
  - Expiration day 
  - Premium

- **Implied Volatility (IV)**: obtained by inverting the Black-Scholes model *(Black & Scholes, 1973)*

---
<!-- _class: image-slide -->

## Implied Volatility (IV) Surface
<br>

<div class="image-middle">

![w:650 h:500](https://nag.com/wp-content/uploads/2023/06/example-implied-volatility-surface.png)

</div>

---

## IV Estimation Issues

- **Deep learning:** LSTM (Hochreiter, Schmidhuber, 1997) - highly accurate, but...
  - considered as "black-box"
  &rarr; **MODEL RISK** = **FINANCIAL LOSS**

- **Econometrics:** VAR (Sims, 1980) - highly interpretable, but...
  - fail to preserve matrix structure
  - have large # of coefficients

- **Fortunately, MAR Lasso** *(Chen et al., 2020)* has the best of both worlds
---

## MAR Lasso vs. LSTM: Lack of Literature

- No comparison:
  - Regularized MAR (Jiang, Shen, Li, Gao, 2024)
  - MAR extension - SIGMAR (Wu, Yang, Yan, Feng, 2025)
- No MAR Lasso:
  - ConvLSTM/LSTM/VAR/VEC (Medvedev, 2019)
  - LSTM/GARCH/GJRGARCH (Chatterjee, Bhowmick, Sen, 2022)
- No IV:
  - LSTM/MAR/VAR (Liu, Fang, 2026)

---
## Purpose
- Compare LSTM and MAR Lasso in Forecasting IV Surfaces
- Research questions:
  1. Does MAR Lasso outperform LSTM in predicting IV surfaces for a 1-day, 30-day, and 90-day horizon?
  2. Does MAR Lasso significantly improve upon unconstrained MAR and VAR in predicting IV surfaces?

---

## Methodology (1): Why not VAR? Too bad!

VAR: Vector Autoregressive Matrix 

$$
\text{vec}(X_t) = \Phi\text{vec}(X_{t-1}) + \text{vec}(E_t)
$$

*where $X_t \in \mathbb{R}^{m \times n}$ represents the IV surface at time $t$, and $vec(.)$ is the vectorization of a matrix by stacking its columns.*

- Fail to capture relationship between rows and columns
- Number of parameters: $O(m^2n^2)$

---

## Methodology (2): MAR? Better!

MAR: Matrix Autoregressive Model

$$
X_t = A X_{t-1} B^\top + E_t
$$

*where $\otimes$ denotes the Kronecker product.*

- Number of parameters: $O(m^2+n^2)$

---

## Methodology (3): MAR Lasso? Much Better! Maybe...

- Lasso: regularization technique
- MAR Lasso *(Jiang, Shen, Li, Gao, 2024)*: further reduces # of parameters 

---

## Methodology (4): First lag is enough!

- IV, obtained by inverting the Black-Scholes formula, is just price
- The Weak Efficient Market Hypothesis *(Eugene Fama, 1970)*: today’s stock prices reflect all the data of past prices
- The First Fundamental Theorem of Asset Pricing *(Harrison, Kreps, 1979)*: no arbitrage iff EMM exists under which the discounted price of all traded assets are martingales.
---

<!-- _class: image-slide -->

## Methodology (5): LSTM

<div class="image-middle">

![w:1000 h:500](https://media.geeksforgeeks.org/wp-content/uploads/20250528172141936296/architecture_of_lstms.webp)

</div>

---

## Research Design
- **Data collection**:
  - Source: Bloomberg
  - Time period: Apr 2016 - May 2026
- **Timeline**:
  - Phase 1: formalize math + acquire data
  - Phase 2: feature engineer/build/train/fine-tune models
  - Phase 3: test/evaluate/compare models
  - Phase 4: document
  - Phase 5: validate and revise

--- 

## Expected Outcomes

- Anticipated findings:
  - LSTM outperforms MAR Lasso
  - MAR Lasso outperforms MAR/VAR
- Potential limitations:
  - "Black box" of deep learning
- Significance:
  - Academic: keep searching for econometric models
  - Practical: tolerate and quantify risks







