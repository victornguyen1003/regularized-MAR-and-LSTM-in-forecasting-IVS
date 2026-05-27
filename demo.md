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

## Options Pricing

- **Equity option**: equity derivative that gives the owner the right, but not the obligation, to execute a trade.

- **Two types**: call & put options 
  - Strike price
  - Expiration day 
  - Premium

---

## Black Scholes Model

- Assumptions:
1. The underlying stock does not pay a dividend and never will.
2. The option must be European-style.
3. Financial markets are efficient.
4. No commissions are charged on the trade.
5. Interest rates remain constant.
6. The underlying stock returns are log-normally distributed.

---

<!-- _class: bs-slide -->

## Black-Scholes's model

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

<div>
S<sub></sub>: stock price <br>
K: strike price
</div>

<div>
r: risk-free interest rate <br>
T: time to maturity <br>
</div>

<div>
&sigma;: volatility
</div>

</div>

---
<!-- _class: image-slide -->

## Implied Volatility (IV) Surface
<br>

<div class="image-middle">

![w:650 h:500](https://nag.com/wp-content/uploads/2023/06/example-implied-volatility-surface.png)

</div>

---

## Methodology (3): MAR Lasso? Much Better! Maybe...

Lasso: further reduces # of parameters

Rewrite MAR into: 

$$
\text{vec}(Y_t) = ((B Y_{t-1}') \otimes I_{p_1}) \text{vec}(A) + \text{vec}(E_t)
$$
$$
\text{vec}(Y_t') = ((A Y_{t-1}) \otimes I_{p_2}) \text{vec}(B) + \text{vec}(E_t')
$$

---
Denote: 
$$
\begin{aligned}
Z_{t-1} &= (B Y_{t-1}') \otimes I_{p_1}, \quad \widehat{Z}_{t-1} = (\widehat{B} Y_{t-1}') \otimes I_{p_1} \\
Z_{t-1}^* &= (A Y_{t-1}) \otimes I_{p_2}, \quad \widehat{Z}_{t-1}^* = (\widehat{A} Y_{t-1}) \otimes I_{p_2}
\end{aligned}
$$

Optimization problems:
$$
\widehat{\alpha} = \arg\min_{\alpha \in \mathbb{R}^{p_1^2}} \left\{ \frac{1}{T} \sum_{t=2}^T \| \mathbf{y}_t - \widehat{\mathbf{Z}}_{t-1} \alpha \|_2^2 + \lambda_{1,T} \|\alpha\|_1 \right\} \tag{1}
$$

$$
\widehat{\beta} = \arg\min_{\beta \in \mathbb{R}^{p_2^2}} \left\{ \frac{1}{T} \sum_{t=2}^T \| \mathbf{y}_t^* - \widehat{\mathbf{Z}}_{t-1}^* \beta \|_2^2 + \lambda_{2,T} \|\beta\|_1 \right\} \tag{2}
$$

---
Algorithms:
1. Obtain $\widehat{A}_0$ and $\widehat{B}_0$ by the method in Chen et al. (2020), denoted as $\widehat{B}^{(0)} = \widehat{B}_0$ and $\widehat{A}^{(0)} = \widehat{A}_0$, respectively.
2. For the $i$-th iteration ($i = 1, 2, \dots$),
   (a) Fix $B = \widehat{B}^{(i-1)}$, apply Lasso to (1) and obtain $\widehat{A}^{(i)}$.
   (b) Fix $A = \widehat{A}^{(i)}$, apply Lasso to (2) and obtain $\widehat{B}^{(i)}$.
   (c) The iteration stops if the convergence criterion is satisfied, otherwise we go to the next iteration and repeat Steps 2(a)–2(b).

---





