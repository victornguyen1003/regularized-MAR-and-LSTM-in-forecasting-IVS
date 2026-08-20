# Regularized MAR and Residual LSTM in Forecasting Implied Volatility Surfaces

## Structure

---

marp: true
----------

theme: gaia

paginate: true

style: |

  secton.table-slide

  section.table-slide .table-middle

  section.table-slide .table-middle table

  section.table-slide .table-footnote

  section.image-slide

  section.image-slide h2

  section.image-slide .image-middle

  section.image-slide .image-middle p

  section.image-slide .image-middle img

  section.bs-slide .bs-legend

  section.bs-slide .bs-legend > div

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
