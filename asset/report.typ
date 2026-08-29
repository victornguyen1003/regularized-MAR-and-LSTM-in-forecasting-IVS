#import "@preview/typslides:1.3.4": *

// Project configuration
#show: typslides.with(
  ratio: "16-9",
  theme: "bluey",
  font: "Fira Sans",
  font-size: 24pt,
  link-style: "color",
  show-progress: true,
)

// The front slide is the first slide of your presentation
#front-slide(
  title: "Regularized MAR vs Residual LSTM in Forecasting IV Surfaces",
  authors: "Victor Nguyen",
  info: [#link("https://github.com/victornguyen1003/regularized-MAR-and-residual-LSTM-in-forecasting-IVS")],
)

// Custom outline
#table-of-contents()

// A simple slide
#slide(title: "Options Pricing Overview", outlined: true)[
- Option: a financial instrument that gives the owner the right, but not the obligation, to buy/sell an underlying asset at a strike price
  - Strike price
  - Time to maturity

- Two types: call options (right to buy) and put options (right to sell)

- European-style options: can only be exercised at expiration
]

#slide(title: "Black-Scholes's Model", outlined: true)[
  $
    C = S_0 N(d_1) - K e^(-r T) N(d_2)\
    P = K e^(-r T) N(-d_2) - S_0N(-d_1)
  $
  where

  $
    d_1 = frac(log (S_0/K) + (r + 1/2 sigma^2) T, sigma sqrt(T)), quad d_2 = d_1 - sigma sqrt(T)
  $
  and 
  $
    &S: "spot price", quad K: "strike price", quad r: "risk-free rate",\
    &T: "time to maturity", quad sigma: "volatility"
  $
]

#slide(title: "Black-Scholes's Model", outlined: true)[
  Assumptions:
  - The underlying stock does not pay a dividend and never will
  - The option must be European-style
  - Financial markets are efficient
  - No commissions are charged on the trade
  - Interest rates remain constant
  - The underlying stock returns are log-normally distributed
]

#slide(title: "Data Collection", outlined: true)[
    #grid(
      columns: 2,
      [*Source*: Bloomberg

      *Time period*: 4/1/2016 - 5/25/2026

      *Moneyness (S/K)*: 80%, 90%, 95%, 97.5%, 100%, 102.5%, 105%, 110%, 120%

      *Time to maturity*: 1M, 2M, 3M, 6M, 12M, 18M, 24M

      _Each IV surface is a 7x9 matrix_],
      [#image("figure/ivs_20260525.png", height: 110%, width: 110%)]
    )
]

#slide(title: "")

// Columns
#slide(title: "Columns")[

  #cols(columns: (2fr, 1fr, 2fr), gutter: 2em)[
    #grayed[Columns can be included using `#cols[...][...]`]
  ][
    #grayed[And this is]
  ][
    #grayed[an example.]
  ]

  - Custom spacing: `#cols(columns: (2fr, 1fr, 2fr), gutter: 2em)[...]`

  // - Sample references: @typst, @typslides.
    - Add a #stress[bibliography slide]...

    1. `#let bib = bibliography("you_bibliography_file.bib")`
    2. `#bibliography-slide(bib)`
]

// Bibliography
// #let bib = bibliography("bibliography.bib")
// #bibliography-slide(bib)