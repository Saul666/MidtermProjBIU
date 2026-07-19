# Data Science Stock Correlation Pair Trading Strategy

## Team Members
* **Idan Gilad**
* **Netanel Farhi**

---

## 1. Project Goal
The primary goal of this project is to develop, backtest, and evaluate a two-stage automated algorithmic trading framework that infuses classical Statistical Arbitrage with supervised machine learning. While the system leverages cointegration to identify asset pairs in long-term equilibrium and detect price divergences, it crucially implements a downstream Gradient Boosting classification layer. Enriched with regime-tracking market features, this ML layer acts as an predictive risk filter—identifying and suppressing catastrophic structural breaks to optimize risk-adjusted alpha and portfolio drawdown profiles.

## 2. Business Problem & Market Significance
In highly volatile equity markets, traditional long-only investment strategies expose portfolios to significant systematic market risk (beta). When macroeconomic shocks occur, broad market declines can severely impact asset returns.

### Why This Matters
* **Market-Neutral Returns:** Pair trading is designed to be market-neutral. By simultaneously going long on an undervalued stock and short on an overvalued stock within a cointegrated pair, the strategy isolates idiosyncratic asset mispricings while hedging against broad market downturns.
* **Institutional Asset Management:** Large-scale funds use these statistical arbitrage models to generate consistent returns across bull and bear markets, providing essential liquidity and improving absolute risk-adjusted performance (Sharpe and Sortino ratios).
* **Automated Alpha Generation:** Automating the detection and execution of divergences removes emotional bias from trading, exploiting brief market inefficiencies that human traders cannot track in real-time.

---

## 3. Data Description & Sources
The project utilizes historical daily equity data spanning multiple years to ensure robust statistical significance across various market cycles.

* **Source:** Yahoo Finance API (`yfinance`)
* **Data Features:** * `Date`: Timestamp of trading days.
    * `Adjusted Close`: Adjusted for corporate actions like splits and dividends (used as the primary price signal).
    * `Volume`: Daily traded volume to filter for liquidity.
* **Asset Universe:** High-liquidity equities and sector-specific ETFs to minimize execution slippage and ensure borrowability for short positions.

---

## 4. Approach & Methodology
The strategy is rooted in quantitative finance and statistical modeling, broken down into key foundational pillars:

1.  **Stationarity & Cointegration Testing:** Rather than simple correlation, the framework uses the **Augmented Dickey-Fuller (ADF) test** and the **Johansen test** to identify asset pairs whose price spread is stationary over time, indicating a reliable mean-reverting relationship ($Y_t - \beta X_t = \epsilon_t$).
2.  **Spread Modeling:** Once a cointegrated pair is established, the historical spread is modeled using rolling ordinary least squares (OLS) regression to dynamically calculate the hedge ratio ($\beta$).
3.  **Signal Generation (Z-Score):** The spread is normalized into a dynamic Z-score:
    $$\text{Z-score} = \frac{\text{Spread} - \text{Rolling Mean}}{\text{Rolling Standard Deviation}}$$
4.  **Trading Execution Rules:**
    * **Entry Long Spread:** Triggered when the Z-score crosses below a predefined negative threshold (e.g., $-2.0$), buying Asset Y and shorting Asset X.
    * **Entry Short Spread:** Triggered when the Z-score crosses above a positive threshold (e.g., $+2.0$), shorting Asset Y and buying Asset X.
    * **Exit / Convergence:** Positions are fully liquidated when the Z-score reverts to the mean ($0.0$), or hits a predefined stop-loss boundary.

---

## 5. Work Process & Modules Tested
The pipeline was developed modularly to isolate data processing, mathematical verification, and backtesting performance.

### Modules Tested
* `Data Acquisition & Cleaning Module`: Fetches multi-year tickers, checks for missing data, handles stock splits, and aligns timestamps across different assets.
* `Statistical Cointegration Engine`: Iterates through an $N \times N$ matrix of selected assets, running pairwise ADF and Engle-Granger tests to filter out pseudo-correlated assets.
* `Signal Generation & Thresholding Optimizer`: Tests different rolling lookback windows (e.g., 20-day vs. 60-day moving averages) and varying Z-score entry/exit thresholds to find optimal balances.
* `Backtesting Engine & Metrics Calculator`: Simulates execution over historical unseen test windows, incorporating basic transaction costs, and computes final performance metrics.

---

## 6. Primary Results & Conclusions

### Primary Results
* **Cointegration Viability:** Out of a tested universe of 50 sector equities, the Cointegration Engine successfully isolated 12 statistically significant pairs ($p < 0.05$).
* **Backtest Performance:** Optimized pairs demonstrated highly stable equity curves during periods of broad market corrections, validating the market-neutral objective. 
* **Drawdown Profile:** The strategy effectively capped maximum drawdowns compared to a basic Buy-and-Hold benchmark of the S&P 500 index, though returns were flatter during strong macro bull runs.

* **Best Model Performance:** The Gradient Boosting Classifier (enriched with macro and regime features) achieved the highest scores, delivering a Precision of 74% on predicting successful trade setups and an F1-Score of 0.68 on capturing structural breaks.

**ROI and A/B testing:**
The Naive Strategy (Stage 1 Only) achieved a Sharpe Ratio of 1.12 but suffered a maximum drawdown of -18.4% due to three unhedged structural breaks.

The ML-Filtered Strategy (Stage 2 Upgraded) improved the Sharpe Ratio to 1.65 and successfully mitigated risk, dropping the maximum drawdown to just -6.2%.

* The Sharpe ratio measures an investment's return relative to its risk. It is calculated by subtracting the risk-free rate from the portfolio's return, then dividing by the standard deviation (volatility) of those returns. A ratio above 1.0 is generally considered good, while above 2.0 is excellent

### Conclusions
Statistical arbitrage via pair trading remains a robust tool for downside protection. However, the strategy is highly sensitive to the **rolling lookback window** and **transaction fees**. Over-optimizing thresholds on historical data can lead to overfitting, meaning continuous walk-forward optimization is required for live implementation.

---

## 7. Repository Structure

```text
├── data/                  # Cached raw and cleaned CSV price data
├── notebooks/             # Step-by-step exploratory analysis
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_model_evaluation.ipynb
|   └── 05_regime_tracking.ipynbb
|   └── 06_model_improvement.ipynb
|   └── 07_feature_enrichment.ipynb
├── src/                   # Production-ready modular source code
├── requirements.txt       # requirement how to setup the project
└── README.md              # Project overview and documentation
```
