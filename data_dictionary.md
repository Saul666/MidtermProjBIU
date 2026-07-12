# Data Dictionary: Stock Correlation Pair Strategy Project

All raw market data is programmatically sourced from the **Yahoo Finance (YFinance) API**.

---

## Main Table downloaded from YFinance : `historical_prices`
This dataset contains the daily historical market activity fetched directly from Yahoo Finance for the configured ticker universe.

### Column Definitions

| Column Name | Significance | Data Type | Possible / Valid Values |
| :--- | :--- | :--- | :--- |
| `Date` | The calendar date of the trading session. Serves as the primary index across all assets. | `Datetime64[ns]` | ISO format `YYYY-MM-DD` (Excludes weekends and market holidays). |
| `Ticker` | The unique stock symbol or ETF identifier assigned to the asset. | `String` / `Categorical` | Examples: `'AAPL'`, `'MSFT'`, `'XLF'`, `'SPY'`. |
| `Open` | The price at which the stock first traded when the market opened for the day. | `Float64` | Positive values ($> 0.0$). |
| `High` | The highest price the stock reached during that specific trading day. | `Float64` | Positive values ($> 0.0$). Must be $\ge$ `Open`, `Low`, and `Close`. |
| `Low` | The lowest price the stock dropped to during that specific trading day. | `Float64` | Positive values ($> 0.0$). Must be $\le$ `Open`, `High`, and `Close`. |
| `Close` | The raw closing price of the stock at the end of regular trading hours. | `Float64` | Positive values ($> 0.0$). |
| `Adj Close` | The adjusted closing price reflecting all corporate actions, including stock splits and dividend distributions. | `Float64` | Positive values ($> 0.0$). Used as the definitive asset price for quantitative evaluation. |
| `Volume` | The total number of shares or contracts traded during the day. | `Int64` | Non-negative integers ($\ge 0$). |

### Data Cleaning & Transformations
Before moving into the statistical modeling stage, the raw data undergoes the following structural steps:
1. **Ticker Alignment & Missing Value Dropping:** Yahoo Finance occasionally returns missing rows (`NaN`) for assets during corporate adjustments or half-day suspensions. Rows containing missing data in critical fields (`Adj Close`) are dropped or forward-filled if the suspension is brief.
2. **Date Index Standardization:** Indices across different assets are inner-joined based on the `Date` column to eliminate anomalies where one ticker has data on a specific day but another does not (e.g., dual-listings or region-specific closures).

---
