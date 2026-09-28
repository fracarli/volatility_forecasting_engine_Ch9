# Quantitative Finance: Volatility Forecasting & Econometric Evaluation

A Python-based framework for downloading historical market data, computing advanced econometric volatility models (GARCH, EWMA, ARCH), and evaluating forecasting accuracy against an ex-post realized volatility benchmark. 

Inspired by the quantitative finance concepts and volatility modeling principles found in literature (such as Paul Wilmott's works, Chapter 9), this repository serves as a practical sandbox to understand time-varying conditional variance and volatility forecasting.

---

## Simulation Preview

*(Example output illustrating a 1-year stock price path with a Train/Test split, accompanied by a comparison dashboard mapping GARCH and EWMA forecasts against the real ex-post volatility benchmark).*

---

## Theoretical Background & Chapter 9 Summary (Paul Wilmott)

Modeling asset volatility accurately is critical for derivatives pricing, risk management, and hedging. This project translates key econometric concepts into code:

### 1. Conditional Heteroskedasticity & ARCH/GARCH Models
Financial time series exhibit volatility clustering—large changes tend to be followed by large changes. 
* **ARCH & GARCH(1,1):** Unlike constant historical volatility, the Generalized Autoregressive Conditional Heteroskedasticity model dynamically updates variance based on past squared shocks and previous variance states ($\omega + \alpha \sigma_{t-1}^2 + \lambda \epsilon_{t-1}^2$).
* **EWMA (Exponentially Weighted Moving Average):** Assigns exponentially decaying weights to past squared returns, placing higher emphasis on recent market movements.

### 2. Advanced Range-Based Estimators
Using only closing prices discards valuable intraday information. This library implements advanced estimators that leverage high and low bounds:
* **Parkinson Estimator:** Captures intraday range efficiency using High and Low prices.
* **Garman-Klass Estimator:** Incorporates Open, High, Low, and Close prices for a more accurate volatility proxy.
* **Rogers-Satchell Estimator:** Accounts for non-zero drift or trend in asset prices.

### 3. Out-of-Sample Evaluation (RMSE)
To rigorously test model validity, the pipeline performs an **In-Sample vs. Out-of-Sample split** (Train/Test). The predictive accuracy of the GARCH forecast is measured against the ex-post realized volatility benchmark using the **Root Mean Squared Error (RMSE)**.

---

## Repository Structure

```text
.
├── main.py                  # Execution script: data acquisition, backtesting, RMSE evaluation, and dashboard
└── src/
    └── volatility_estimator.py # Econometric models library (GARCH, EWMA, Parkinson, Garman-Klass, etc.)