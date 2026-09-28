# =====================================================================
# Financial Engineering Library: Volatility Forecasting & Evaluation
# Author: Francesco Carli
# Description: Downloads historical market data, computes econometric
#              volatility models (GARCH, EWMA), and evaluates forecasting
#              accuracy against an ex-post realized volatility benchmark.
# =====================================================================

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from src.volatility_estimator import (
    EWMA,
    GARCH,
    garman_klass,
    parkinson,
    rogers_satchell,
)

if __name__ == "__main__":
  print("=" * 65)
  print("       REAL MARKET VOLATILITY & PREDICTION EVALUATION       ")
  print("=" * 65)

  # -----------------------------------------------------------------
  # 1. Market Data Acquisition
  # -----------------------------------------------------------------
  ticker_symbol = "DIS"  
  print(f"Downloading 1 year of historical daily data for {ticker_symbol}...")

  data = yf.download(ticker_symbol, period="1y", interval="1d")

  if data.empty:
    print(
        "[ERROR] Data retrieval failed. Verify the ticker symbol or network."
    )
    exit()

  # Extract price series components
  close = data["Close"].squeeze().values
  open_p = data["Open"].squeeze().values
  high = data["High"].squeeze().values
  low = data["Low"].squeeze().values

  total_days = len(close)
  print(f"  [OK] Successfully loaded {total_days} trading days.")

  # -----------------------------------------------------------------
  # 2. Train/Test Split (In-Sample vs Out-of-Sample)
  # -----------------------------------------------------------------
  split_idx = total_days // 2

  train_close = close[:split_idx]
  test_close = close[split_idx:]

  print(f"  Training Period (First 6 months): {len(train_close)} days")
  print(f"  Testing Period  (Last 6 months) : {len(test_close)} days")
  print("-" * 65)

  # Compute logarithmic returns for econometric models
  log_returns = np.diff(np.log(close))
  n_returns = len(log_returns)

  # -----------------------------------------------------------------
  # 3. Volatility Modeling & Benchmark Calculation
  # -----------------------------------------------------------------
  window = 20  # Rolling window size (~1 trading month)

  # GARCH(1,1) dynamic conditional volatility
  garch_sigmas = GARCH(
      omega=0.00001,
      alpha=0.05,
      lambda_=0.90,
      initial_sigma=0.20,
      R_i=log_returns,
  )

  # Exponentially Weighted Moving Average (EWMA) volatility
  ewma_series = np.zeros(n_returns)
  for i in range(window, n_returns):
    ewma_series[i] = EWMA(lambda_=0.94, R_i=log_returns[i - window : i])

  # Ex-post Realized Volatility (Rolling standard deviation benchmark)
  realized_vol = np.zeros(n_returns)
  for i in range(window, n_returns):
    realized_vol[i] = np.std(log_returns[i - window : i]) * np.sqrt(252)

  print("  [OK] Volatility estimators and realized benchmark computed.")
  print("-" * 65)

  # -----------------------------------------------------------------
  # 4. Model Evaluation (Out-of-Sample RMSE)
  # -----------------------------------------------------------------
  test_garch_annual = garch_sigmas[split_idx:] * np.sqrt(252)
  test_realized = realized_vol[split_idx:]

  # Root Mean Squared Error between GARCH forecast and actual realized volatility
  rmse_garch = np.sqrt(np.mean((test_garch_annual - test_realized) ** 2))
  print(
      f"  [EVALUATION] GARCH vs Realized Volatility RMSE (Test Phase):"
      f" {rmse_garch:.4f}"
  )
  print("-" * 65)

  # -----------------------------------------------------------------
  # 5. Visualization Dashboard
  # -----------------------------------------------------------------
  fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
  fig.suptitle(
      f"Predictive Volatility Evaluation ({ticker_symbol}) - Train vs Test",
      fontsize=14,
      fontweight="bold",
  )

  # Subplot 1: Asset Price Path & Split Boundary
  days_axis = np.arange(total_days)
  axs[0].plot(
      days_axis[:split_idx],
      train_close,
      color="royalblue",
      lw=1.5,
      label="Train Price",
  )
  axs[0].plot(
      days_axis[split_idx:],
      test_close,
      color="darkorange",
      lw=1.5,
      label="Test Price (Future)",
  )
  axs[0].axvline(
      x=split_idx,
      color="red",
      linestyle="--",
      alpha=0.7,
      label="Train/Test Split",
  )
  axs[0].set_title(f"{ticker_symbol} Stock Price Path")
  axs[0].set_ylabel("Price ($)")
  axs[0].set_xlabel("Trading Days")
  axs[0].grid(True, alpha=0.3)
  axs[0].legend(loc="upper left")

  # Subplot 2: Volatility Forecasts vs Realized Benchmark
  returns_axis = np.arange(1, total_days)

  axs[1].plot(
      returns_axis[window:],
      realized_vol[window:],
      color="black",
      lw=1.8,
      linestyle="--",
      label="Realized Volatility (Benchmark Ex-Post)",
  )
  axs[1].plot(
      returns_axis[window:],
      garch_sigmas[window:] * np.sqrt(252),
      color="crimson",
      lw=1.5,
      label="GARCH(1,1) Forecast",
  )
  axs[1].plot(
      returns_axis[window:],
      ewma_series[window:] * np.sqrt(252),
      color="blue",
      lw=1.2,
      label="EWMA Forecast",
  )

  axs[1].axvline(
      x=split_idx,
      color="red",
      linestyle="--",
      alpha=0.7,
      label="Prediction Start (Test Phase)",
  )

  axs[1].set_title(
      f"Volatility Models vs Realized Volatility (Test RMSE: {rmse_garch:.4f})"
  )
  axs[1].set_xlabel("Trading Days")
  axs[1].set_ylabel("Annualized Volatility")
  axs[1].grid(True, alpha=0.3)
  axs[1].legend(loc="upper right")

  plt.tight_layout()
  plt.show()