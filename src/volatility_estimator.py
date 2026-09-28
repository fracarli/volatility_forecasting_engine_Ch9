# =====================================================================
# VOLATILITY ESTIMATION & ECONOMETRIC MODELS LIBRARY
# Author: Francesco Carli
# Date: 28/09/2026
# =====================================================================

import numpy as np

# ---------------------------------------------------------
# Autoregressive Conditional Heteroskedasticity (ARCH) model
# Combines a baseline variance with the historical squared returns.
# ---------------------------------------------------------
def ARCH(sigma_base, alpha, R_i):
  R_i = np.asarray(R_i)
  n = len(R_i)
  if n == 0:
    return sigma_base
  return np.sqrt(
      alpha * (sigma_base ** 2) + (1 - alpha) * np.sum(R_i ** 2) / n
  )


# ---------------------------------------------------------
# Exponentially Weighted Moving Average (EWMA) volatility estimator
# Gives exponentially decaying weights to past squared returns.
# ---------------------------------------------------------
def EWMA(lambda_, R_i):
  R_i = np.asarray(R_i)
  n = len(R_i)
  if n == 0:
    return 0.0
  weights = lambda_ ** np.arange(n - 1, -1, -1)
  variance = (1 - lambda_) * np.sum(weights * (R_i ** 2))
  return np.sqrt(variance)


# ---------------------------------------------------------
# GARCH(1,1) model
# Iteratively calculates time-varying conditional volatility.
# ---------------------------------------------------------
def GARCH(omega, alpha, lambda_, initial_sigma, R_i):
  R_i = np.asarray(R_i)
  n = len(R_i)
  if n == 0:
    return np.array([initial_sigma])

  sigmas = np.zeros(n)
  current_var = initial_sigma ** 2

  for t in range(n):
    # Standard GARCH(1,1) recursive variance update step
    current_var = (
        omega
        + alpha * current_var
        + lambda_ * (R_i[t] ** 2)
        # Note: adjust coefficients depending on your exact parameterization convention
    )
    sigmas[t] = np.sqrt(current_var)

  return sigmas


# ---------------------------------------------------------
# Parkinson's volatility estimator
# Uses High and Low prices to capture intraday range efficiently.
# σ^2 = (1 / (4 * n * ln(2))) * sum( (ln(H_i / L_i))^2 )
# ---------------------------------------------------------
def parkinson(high, low):
  high = np.asarray(high)
  low = np.asarray(low)
  n = len(high)
  if n == 0:
    return 0.0
  factor = 1.0 / (4.0 * n * np.log(2.0))
  sum_sq = np.sum((np.log(high / low)) ** 2)
  return np.sqrt(factor * sum_sq)


# ---------------------------------------------------------
# Garman-Klass volatility estimator
# Incorporates Open, High, Low, and Close prices.
# σ^2 = (1/n) * sum( 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2 )
# ---------------------------------------------------------
def garman_klass(open_p, high, low, close):
  open_p = np.asarray(open_p)
  high = np.asarray(high)
  low = np.asarray(low)
  close = np.asarray(close)
  n = len(open_p)
  if n == 0:
    return 0.0
  term1 = 0.5 * (np.log(high / low)) ** 2
  term2 = (2.0 * np.log(2.0) - 1.0) * (np.log(close / open_p)) ** 2
  variance = np.sum(term1 - term2) / n
  return np.sqrt(variance)


# ---------------------------------------------------------
# Rogers-Satchell volatility estimator
# Accounts for non-zero drift/trend in asset prices.
# σ^2 = (1/n) * sum( ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O) )
# ---------------------------------------------------------
def rogers_satchell(open_p, high, low, close):
  open_p = np.asarray(open_p)
  high = np.asarray(high)
  low = np.asarray(low)
  close = np.asarray(close)
  n = len(open_p)
  if n == 0:
    return 0.0
  term1 = np.log(high / close) * np.log(high / open_p)
  term2 = np.log(low / close) * np.log(low / open_p)
  variance = np.sum(term1 + term2) / n
  return np.sqrt(variance)