import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, AMBER, GREY = "#12284C", "#F2A900", "#8A94A6"


def classical_decompose(s: pd.Series, period: int = 24 * 7) -> pd.DataFrame:
    """Additive classical decomposition: trend (centred moving average),
    seasonal (mean of detrended values per position in the cycle), residual."""
    trend = s.rolling(window=period, center=True, min_periods=period // 2).mean()
    detrended = s - trend
    pos = np.arange(len(s)) % period
    seasonal_means = detrended.groupby(pos).mean()
    seasonal = pd.Series(seasonal_means.values[pos], index=s.index)
    resid = s - trend - seasonal
    return pd.DataFrame({"observed": s, "trend": trend, "seasonal": seasonal, "residual": resid})


def autocorrelation(s: pd.Series, max_lag: int = 24 * 8) -> np.ndarray:
    x = s.values - s.values.mean()
    denom = np.dot(x, x)
    return np.array([np.dot(x[: len(x) - k], x[k:]) / denom for k in range(max_lag + 1)])


def plot_overview(s: pd.Series, out):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(s.index, s.values, color=NAVY, lw=0.4)
    ax.set_title("Hourly electricity demand (full history)")
    ax.set_ylabel("Demand (MW)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, AMBER, GREY = "#12284C", "#F2A900", "#8A94A6"


def classical_decompose(s: pd.Series, period: int = 24 * 7) -> pd.DataFrame:
    """Additive classical decomposition: trend (centred moving average),
    seasonal (mean of detrended values per position in the cycle), residual."""
    trend = s.rolling(window=period, center=True, min_periods=period // 2).mean()
    detrended = s - trend
    pos = np.arange(len(s)) % period
    seasonal_means = detrended.groupby(pos).mean()
    seasonal = pd.Series(seasonal_means.values[pos], index=s.index)
    resid = s - trend - seasonal
    return pd.DataFrame({"observed": s, "trend": trend, "seasonal": seasonal, "residual": resid})


def autocorrelation(s: pd.Series, max_lag: int = 24 * 8) -> np.ndarray:
    x = s.values - s.values.mean()
    denom = np.dot(x, x)
    return np.array([np.dot(x[: len(x) - k], x[k:]) / denom for k in range(max_lag + 1)])


def plot_overview(s: pd.Series, out):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(s.index, s.values, color=NAVY, lw=0.4)
    ax.set_title("Hourly electricity demand (full history)")
    ax.set_ylabel("Demand (MW)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)