from pathlib import Path

import numpy as np
import pandas as pd
import config

def load_demand(path=None) -> tuple[pd.Series, str]:
    
    if path is None:
        path = config.SYNTHETIC_DEMAND_CSV
        source = "synthetic"
    else:
        source = "real"
    df = pd.read_csv(path)
    time_col, value_col = df.columns[0], df.columns[1]
    df[time_col] = pd.to_datetime(df[time_col])
    s = df.set_index(time_col)[value_col].astype(float).sort_index()
    return clean_series(s), source

def clean_series(s: pd.Series) -> pd.Series:
    s = s[~s.index.duplicated(keep="first")]
    s = s.asfreq("h")                       
    n_missing = int(s.isna().sum())
    s = s.interpolate(method="time", limit=6).ffill().bfill()
    s.attrs["n_missing_filled"] = n_missing
    return s

def _fixed_holidays(index: pd.DatetimeIndex) -> np.ndarray:
    fixed = {(1, 1), (7, 4), (12, 25), (12, 26)}
    return np.array([(d.month, d.day) in fixed for d in index], dtype=int)

def holiday_flag(index: pd.DatetimeIndex, country: str = "US") -> np.ndarray:
    try:
        import holidays
        cal = holidays.country_holidays(country, years=sorted(set(index.year)))
        return np.array([d.date() in cal for d in index], dtype=int)
    except Exception:
        return _fixed_holidays(index)

def chronological_split(s: pd.Series, test_fraction: float = config.TEST_FRACTION):
    cut = int(len(s) * (1 - test_fraction))
    return s.iloc[:cut], s.iloc[cut:]
