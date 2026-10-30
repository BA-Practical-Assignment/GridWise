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
    s = s.asfreq("h")                       # PJM files have missing hours and DST duplicates
    n_missing = int(s.isna().sum())
    s = s.interpolate(method="time", limit=6).ffill().bfill()
    s.attrs["n_missing_filled"] = n_missing
    return s