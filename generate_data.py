#Generating SYNTHETIC data used when the real datasets are not found.
import json
import random

import numpy as np
import pandas as pd
import config

# 1. Synthetic hourly electricity demand (MW)
def generate_demand(years: int = 4, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-01", periods=years * 365 * 24, freq="h")
    hour = idx.hour.values
    dow = idx.dayofweek.values
    doy = idx.dayofyear.values

    base = 15000.0
    trend = np.linspace(0, 600, len(idx))                                   
    daily = 2600 * np.sin((hour - 9) / 24 * 2 * np.pi) \
        + 900 * np.sin((hour - 3) / 12 * 2 * np.pi)                         
    weekly = np.where(dow >= 5, -1400, 0)                                  
    yearly = 2200 * np.cos((doy - 20) / 365 * 2 * np.pi) \
        + 1200 * np.cos((doy - 200) / 365 * 4 * np.pi)                      
    noise = rng.normal(0, 450, len(idx))
    demand = base + trend + daily + weekly + yearly + noise

    # public holidays: demand drops
    holiday_mask = np.array([(d.month, d.day) in {(1, 1), (7, 4), (12, 25), (12, 26)} for d in idx])
    demand[holiday_mask] -= 1800

    # a few random heat-wave episodes (demand spikes)
    for _ in range(years * 2):
        start = rng.integers(0, len(idx) - 96)
        demand[start:start + 72] += rng.uniform(1500, 3000)

    return pd.DataFrame({"Datetime": idx, "DEMAND_MW": demand.round(1)})




