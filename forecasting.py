from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from data_utils import holiday_flag

H = config.HORIZON
LAGS = [H, H + 24, 168, 336]          



def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_true - y_pred
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(np.abs(err / y_true)) * 100),
    }


def make_features(s: pd.Series, country: str = "US") -> pd.DataFrame:
    
    df = pd.DataFrame(index=s.index)
    for k in LAGS:
        df[f"lag_{k}"] = s.shift(k)
    base = s.shift(H)
    df["roll_mean_24"] = base.rolling(24).mean()
    df["roll_mean_168"] = base.rolling(168).mean()
    df["roll_std_24"] = base.rolling(24).std()
    idx = s.index
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_holiday"] = holiday_flag(idx, country)
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    return df



def build_supervised(s: pd.Series, country: str = "US"):
    X = make_features(s, country)
    y = s
    ok = X.notna().all(axis=1)
    return X[ok], y[ok]



def tune_hist_gb(X_train, y_train) -> HistGradientBoostingRegressor:
    
    grid = [
        {"learning_rate": lr, "max_depth": md, "max_iter": it}
        for lr in (0.05, 0.1) for md in (4, 8) for it in (200,)
    ]
    tscv = TimeSeriesSplit(n_splits=3)
    best, best_score = None, np.inf
    for params in grid:
        scores = []
        for tr, va in tscv.split(X_train):
            m = HistGradientBoostingRegressor(random_state=config.RANDOM_STATE, **params)
            m.fit(X_train.iloc[tr], y_train.iloc[tr])
            scores.append(np.mean(np.abs(y_train.iloc[va] - m.predict(X_train.iloc[va]))))
        score = float(np.mean(scores))
        if score < best_score:
            best, best_score = params, score
    model = HistGradientBoostingRegressor(random_state=config.RANDOM_STATE, **best)
    model.fit(X_train, y_train)
    model.best_params_ = best
    model.cv_mae_ = best_score
    return model