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


def train_mlp(X_train, y_train):
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=300,
                     random_state=config.RANDOM_STATE),
    )
    model.fit(X_train, y_train)
    return model



def sarima_daily(train: pd.Series, test: pd.Series):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        return None
    d_train = train.resample("D").mean()
    d_test = test.resample("D").mean()
    model = SARIMAX(d_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    pred = fit.forecast(steps=len(d_test))
    return {"actual": d_test, "pred": pd.Series(pred.values, index=d_test.index),
            "metrics": metrics(d_test.values, pred.values)}


def lstm_forecast(train: pd.Series, test: pd.Series, window: int = 168, epochs: int = 12):
    
    try:
        import torch
        from torch import nn
    except ImportError:
        return None

    torch.manual_seed(config.RANDOM_STATE)
    full = pd.concat([train, test])
    mu, sd = train.mean(), train.std()
    z = ((full - mu) / sd).values.astype("float32")

   
    n = len(z) - window - H + 1
    Xs = np.stack([z[i:i + window] for i in range(n)])[..., None]
    ys = np.array([z[i + window - 1 + H] for i in range(n)], dtype="float32")
    target_idx = full.index[window - 1 + H: window - 1 + H + n]

    is_test = target_idx >= test.index[0]
    X_tr, y_tr = Xs[~is_test], ys[~is_test]
    X_te, y_te = Xs[is_test], ys[is_test]