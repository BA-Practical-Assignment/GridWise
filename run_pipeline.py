import argparse
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis
import config
import data_utils as du
import fault_classifier as fcls
import forecasting as fc
import generate_data
from chatbot import GridWiseChatbot

NAVY, AMBER = analysis.NAVY, analysis.AMBER


def ensure_synthetic_files():
    if not (config.INTENTS_JSON.exists() and config.FAULT_REPORTS_CSV.exists()
            and config.SYNTHETIC_DEMAND_CSV.exists()):
        generate_data.main()


def plot_forecast(actual, preds: dict, out, days: int = 14):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    a = actual.iloc[:24 * days]
    ax.plot(a.index, a.values, color="black", lw=1.4, label="Actual")
    colors = {"seasonal_naive": "#9AA5B5", "hist_gb": AMBER, "mlp": "#2E7D6B", "lstm": "#C0392B"}
    for name, p in preds.items():
        p = p.loc[a.index]
        ax.plot(p.index, p.values, lw=1.0, color=colors.get(name, None), label=name)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.set_title(f"Day-ahead forecast vs actual - first {days} days of the test period")
    ax.set_ylabel("Demand (MW)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

def plot_model_comparison(results: dict, out):
    names = list(results)
    mape = [results[n]["MAPE"] for n in names]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bars = ax.bar(names, mape, color=[NAVY if n != min(results, key=lambda k: results[k]["MAPE"]) else AMBER
                                      for n in names])
     for b, v in zip(bars, mape):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}%", ha="center", fontsize=9)
        ax.set_ylabel("MAPE (%) - lower is better")
        ax.set_title("Model comparison on the test period")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
    
def main(demand_path=None):
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)
    ensure_synthetic_files()
    R = config.RESULTS_DIR

    
    series, source = du.load_demand(demand_path)
    print(f"[data] source={source} rows={len(series)} range={series.index[0]} -> {series.index[-1]} "
          f"(filled {series.attrs.get('n_missing_filled', 0)} missing hours)")
    analysis.plot_overview(series, R / "01_overview.png")
    analysis.plot_decomposition(series, R / "02_decomposition.png")
    analysis.plot_acf(series, R / "03_acf.png")
    analysis.plot_profiles(series, R / "04_profiles.png")

    train, test = du.chronological_split(series)
        X, y = fc.build_supervised(series)
        cut = test.index[0]
        X_tr, y_tr = X[X.index < cut], y[y.index < cut]
        X_te, y_te = X[X.index >= cut], y[y.index >= cut]
    
        preds = {"seasonal_naive": series.shift(168).loc[X_te.index]}
        print("[forecast] tuning HistGradientBoosting with walk-forward CV ...")
        gb = fc.tune_hist_gb(X_tr, y_tr)
        preds["hist_gb"] = pd.Series(gb.predict(X_te), index=X_te.index)
        print("[forecast] training MLP ...")
        mlp = fc.train_mlp(X_tr, y_tr)
        preds["mlp"] = pd.Series(mlp.predict(X_te), index=X_te.index)
    
        results = {n: fc.metrics(y_te, p) for n, p in preds.items()}

    print("[forecast] LSTM (needs torch) ...")
        lstm = fc.lstm_forecast(train, test)
        if lstm is not None:
            common = X_te.index.intersection(lstm["pred"].index)
            preds["lstm"] = lstm["pred"].loc[common].reindex(X_te.index)
            results["lstm"] = fc.metrics(y_te.loc[common], lstm["pred"].loc[common])
        else:
            print("   torch not installed -> LSTM skipped")
    
        print("[forecast] SARIMA on daily means (needs statsmodels) ...")
        sar = fc.sarima_daily(train, test)
        extra = {}
        if sar is not None:
            extra["sarima_daily"] = sar["metrics"]
        else:
            print("   statsmodels not installed -> SARIMA skipped")