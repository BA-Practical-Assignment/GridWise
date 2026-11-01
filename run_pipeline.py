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