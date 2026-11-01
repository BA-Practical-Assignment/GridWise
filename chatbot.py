from __future__ import annotations

import json
import random
import re
import uuid

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

import config

CONFIDENCE_THRESHOLD = 0.30
FALLBACK = ("Sorry, I did not quite understand that. I can help you report a power outage, a meter or "
            "street-light fault or a damaged cable, answer billing questions, give energy-saving tips "
            "or share the demand forecast. Could you rephrase?")
FAULT_INTENTS = {"report_outage", "meter_fault", "streetlight", "cable_damage", "billing_query", "outage_status"}

def _pipeline() -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)),
    ])
    return Pipeline([("tfidf", feats), ("clf", LogisticRegression(max_iter=3000, C=10.0))])


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class GridWiseChatbot:
    def __init__(self, intents_path=config.INTENTS_JSON, fault_models=None,
                 forecast_fn=None, seed: int = config.RANDOM_STATE):
        with open(intents_path, encoding="utf-8") as f:
            self.intents = json.load(f)
        self.fault_models = fault_models        
        self.forecast_fn = forecast_fn          
        self.rng = random.Random(seed)
        self.tickets: list[dict] = []
        self.model = None

    def _xy(self):
        X, y = [], []
        for intent, d in self.intents.items():
            for p in d["patterns"]:
                X.append(_clean(p))
                y.append(intent)
        return X, y

    def train(self):
        X, y = self._xy()
        self.model = _pipeline().fit(X, y)
        return self

    def cross_validate(self, folds: int = 3) -> float:
        X, y = self._xy()
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)
        pred = cross_val_predict(_pipeline(), X, y, cv=skf)
        return float(accuracy_score(y, pred))

    def save(self, path=None):
        path = path or config.MODELS_DIR / "chatbot_intents.joblib"
        config.MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(self.model, path)

    def predict_intent(self, text: str):
        probs = self.model.predict_proba([_clean(text)])[0]
        i = int(np.argmax(probs))
        return self.model.classes_[i], float(probs[i])