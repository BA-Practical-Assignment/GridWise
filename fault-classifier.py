import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

import config

CATEGORIES = ["outage", "billing", "meter_fault", "streetlight", "cable_damage"]


def build_pipeline() -> Pipeline:
    
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", feats),
                     ("clf", LogisticRegression(max_iter=2000, C=5.0))])


def train_and_evaluate(df: pd.DataFrame) -> dict:
    out = {}
    for target in ("category", "urgency"):
        X_tr, X_te, y_tr, y_te = train_test_split(
            df["report"], df[target], test_size=0.2, stratify=df[target],
            random_state=config.RANDOM_STATE)
        pipe = build_pipeline().fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        labels = sorted(df[target].unique())
        out[target] = {
            "model": pipe,
            "accuracy": float(accuracy_score(y_te, pred)),
            "macro_f1": float(f1_score(y_te, pred, average="macro")),
            "labels": labels,
            "confusion": confusion_matrix(y_te, pred, labels=labels).tolist(),
            "report": classification_report(y_te, pred, zero_division=0),
            "n_test": int(len(y_te)),
        }
    return out


def save(models: dict, path=None):
    path = path or config.MODELS_DIR / "fault_classifier.joblib"
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({k: v["model"] for k, v in models.items()}, path)


def load(path=None):
    path = path or config.MODELS_DIR / "fault_classifier.joblib"
    return joblib.load(path)


def classify(models: dict, text: str) -> dict:
    cat = models["category"].predict([text])[0]
    cat_conf = float(models["category"].predict_proba([text]).max())
    urg = models["urgency"].predict([text])[0]
    return {"category": cat, "category_confidence": cat_conf, "urgency": urg}
