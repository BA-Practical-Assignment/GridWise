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