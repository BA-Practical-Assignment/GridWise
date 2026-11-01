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