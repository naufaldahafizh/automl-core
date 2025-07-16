# src/model_selector.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class ModelSelector:
    def __init__(self):
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(eval_metric='logloss')
        }

    def get_models(self):
        return self.models
