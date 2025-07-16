# src/evaluator.py

from sklearn.model_selection import cross_val_score
import numpy as np

class Evaluator:
    def __init__(self, scoring='accuracy', cv=5):
        self.scoring = scoring
        self.cv = cv

    def evaluate(self, model, X, y):
        scores = cross_val_score(model, X, y, scoring=self.scoring, cv=self.cv)
        return {
            "mean_score": np.mean(scores),
            "std": np.std(scores),
            "all_scores": scores
        }
