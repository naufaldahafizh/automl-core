# src/automl_pipeline.py

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))
from src.preprocessor import Preprocessor
from src.model_selector import ModelSelector
from src.evaluator import Evaluator
from sklearn.datasets import load_breast_cancer
import joblib
import json

class AutoMLPipeline:
    def __init__(self, scoring='accuracy', cv=5):
        self.preprocessor = Preprocessor()
        self.model_selector = ModelSelector()
        self.evaluator = Evaluator(scoring=scoring, cv=cv)
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.best_model_name = ""

    def save_model(self, path="models/best_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.best_model, path)
        print(f"Model terbaik disimpan di: {path}")

    def save_results(self, path="results/evaluation_report.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Konversi NumPy types agar bisa disimpan ke JSON
        serializable_results = {
            k: {
                "mean_score": float(v["mean_score"]),
                "std": float(v["std"]),
                "all_scores": list(map(float, v["all_scores"]))
            } for k, v in self.results.items()
        }
        with open(path, "w") as f:
            json.dump(serializable_results, f, indent=4)
        print(f"Hasil evaluasi disimpan di: {path}")
    
    def run(self):
        # Load data
        data = load_breast_cancer()
        X, y = data.data, data.target

        # Preprocessing
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        X_train_scaled, X_test_scaled = self.preprocessor.scale(X_train, X_test)

        # Evaluate each model
        for name, model in self.model_selector.get_models().items():
            print(f"[INFO] Evaluating model: {name}")
            model.fit(X_train_scaled, y_train)
            metrics = self.evaluator.evaluate(model, X_train_scaled, y_train)
            self.results[name] = metrics

            mean_score = metrics["mean_score"]
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_model = model
                self.best_model_name = name

        print(f"\n✅ Best model: {self.best_model_name} with score: {self.best_score:.4f}")
        return self.best_model_name, self.results
