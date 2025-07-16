import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.automl_pipeline import AutoMLPipeline

def test_pipeline_runs_and_returns_model():
    automl = AutoMLPipeline()
    best_model_name, results = automl.run()

    assert best_model_name in results
    assert "mean_score" in results[best_model_name]

    automl.save_model("models/test_model.pkl")
    automl.save_results("results/test_results.json")
