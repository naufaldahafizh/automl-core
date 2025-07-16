import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_selector import ModelSelector

def test_model_selector_returns_models():
    selector = ModelSelector()
    models = selector.get_models()
    assert isinstance(models, dict)
    assert "LogisticRegression" in models
