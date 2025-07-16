import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluator import Evaluator
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def test_evaluator_accuracy():
    data = load_breast_cancer()
    X_train, _, y_train, _ = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    X_train = StandardScaler().fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    evaluator = Evaluator()
    result = evaluator.evaluate(model, X_train, y_train)

    assert "mean_score" in result
    assert result["mean_score"] > 0.5
