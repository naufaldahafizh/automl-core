import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessor import Preprocessor
from sklearn.datasets import load_breast_cancer

def test_split_and_scale():
    data = load_breast_cancer()
    X, y = data.data, data.target
    prep = Preprocessor()

    X_train, X_test, y_train, y_test = prep.split_data(X, y)
    X_train_scaled, X_test_scaled = prep.scale(X_train, X_test)

    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
