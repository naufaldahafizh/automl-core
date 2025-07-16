# src/preprocessor.py

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.scaler = StandardScaler()
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, X, y):
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def scale(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
