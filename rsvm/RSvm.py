import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class RSvm(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1, random_state=None):
        self.C = C
        self.random_state = random_state

    def fit(self, X: np.array, y: np.array) -> "RSvm":
        return self

    def predict(self, X: np.array) -> np.array:
        return np.random.randint(low=0, high=2, size=(X.shape[0],))
