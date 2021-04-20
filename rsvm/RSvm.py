import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class RSvm(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1, random_state=None):
        self.C = C
        self.random_state = random_state

    def fit(self, X: np.array, y: np.array) -> "RSvm":
        self.fitted_ = True
        return self

    def predict(self, X: np.array) -> np.array:
        return np.ones((X.shape[0],), dtype=np.int8)
