import numpy as np
from sklearn.datasets import load_wine
import unittest
from rsvm import RSvm


class RSvm_test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._random_state = 1
        self._clf = RSvm(random_state=self._random_state)
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_dataset():
        return load_wine(return_X_y=True)

    def test_fit(self):
        X, y = self.get_dataset()
        self._clf.fit(X, y)
        self.assertTrue(self._clf.fitted_)

    def test_predict(self):
        X, y = self.get_dataset()
        computed = self._clf.fit(X, y).predict(X)
        expected = np.ones((X.shape[0],), dtype=np.int8).tolist()
        self.assertListEqual(expected, computed.tolist())
