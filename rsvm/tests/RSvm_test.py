from sklearn.datasets import load_iris
import unittest
from rsvm import RSvm
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


class RSvm_test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._random_state = 1
        self.clf = RSvm(random_state=self._random_state)
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_dataset(binary=True):
        X, y = load_iris(return_X_y=True)
        if binary:
            y[y == 2] = 1
        return X, y

    def test_verbose(self):
        X, y = self.get_dataset()
        self.clf.set_params(**{"verbose": True})
        self.clf.fit(X, y)
        self.assertTrue(self.clf.verbose)

    def test_fit(self):
        X, y = self.get_dataset()
        self.clf.fit(X, y)
        check_is_fitted(self.clf, ["fitted_"])

    def test_C_bad_value(self):
        clf = RSvm(C=-1)
        with self.assertRaises(ValueError):
            clf.fit(*self.get_dataset())

    def test_not_binary(self):
        with self.assertRaises(ValueError):
            self.clf.fit(*self.get_dataset(binary=False))

    def test_predict(self):
        X, y = self.get_dataset()
        computed = self.clf.fit(X, y).predict(X)
        self.assertListEqual(y.tolist(), computed.tolist())

    def test_predict_not_fitted(self):
        X, y = self.get_dataset()
        with self.assertRaises(NotFittedError):
            self.clf.predict(X)
