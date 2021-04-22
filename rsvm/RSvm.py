import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import (
    check_classification_targets,
    type_of_target,
)
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)


class RSvm(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        C: float = 1,
        kernel: str = "linear",
        max_iter: int = 1000,
        random_state: int = None,
        degree: int = 3,
        gamma: float = 1.0,
        verbose: bool = False,
    ):
        self.C = C
        self.kernel = kernel
        self.max_iter = max_iter
        self.random_state = random_state
        self.degree = degree
        self.gamma = gamma
        self.verbose = verbose

    def fit(self, X: np.array, y: np.array) -> "RSvm":
        if self.C < 0:
            raise ValueError(
                f"Penalty term must be positive... got (C={self.C:f})"
            )
        self._kernel = {
            "poly": lambda x, y: np.dot(x, y.T) ** self.degree,
            "rbf": lambda x, y: np.exp(
                -self.gamma * np.sum((y - x[:, np.newaxis]) ** 2, axis=-1)
            ),
            "linear": lambda x, y: np.dot(x, y.T),
            "sigmoid": lambda x, y: np.tanh(self.gamma * np.dot(x, y.T)),
        }[self.kernel]
        if type_of_target(y) != "binary":
            labels = np.unique(y)
            raise ValueError(
                f"Only binary problems allowed, found {labels} labels"
            )
        check_classification_targets(y)
        X, y = check_X_y(X, y)
        # self.y_ = RSvm.normalize_label(y)
        self.X_ = X.copy()
        self.y_ = y * 2 - 1
        self._lambdas = np.zeros_like(self.y_, dtype=float)
        # Vectorized version of Wolfe dual problem (p. 61)?
        self._K = (
            self._kernel(self.X_, self.X_) * self.y_[:, np.newaxis] * self.y_
        )

        for iter in range(self.max_iter):
            if self.verbose and iter % 100 == 0:
                print(f"{iter}, ", end="", flush=True)
            for idxM in range(len(self._lambdas)):
                idxL = np.random.randint(0, len(self._lambdas))
                Q = self._K[
                    [[idxM, idxM], [idxL, idxL]], [[idxM, idxL], [idxM, idxL]]
                ]
                v0 = self._lambdas[[idxM, idxL]]
                k0 = 1 - np.sum(self._lambdas * self._K[[idxM, idxL]], axis=1)
                u = np.array([-self.y_[idxL], self.y_[idxM]])
                t_max = np.dot(k0, u) / (np.dot(np.dot(Q, u), u) + 1e-15)
                self._lambdas[[idxM, idxL]] = v0 + u * self.restrict_to_square(
                    t_max, v0, u
                )

        (idx,) = np.nonzero(self._lambdas > 1e-15)
        self._b = np.mean(
            (1.0 - np.sum(self._K[idx] * self._lambdas, axis=1)) * self.y_[idx]
        )
        self.fitted_ = True
        return self

    def restrict_to_square(self, t, v0, u):
        t = (np.clip(v0 + t * u, 0, self.C) - v0)[1] / u[1]
        return (np.clip(v0 + t * u, 0, self.C) - v0)[0] / u[0]

    def decision_function(self, X: np.array) -> np.array:
        """Evaluate decision_function for the samples in X

        Parameters
        ----------
        X : np.array
            The samples to apply the decision function

        Returns
        -------
        np.array
            distances of the samples to the hyperplane
        """
        return (
            np.sum(self._kernel(X, self.X_) * self.y_ * self._lambdas, axis=1)
            + self._b
        )

    def predict(self, X: np.array) -> np.array:
        check_is_fitted(self, ["fitted_"])
        X = check_array(X)
        return (np.sign(self.decision_function(X)) + 1) // 2
