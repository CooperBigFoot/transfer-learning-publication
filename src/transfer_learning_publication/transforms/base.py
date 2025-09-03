from abc import ABC, abstractmethod

import numpy as np


class BaseTransform(ABC):
    def __init__(self):
        self._is_fitted = False
        self._fitted_state = {}

    @abstractmethod
    def _fit(self, X: np.ndarray) -> dict:
        pass

    @abstractmethod
    def _transform(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit(self, X: np.ndarray) -> "BaseTransform":
        self._validate_input(X)
        self._fitted_state = self._fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Transform must be fitted before transform()")
        self._validate_input(X)
        return self._transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Transform must be fitted before inverse_transform()")
        self._validate_input(X)
        return self._inverse_transform(X)

    def _validate_input(self, X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if X.ndim != 2:
            raise ValueError(f"Input must be 2D array, got {X.ndim}D")
