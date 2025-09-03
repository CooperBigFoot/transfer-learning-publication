import numpy as np

from .base import BaseTransform


class Log(BaseTransform):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def _fit(self, X: np.ndarray) -> dict:
        return {}

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return np.log(X + self.epsilon)

    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.exp(X) - self.epsilon
