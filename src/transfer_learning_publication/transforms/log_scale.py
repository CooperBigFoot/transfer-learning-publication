import numpy as np

from .base import BaseTransform


class Log(BaseTransform):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def _fit(self, X: np.ndarray) -> dict:
        return {}

    def _transform(self, X: np.ndarray) -> np.ndarray:
        # np.log handles NaN by returning NaN; epsilon shifts values to avoid log(0)
        return np.log(X + self.epsilon)

    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        # np.exp handles NaN by returning NaN
        return np.exp(X) - self.epsilon
