import numpy as np

from .base import BaseTransform


class ZScore(BaseTransform):
    def _fit(self, X: np.ndarray) -> dict:
        # Use regular mean/std so NaN values propagate to results
        return {"mean": np.mean(X, axis=0), "std": np.std(X, axis=0)}

    def _transform(self, X: np.ndarray) -> np.ndarray:
        mean = self._fitted_state["mean"]
        std = self._fitted_state["std"]
        # Handle zero std (avoid division by zero)
        std = np.where(std == 0, 1.0, std)
        # NaN-safe subtraction and division
        return (X - mean) / std

    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        mean = self._fitted_state["mean"]
        std = self._fitted_state["std"]
        std = np.where(std == 0, 1.0, std)
        # NaN-safe operations
        return X * std + mean
