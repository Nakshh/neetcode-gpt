import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-15
        loss = 0
        for i in range(len(y_true)):
            ln = y_true[i] * np.log(y_pred[i]+epsilon) + (1-y_true[i]) * np.log(1-y_pred[i]+epsilon)
            loss += ln
        loss = -loss/len(y_true)
        return round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-15
        loss = 0
        for true, pred in zip(y_true, y_pred):
            for i in range(len(true)):
                ln = true[i] * np.log(pred[i]+epsilon)
                loss += ln
        loss = -loss/len(y_true)
        return round(loss, 4)
