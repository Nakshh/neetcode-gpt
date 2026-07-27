import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)

        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)

        y_true = np.array(y_true, dtype=float)

        # Forward pass
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)
        z2 = W2 @ a1 + b2

        # loss
        loss = np.mean((z2-y_true)**2)
        
        # z2 layer
        dz2 = (2/len(y_true)) * (z2 - y_true)
        dW2 = np.outer(dz2, a1)
        db2 = dz2

        # a1 layer
        da1 = dz2 @ W2

        # z1 layer
        dz1 = da1 * (z1 > 0)
        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {
            "loss": np.round(loss, 4),
            "dW1": np.round(dW1, 4),
            "db1": np.round(db1, 4),
            "dW2": np.round(dW2, 4),
            "db2": np.round(db2, 4)
        }

