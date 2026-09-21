"""Categorical cross-entropy loss, one-hot multi-class targets."""

import numpy as np


class CategoricalCrossEntropyLoss:
    """Compute cross-entropy loss over C classes.

    Does not subclass Module -- see the note in Chapter 6.
    """

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the average categorical cross-entropy loss.

        Args:
            predictions (np.ndarray): softmax probabilities,
                shape (m, C). Clip away from exactly 0 before use.
                --see "The same clipping requirements as Chapter 6" above.
            targets (np.ndarray): one-hot true labels, shape (m, C),

        Returns:
            float: the scalar loss, averaged over the batch.
        """
        self.predictions = np.clip(predictions, 1e-12, 1 - 1e-12)
        self.targets = targets
        m = predictions.shape[0]
        return float(-np.sum(targets * np.log(self.predictions)) / m)

    def backward(self) -> np.ndarray:
        """Compute the gradient of the loss w.r.t predictions.

        Returns:
            np.ndarray: dL/da, shape (m, C), same shape as the predictions passed
                to forward. Use the same clipped predictions here as in
                forward.
        """
        m = self.predictions.shape[0]
        return -self.targets / self.predictions / m
