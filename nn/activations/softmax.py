"""Softmax activation: converts logits into a probability distribution."""

import numpy as np

from nn.module import Module


class Softmax(Module):
    """Softmax activation, applied row-wise to a batch of logits.

    Unlike ReLU or Sigmoid, each output depends on every logit in its own row,
    not just the matching input -- see "A shape subtlety" above.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities for a batch of logits.

        Args:
            x (np.ndarray): logits, shape (batch_size, C).

        Returns:
            np.ndarray: probabilities, shape (batch_size, C).
                Each row sums to 1.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self._output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self._output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Args:
            grad_output (np.ndarray): gradient of the loss with respect
                to this layer's output, shape
                (batch_size, C).

        Returns:
            np.ndarray: gradient of the loss with respect to this module's
                input, shape
                (batch_size, C).
        """
        probabilities = self._output
        return probabilities * (
            grad_output
            - np.sum(grad_output * probabilities, axis=1, keepdims=True)
        )
