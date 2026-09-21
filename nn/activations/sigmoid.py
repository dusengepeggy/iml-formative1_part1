"""Sigmoid activation: squashes real values into (0, 1)."""

import numpy as np

from nn.module import Module


class Sigmoid(Module):
    """Sigmoid activation, applied elementwise: 1 / (1 + e^{-x})."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute sigmoid elementwise, and remember the output.

        Args:
            x (np.ndarray): input, any shape.

        Returns:
            np.ndarray: sigmoid(x), elementwise, same shape as x.
        """
        positive = x >= 0
        exp_neg = np.exp(-np.where(positive, x, 0.0))
        exp_pos = np.exp(np.where(positive, 0.0, x))
        self.output = np.where(positive, 1 / (1 + exp_neg), exp_pos / (1 + exp_pos))
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Args:
            grad_output (np.ndarray): gradient of the loss with respect
                to this layer's output, same shape as the original input to forward.

        Returns:
            np.ndarray: gradient of the loss with respect to this layer's
                input, same shape as grad_output.
        """
        return grad_output * self.output * (1 - self.output)
