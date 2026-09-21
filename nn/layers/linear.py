"""Linear (fully connected) layer: z = xW + b."""

import numpy as np 

from nn.module import Module

class Linear(Module):
    """A fully connected layer computing z = xW + b. 

    Attributes:
        W (np.ndarray): weight matrix, shape
            (in_features, out_features).
        b (np.ndarray): bias vector, shape (out_features,).
    """
    
    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize the layer's weights and bias.

        Args:
            in_features (int): number of input features.
            out_features (int): number of output features.

        Sets:
            self.W (np.ndarray): weight matrix, shape 
                (in_features, out_features). Xavier-initialized,
                not zeros (see "Weights Initialization" below).
            self.b (np.ndarray): bias vector, shape 
                (out_features,). Initialized to zero.
        """
        super().__init__()
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute this layer's output for a batch of inputs.

        Args:
            x (np.ndarray): input, shape (batch_size, in_features).

        Returns:
            np.ndarray: output, shape (batch_size, out_features).
        """
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Args:
            grad_output (np.ndarray): gradient of the loss with respect
                to this layer's output, shape 
                (batch_size, out_features).   

        Returns:
            np.ndarray: gradient of the loss with respect to this layer's
                input, shape 
                (batch_size, in_features).
        """
        self.dW[...] = self.x.T @ grad_output
        self.db[...] = np.sum(grad_output, axis=0)
        return grad_output @ self.W.T

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return this layer's learnable parameters.

        Returns:
            list[tuple[np.ndarray, np.ndarray]]: pairs of 
                (parameter, gradient) --
                [(self.W, self.dW), (self.b, self.db)]. 
        """
        return [(self.W, self.dW), (self.b, self.db)]   
