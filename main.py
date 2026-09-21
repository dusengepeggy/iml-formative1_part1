"""Training entry point: wires Linear, Sigmoid, and CrossEntropyLoss into a loop."""

import numpy as np

from nn.activations import Sigmoid
from nn.layers import Linear
from nn.losses import CrossEntropyLoss
from nn.optim import SGD

_model = None
_loss_fn = None
_data = None


def toy_data() -> tuple[np.ndarray, np.ndarray]:
    """Build the AND-gate toy dataset used for the convergence sanity check.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y) where X has shape (4, 2) and
            y has shape (4, 1), the binary AND-gate labels.
    """
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [0.0], [0.0], [1.0]])
    return x, y


def train(epochs: int = 4000, lr: float = 1.0, seed: int = 0) -> list[float]:
    """Train a Linear + Sigmoid model with CrossEntropyLoss on the toy data.

    Args:
        epochs (int): number of full-batch training steps to run.
        lr (float): learning rate passed to the SGD optimizer.
        seed (int): seed for numpy's global RNG, for reproducible weights.

    Returns:
        list[float]: the loss recorded at every epoch, in order.
    """
    global _model, _loss_fn, _data

    np.random.seed(seed)
    x, y = toy_data()
    _data = (x, y)

    linear = Linear(in_features=x.shape[1], out_features=1)
    sigmoid = Sigmoid()
    _model = (linear, sigmoid)
    _loss_fn = CrossEntropyLoss()

    optimizer = SGD(linear.parameters(), lr=lr)

    history = []
    for _ in range(epochs):
        z = linear.forward(x)
        a = sigmoid.forward(z)
        loss = _loss_fn.forward(a, y)
        history.append(loss)

        grad = _loss_fn.backward()
        grad = sigmoid.backward(grad)
        linear.backward(grad)

        optimizer.step()
        optimizer.zero_grad()

    return history


def accuracy(loss_history: list[float] = None) -> float:
    """Report classification accuracy of the most recently trained model.

    Args:
        loss_history (list[float]): unused; accepted for interface
            compatibility with the required signature. Call train()
            beforehand to populate the model this function evaluates.

    Returns:
        float: fraction of toy-data examples classified correctly.
    """
    del loss_history
    if _model is None:
        train()
    linear, sigmoid = _model
    x, y = _data
    predictions = sigmoid.forward(linear.forward(x))
    predicted_labels = (predictions >= 0.5).astype(float)
    return float(np.mean(predicted_labels == y))


if __name__ == "__main__":
    loss_history = train()
    for epoch in (0, 999, 1999, 2999, 3999):
        if epoch < len(loss_history):
            print(f"epoch {epoch}: loss = {loss_history[epoch]:.6f}")
    print(f"final accuracy: {accuracy():.2%}")
