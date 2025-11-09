import numpy as np

from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        loss = np.mean((input - target) ** 2)
        return np.array(loss, dtype=input.dtype)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        grad_input = 2.0 * (input - target) / input.size
        return grad_input


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.label_smoothing = label_smoothing

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        batch_size, num_classes = input.shape
        log_probs = self.log_softmax(input)
        smoothing = self.label_smoothing
        if smoothing > 0:
            true_dist = np.full_like(log_probs, smoothing / num_classes)
            true_dist[np.arange(batch_size), target] += 1.0 - smoothing
            loss = -np.sum(true_dist * log_probs) / batch_size
        else:
            loss = -np.mean(log_probs[np.arange(batch_size), target])
        return np.array(loss, dtype=input.dtype)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        batch_size, num_classes = input.shape
        log_probs = self.log_softmax.output
        if log_probs is None:
            log_probs = self.log_softmax(input)
        probs = np.exp(log_probs)
        smoothing = self.label_smoothing
        if smoothing > 0:
            true_dist = np.full_like(probs, smoothing / num_classes)
            true_dist[np.arange(batch_size), target] += 1.0 - smoothing
        else:
            true_dist = np.zeros_like(probs)
            true_dist[np.arange(batch_size), target] = 1.0

        grad_input = (probs - true_dist) / batch_size
        return grad_input
