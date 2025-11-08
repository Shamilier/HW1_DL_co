import numpy as np
from scipy.special import erf
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(0.0, input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        grad_input = grad_output * (input > 0)
        return grad_input


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        output = 1.0 / (1.0 + np.exp(-input))
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        sigmoid = self.output if self.output is not None else 1.0 / (1.0 + np.exp(-input))
        grad_input = grad_output * sigmoid * (1.0 - sigmoid)
        return grad_input


class GELU(Module):
    """
    Applies element-wise GELU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return 0.5 * input * (1.0 + erf(input / np.sqrt(2.0)))

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        erf_term = erf(input / np.sqrt(2.0))
        exp_term = np.exp(-0.5 * input ** 2)
        grad_gelu = 0.5 * (1.0 + erf_term) + input * exp_term / np.sqrt(2.0 * np.pi)
        return grad_output * grad_gelu


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        shifted = input - np.max(input, axis=-1, keepdims=True)
        exp_shifted = np.exp(shifted)
        sums = np.sum(exp_shifted, axis=-1, keepdims=True)
        return exp_shifted / sums

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        softmax = self.output if self.output is not None else self.compute_output(input)
        dot = np.sum(grad_output * softmax, axis=-1, keepdims=True)
        return softmax * (grad_output - dot)


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        shifted = input - np.max(input, axis=-1, keepdims=True)
        logsumexp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        return shifted - logsumexp

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        log_softmax = self.output if self.output is not None else self.compute_output(input)
        softmax = np.exp(log_softmax)
        sum_grad = np.sum(grad_output, axis=-1, keepdims=True)
        return grad_output - softmax * sum_grad
