import math
import numpy as np


class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        return math.ceil(self.num_samples() / self.batch_size)

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        return self.X.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        self.batch_id = 0
        num_samples = self.num_samples()
        if self.shuffle:
            self._indices = np.random.permutation(num_samples)
        else:
            self._indices = np.arange(num_samples)
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        start = self.batch_id * self.batch_size
        if start >= self.num_samples():
            raise StopIteration

        end = min(start + self.batch_size, self.num_samples())
        indices = self._indices[start:end]

        X_batch = self.X[indices]
        y_batch = self.y[indices]

        self.batch_id += 1
        return X_batch, y_batch
