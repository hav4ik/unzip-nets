import tensorflow as tf
import tensorflow.keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.datasets


class toy:
    """A feeder for toy datasets, such as MNIST, CIFAR...
    """
    def __init__(self, batch_size, phase, dataset, shuffle=False):
        dataset_loader = getattr(tf.keras.datasets, dataset)
        (x_train, y_train), (x_test, y_test) = dataset_loader.load_data()
        if phase == 'training':
            self.n = x_train.shape[0]
            self.x, self.y = x_train[:, :, :, np.newaxis], tf.keras.utils.to_categorical(y_train, 10)
        elif phase == 'validating':
            self.n = x_test.shape[0]
            self.x, self.y = x_test[:, :, :, np.newaxis], tf.keras.utils.to_categorical(y_test, 10)
        self._indices = np.arange(self.x.shape[0])
        if shuffle:
            np.random.shuffle(self._indices)
        self.batch_size = batch_size
        self._k = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._k >= self.n:
            np.random.shuffle(self._indices)
            self._k = 0
        x_slice = self.x[self._indices[self._k:self._k+self.batch_size]]
        y_slice = self.y[self._indices[self._k:self._k+self.batch_size]]
        self._k += self.batch_size
        return x_slice, y_slice
