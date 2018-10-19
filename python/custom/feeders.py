import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2


class toy:
    """A feeder for toy datasets, such as MNIST, CIFAR...
    """
    def __init__(self,
                 batch_size,
                 phase,
                 dataset='mnist',
                 mode='gray',
                 normalization=False,
                 augmentation=False,
                 size=None,
                 shuffle=False):
        dataset_loader = getattr(tf.keras.datasets, dataset)
        (x_train, y_train), (x_test, y_test) = dataset_loader.load_data()
        n = y_test.max() + 1
        if phase == 'training':
            self.n = x_train.shape[0]
            self.x, self.y = x_train, tf.keras.utils.to_categorical(y_train, n)
            if len(self.x.shape) == 3:
                self.x = self.x[:, :, :, np.newaxis]
        elif phase == 'validating':
            self.n = x_test.shape[0]
            self.x, self.y = x_test, tf.keras.utils.to_categorical(y_test, n)
            if len(self.x.shape) == 3:
                self.x = self.x[:, :, :, np.newaxis]

        if mode == 'gray' and self.x.shape[-1] == 3:
            self.x = np.mean(self.x, axis=-1, keepdims=True)
        if mode == 'rgb' and self.x.shape[-1] == 1:
            self.x = np.concatenate([self.x] * 3, axis=-1)

        if size is not None:
            images = np.zeros(shape=(self.n, size, size, self.x.shape[-1]))
            for i in range(self.n):
                images[i, ...] = cv2.resize(self.x[i], (size, size))
            self.x = images

        if normalization:
            self.x = self.x / 127.5 - 1.

        self._indices = np.arange(self.x.shape[0])
        if shuffle:
            np.random.shuffle(self._indices)
        self.batch_size = batch_size
        self._k = 0
        self.shuffle = shuffle

        self._augmentation = augmentation
        if augmentation:
            generator = ImageDataGenerator(rotation_range=10,
                                           zoom_range=0.1,
                                           horizontal_flip=True,
                                           vertical_flip=False)
            self.feeder = generator.flow(
                    self.x, self.y,
                    batch_size=self.batch_size, shuffle=shuffle)

    def __iter__(self):
        return self

    def __next__(self):
        if not self._augmentation:
            if self._k + self.batch_size > self.n:
                if self.shuffle:
                    np.random.shuffle(self._indices)
                self._k = 0
            x_slice = self.x[self._indices[self._k:self._k+self.batch_size]]
            y_slice = self.y[self._indices[self._k:self._k+self.batch_size]]
            self._k += self.batch_size
        else:
            x_slice, y_slice = next(self.feeder)
        return x_slice, y_slice
