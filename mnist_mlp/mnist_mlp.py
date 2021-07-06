'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import argparse

def main():
    batch_size = 128
    num_classes = 10
    epochs = 20

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("-e", "--epoch", type=int, default=10)
    parser.add_argument("-b", "--batch", type=int, default=128)
    parser.add_argument("-d", "--device", type=str, default="/device:ve:0")
    args = parser.parse_args()

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    if args.verbose > 0:
      print(f'{x_train.shape[0]} train samples')
      print(f'{x_test.shape[0]} test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    with tf.device(args.device):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(784,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))

        if args.verbose > 0:
          model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])

        t0 = time.time()
        history = model.fit(x_train, y_train,
                            batch_size=args.batch,
                            epochs=args.epoch,
                            verbose=args.verbose,
                            validation_data=(x_test, y_test))
        elapsed = time.time() - t0
        score = model.evaluate(x_test, y_test, verbose=0)
        if args.verbose > 0:
            print(f'Test loss: {score[0]}')
            print(f'Test accuracy: {score[1]}')

        images = args.epoch * x_train.shape[0]
        if args.verbose > 0:
            print(f'Elapsed time: {elapsed:8.3f} sec for {args.epoch} epochs. {images / elapsed / 1e3:8.3f} Kimages/sec')

        return [elapsed, score]

if __name__ == "__main__":
    main()
