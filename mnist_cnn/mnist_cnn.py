# https://raw.githubusercontent.com/keras-team/keras/master/examples/mnist_cnn.py

'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument("--log-device", action="store_true")
parser.add_argument("--nhwc", action="store_true")
parser.add_argument("--epoch", type=int)
parser.add_argument("--data-size", type=int)
parser.add_argument("--no-evaluate", action="store_true")
parser.add_argument("--no-validation", action="store_true")
parser.add_argument("--num-filters", type=int)
#parser.add_argument("--inter-op", type=int)
parser.add_argument("--profile", type=str)
parser.add_argument("--disable_eager", action="store_true")
args = parser.parse_args()

if args.disable_eager :
  tf.compat.v1.disable_eager_execution() 

if args.profile :
  from datetime import datetime
  dirname = datetime.now().strftime("%Y%m%d-%H%M")
  log_filepath = args.profile + "/" + dirname
  tbcb = keras.callbacks.TensorBoard(log_dir=log_filepath)
  callbacks = [tbcb]
else :
  callbacks = []

if args.nhwc:
    K.set_image_data_format('channels_last')
else:
    K.set_image_data_format('channels_first')

batch_size = 128
num_classes = 10
epochs = 12
if args.epoch:
    epochs = args.epoch

print("batch_size={} num_classes={} epoch={}".format(batch_size, num_classes, epochs))


# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if args.data_size:
    x_train = x_train[0:args.data_size]
    y_train = y_train[0:args.data_size]
    x_test = x_test[0:args.data_size]
    y_test = y_test[0:args.data_size]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

if args.no_validation:
  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks)
else :
  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks=callbacks)

if not args.no_evaluate:
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

