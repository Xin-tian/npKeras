##################################################################################
# MIT License
#
# Copyright (c) 2018 新的天 Xin de Tian
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
##################################################################################
# https://github.com/Xin-tian/npKeras
##################################################################################

import numpy as np
import pickle as pkl

from keras import Model
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
####################################################
from keras.datasets import mnist
####################################################
from keras_weights_exporter import keras_weights


def myModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation = 'relu', padding='same',
                      input_shape=(28, 28, 1), name='conv2d_1'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                      name='conv2d_2'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                      name='conv2d_3'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                      name='conv2d_4'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                      name='conv2d_5'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                      name='conv2d_6'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu',   name='dense_1'))
    model.add(Dense(10, activation='softmax', name='dense_2'))
    return model


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.reshape(X_train,(-1, 28, 28, 1))
    X_test  = np.reshape(X_test,(-1, 28, 28, 1))

    model = myModel()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=32, epochs=3,
              validation_split=0.2, verbose=1)

    y_preds = np.argmax(model.predict(X_test, verbose=1), axis=1)
    print('Accuracy = ' + str(np.mean(y_test == y_preds)))

    ######################################################
    # Export Keras model weights
    keras_weights(model, 'MNIST_weights.pkl', verbose=1)
    ######################################################
