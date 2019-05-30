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
import time             # for time stats

####################################################
from npKeras import Sequential
from npKeras import Conv2D, MaxPooling2D, Flatten, Dense
from npKeras import ReLU, softmax
from npKeras import SparseCategory  # Extra Layer not in Keras
####################################################


#%% Model definition 
def cnn(input_shape=(1, 28, 28), num_class=10):
    model = Sequential(input_shape=input_shape ) 
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', 
                        activation='relu', name='conv2d_1'))
    #model.add(ReLU()) # Activation as separate layer
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', 
                        activation='relu', name='conv2d_2'))
    model.add(MaxPooling2D(size=2, strides=2))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', 
                        activation='relu', name='conv2d_3'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', 
                        activation='relu', name='conv2d_4'))
    model.add(MaxPooling2D(size=2, strides=2))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', 
                        activation='relu', name='conv2d_5'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', 
                        activation='relu', name='conv2d_6'))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name='dense_1'))
    model.add(Dense(num_class, activation='softmax', name='dense_2'))
    model.add(SparseCategory())
    return model


def main():

    # Load MNIST dataset
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    (X_train, y_train), (X_test, y_test) = pkl.load(open('MNIST_data.pkl', 'rb'))

    # 注意 Format (NCHW) N x Color x Height x Width
    X_train = np.reshape(X_train, (-1, 1, 28, 28))  
    X_test  = np.reshape(X_test, (-1, 1, 28, 28))  

    # Model preparation
    model = cnn()
    print('CNN model')
    model.summary()
    model.load_weights('MNIST_weights.pkl', verbose=1)

    # Prediction
    #####################################################################
    y_preds = model.predict(X_test, batch_size=250, n_proc=0,  verbose=1)
    #####################################################################
    
    acc = np.mean(y_test == y_preds)
    print('.')
    print('*********************')
    print(f'accuracy = {acc:>10.5f}')
    print('*********************')
    print('.')

    #%% Demo
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.title(f'{y_preds[i]}/{y_test[i]}')
        fig = plt.imshow(np.reshape(X_test[i],(28,28)),cmap = 'Greys')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.show()

    print('Trying execute with multiprocessing ...')
    try:
        import sys
        sys.tracebacklimit = 0        
        ####################################################################
        y_preds = model.predict(X_test, batch_size=250, n_proc=5,  verbose=1)
        ####################################################################
        
        acc = np.mean(y_test==y_preds)
        print('*********************')
        print(f'accuracy = {acc:>10.5f}')
        print('*********************')
    except Exception:
        print('Sorry: Multiprocessig not work in your Python environment\n')

if __name__ == "__main__":
    try:
        main()
    except:
        pass
    print('Done')
