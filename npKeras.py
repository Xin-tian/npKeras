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
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
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
import time

class Layer():
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    def build(self, *args, **kwargs):
        pass
    def build_(self, input_shape):
        if input_shape is not None:
            if input_shape is int:
                input_shape = (input_shape,)
            if self.args == ():
                self.args = [input_shape]
            else:
                self.args = [input_shape] + list(self.args)
        self.build(*self.args, **self.kwargs)

    def get_im2col_indices(self, x_shape, field_height=3, field_width=3,
                           padding=1, h_stride=1, w_stride=1):
        _, C, H, W = x_shape
        assert (H + 2 * padding - field_height) % h_stride == 0
        assert (W + 2 * padding - field_width) % w_stride == 0
        out_height = (H + 2 * padding - field_height) / h_stride + 1
        out_width = (W + 2 * padding - field_width) / w_stride + 1

        i0 = np.repeat(np.arange(field_height, dtype='int32'), field_width)
        i0 = np.tile(i0, C)
        i1 = w_stride * np.repeat(np.arange(out_height, dtype='int32'),
                                  out_width)
        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = w_stride * np.tile(np.arange(out_width, dtype='int32'),
                                int(out_height))
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C, dtype='int32'),
                      field_height * field_width).reshape(-1, 1)
        return (k, i, j)

    def im2col_indices(self, x, field_height=3, field_width=3, padding=1,
                       h_stride=1, w_stride=1):
        """ An implementation of im2col based on some fancy indexing """
        # Zero-pad the input
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        k, i, j = self.get_im2col_indices(x.shape, field_height, field_width,
                                          padding, h_stride, w_stride)
        cols = x_padded[:, k, i, j]
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape( \
                                        field_height * field_width * C, -1)
        return cols

class Conv2D(Layer):
    def build(self, input_shape, filters, kernel_size=(3, 3), strides=(1, 1),
              activation=None, padding='same', name=None):
        self.d_X, self.h_X, self.w_X = input_shape

        self.filters = filters
        if type(kernel_size) is int:
            self.h_filter, self.w_filter = kernel_size, kernel_size
        elif type(kernel_size) is tuple:
            self.h_filter, self.w_filter = kernel_size
        if type(strides) is int:
            self.h_stride, self.w_stride = strides, strides
        elif type(strides) is tuple:
            self.h_stride, self.w_stride = strides
        if padding == 'same':
            self.padding = 1
        else:
            self.padding = 0
        self.W = np.random.randn(
            filters, self.d_X, self.h_filter, self.w_filter) \
                           / np.sqrt(filters / 2.)
        self.b = np.zeros((self.filters, 1))
        self.params = [self.W, self.b]
        self.h_out = (self.h_X - self.h_filter + 2 * self.padding) \
                           /  self.h_stride + 1
        self.w_out = (self.w_X - self.w_filter + 2 * self.padding) \
                           / self.w_stride + 1
        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception(f"Invalid dimensions! {self.h_out} {self.w_out}")
        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.activation = activation
        self.output_shape = (self.filters, self.h_out, self.w_out)
        self.name = name
        self.type = 'Conv2D'

    def forward(self, X):
        self.n_X = X.shape[0]
        X_col = super().im2col_indices(X, self.h_filter, self.w_filter,
            h_stride=self.h_stride, w_stride=self.w_stride, padding=self.padding)
        W_row = self.W.reshape(self.filters, -1)
        out = W_row @ X_col + self.b
        out = out.reshape(self.filters, self.h_out, self.w_out, self.n_X)
        out = out.transpose(3, 0, 1, 2)
        return out

    def set_weights(self, W, b):
        self.W = W
        self.b = np.reshape(b, (-1, 1))

class MaxPooling2D(Layer):
    def build(self, input_shape, size, strides, name=None):
        self.d_X, self.h_X, self.w_X = input_shape
        self.params = []
        self.size = size
        if type(strides) is int:
            self.h_stride, self.w_stride = strides, strides
        elif type(strides) is tuple:
            self.h_stride, self.w_stride = strides
        self.h_out = (self.h_X - size) / self.h_stride + 1
        self.w_out = (self.w_X - size) / self.w_stride + 1
        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")
        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.output_shape = (self.d_X, self.h_out, self.w_out)
        self.name = name
        self.type = 'MaxPooling2D'

    def forward(self, X):
        self.n_X = X.shape[0]
        X_reshaped = X.reshape(X.shape[0] * X.shape[1], 1, X.shape[2],
                               X.shape[3])
        X_col = super().im2col_indices(X_reshaped, self.size, self.size,
                                       padding=0, h_stride=self.h_stride,
                                       w_stride=self.w_stride)
        self.max_indexes = np.argmax(X_col, axis=0)
        out = X_col[self.max_indexes, range(self.max_indexes.size)]
        out = out.reshape(self.h_out, self.w_out, self.n_X,
                          self.d_X).transpose(2, 3, 0, 1)
        return out

class Flatten(Layer):
    def build(self, input_shape, name=None):
        self.params = []
        self.name = name
        self.type = 'Flatten'
        self.output_shape=(np.prod(input_shape),)
    def forward(self, X):
        self.X_shape = X.shape
        self.out_shape = (self.X_shape[0], -1)
        out = X.transpose(0, 2, 3, 1).ravel().reshape(self.out_shape)
        self.out_shape = self.out_shape[1]
        return out

class Dense(Layer):
    def build(self, input_shape, units, activation=None, name=None):
        in_size = input_shape[0]
        self.W = np.random.randn(in_size, units) / np.sqrt(in_size / 2.)
        self.b = np.zeros((1, units))
        self.params = [self.W, self.b]
        self.activation = activation
        self.output_shape = (units,)
        self.name = name
        self.type = 'Dense'
    def forward(self, X):
        out = X @ self.W + self.b
        return out
    def set_weights(self, W, b):
        self.W = W
        self.b = np.reshape(b, (1,-1))


class SparseCategory(Layer):
    def build(self, input_shape, name=None):
        self.output_shape = (1,)
        self.name = name
        self.type = 'SparseCategory'
    def forward(self, X):
        return np.argmax(X, axis=1)

class ReLU(Layer):
    def build(self, input_shape, name=None):
        self.params = []
        self.output_shape = input_shape
        self.name = name
        self.type = 'ReLU'
    def forward(self, X):
        # self.X = X
        return np.maximum(X, 0)

class softmax(Layer):
    def build(self, input_shape, name=None):
        self.params = []
        self.output_shape = input_shape
        self.name = name
        self.type = 'softmax'
    def forward(self, X):
        def softmax_(x):
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return softmax_(X)

class Sequential:
    def __init__(self, input_shape=None, name=None):
        self.params = []
        self.name = name
        self.type = 'Model'
        self.layers = []
        self.output_shape=input_shape

    def add(self, layer):
        self.layers.append(layer)
        layer.build_(self.output_shape)
        self.output_shape = layer.output_shape
        if layer.name is None:
            layer.name = f'_{layer.type.lower()}_{str(len(self.layers))}'
        if hasattr(layer, 'activation') == True and layer.activation != None:
            if layer.activation == 'relu':
                self.add(ReLU())
            elif layer.activation == 'softmax':
                self.add(softmax())

    def set_weights(self, weights, verbose=1):
        for key in weights.keys():
            for layer in self.layers:
                if layer.name == key:
                    W_shape_old = layer.W.shape
                    b_shape_old = layer.b.shape
                    W, b = weights[key]
                    W_shape_new = layer.W.shape
                    b_shape_new = layer.b.shape
                    if W_shape_old != W_shape_new or b_shape_old != b_shape_new:
                        raise Exception("Weights shape mismatch")
                    layer.set_weights(W, b)
                    if verbose == 1:
                        print( \
                 f' Layer {layer.name}({layer.type}) W={W.shape} b={b.shape}')

    def load_weights(self, file_, verbose=1):
        pickle_in = open(file_, 'rb')
        weights = pkl.load(pickle_in)
        pickle_in.close()
        if verbose == 1:
            print(f'\nLoading model weights from file {file_}')
        self.set_weights(weights, verbose=verbose)
        if verbose == 1:
            print(' ')

    def _predict(self, X):
        try:
            x = X
            for layer in self.layers:
                x = layer.forward(x)
            return x
        except Exception:
            raise Exception("error in prediction")

    def predict(self, X, batch_size=0, n_proc=0, verbose=0):
        batches = int(np.ceil(X.shape[0]/batch_size))
        parts = []
        if batch_size == 0:
            return self._predict(X)
        for i in range(batches):
            parts.append(X[i*batch_size:min(X.shape[0], (i+1)*batch_size)])

        if n_proc <= 0:
            if verbose == 1:
                t0 = time.time()
                print(f'Predicting N={X.shape[0]} batch_size={batch_size} single thread')
            y = np.concatenate([self._predict(part) for part in parts], axis=0)
            if verbose == 1:
                print(f'==> done in {time.time() - t0:.2f}s')
            return y
        from multiprocessing import Pool
        if verbose == 1:
            t0 = time.time()
            print(f'Predicting N={X.shape[0]} batch_size={batch_size} n_proc={n_proc}')
        y = np.concatenate(Pool(n_proc).map(self._predict, parts), axis=0)
        if verbose == 1:
            print(f'==> done in {time.time() - t0:.2f}s')
        return y

    def accuracy(self, y_true, y_preds):
        return np.mean(y_true == y_preds)

    def evaluate(self, X, y_true, batch_size=0, n_proc=1):
        y_preds = self.predict(X)
        return self.accuracy(y_true, y_preds)

    def summary(self):
        print('==============================================================')
        print('Layer (type)                     Output Shape        Param #')
        print('==============================================================')
        cnt = 0
        total_parms = 0
        for layer in self.layers:
            cnt += 1
            if layer.name is not None:
                name = f'{layer.name} ({layer.type})'
            else:
                name = f'_{layer.type.lower()}_{str(cnt)} ({layer.type})'
                layer.name = name
            if hasattr(layer, 'W'):
                parms = int(np.prod(layer.W.shape) + np.prod(layer.b.shape))
            else:
                parms = 0
            total_parms += parms
            print(f'{name:<32} {str(layer.output_shape):<12} {parms:15d}')
            print('--------------------------------------------------------------')
        print(f'Total params: {total_parms:,d}')
        print('==============================================================')
