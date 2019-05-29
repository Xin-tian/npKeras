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
from keras import Model
from keras.layers import Conv2D, Dense
import pickle as pkl

def keras_weights(model, file_, verbose=1):
    weight_dict = dict()
    for layer in model.layers:
        if type(layer) is Conv2D:
            weight_dict[layer.get_config()['name']] = \
            ( np.transpose(layer.get_weights()[0], (3, 2, 0, 1)),
              layer.get_weights()[1] )
        elif type(layer) is Dense:
            weight_dict[layer.get_config()['name']] = \
            ( np.transpose(layer.get_weights()[0], (0, 1)), 
              layer.get_weights()[1] )

    pickle_out = open( file_, 'wb')
    pkl.dump( weight_dict, pickle_out)
    pickle_out.close()
    if verbose==1:
      print(f'Weights exported to file: {file_}')
