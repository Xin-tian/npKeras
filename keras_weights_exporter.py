import pickle as pkl
from keras import Model

def keras_weights(model, file_):
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
