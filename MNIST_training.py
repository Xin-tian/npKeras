import numpy as np
import pickle as pkl

from keras import Model
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
######################################
from keras.datasets import mnist
######################################

def myModel():
    model = Sequential()
    model.add( Conv2D(32, (3, 3), activation='relu', padding='same', 
                      input_shape=(28,28,1), name='conv2d_1') )        
    model.add( Conv2D(32, (3, 3), activation='relu', padding='same', 
                      name='conv2d_2') )     
    model.add( MaxPooling2D((2, 2)) )
    model.add( Conv2D(32, (3, 3), activation='relu', padding='same', 
                      name='conv2d_3') )
    model.add( Conv2D(32, (3, 3), activation='relu', padding='same', 
                      name='conv2d_4') )
    model.add( MaxPooling2D((2, 2)) )
    model.add( Conv2D(64, (3, 3), activation='relu', padding='same', 
                      name='conv2d_5') )
    model.add( Conv2D(64, (3, 3), activation='relu', padding='same', 
                      name='conv2d_6') )
    model.add( Flatten() )
    model.add( Dense(64, activation='relu',   name='dense_1') )
    model.add( Dense(10, activation='softmax', name='dense_2') )
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.reshape(X_train,(-1,28,28,1))
    X_test  = np.reshape(X_test,(-1,28,28,1))
    
    model = myModel()
    model.fit(x=X_train, y=y_train, batch_size=32, epochs=10, 
              validation_split=0.2, verbose=1 )

    y_preds = np.argmax(model.predict( x=X_test, verbose=1 ), axis=1 )
    print('Accuracy = ' + str(np.mean(y_true==y_preds) ) )

    ###################################################
    from .keras_model_exporter import keras_weights

    keras_weights( model, 'MNIST_weights.pkl' )
    ###################################################
