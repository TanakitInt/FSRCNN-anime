# import library

import sys
import keras
import tensorflow as tf

print("Python version : " + sys.version)
print("Keras version : " + keras.__version__)

# import model packages

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Input, Activation, LeakyReLU
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
import numpy as np
import math
import os
import h5py

# import visualization packages

import json
import pydotplus
from matplotlib import pyplot as plt
from keras.utils.vis_utils import model_to_dot
keras.utils.vis_utils.pydot = pydotplus
#os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'

# define the FSRCNN model
def model():
    
    # define model type
    FSRCNN = Sequential()
    
    # add model layers
    FSRCNN.add(Conv2D(filters=56, kernel_size = (5, 5), strides = (1, 1), kernel_initializer='glorot_uniform', padding='same', use_bias=True, input_shape=(None, None, 1)))
    FSRCNN.add(LeakyReLU(alpha=0.1))

    FSRCNN.add(Conv2D(filters=16, kernel_size = (1, 1), strides = (1, 1), kernel_initializer='glorot_uniform', padding='same', use_bias=True))
    FSRCNN.add(LeakyReLU(alpha=0.1))

    FSRCNN.add(Conv2D(filters=12, kernel_size = (3, 3), strides = (1, 1), kernel_initializer='glorot_uniform', padding='same', use_bias=True))
    FSRCNN.add(LeakyReLU(alpha=0.1))

    FSRCNN.add(Conv2D(filters=12, kernel_size = (3, 3), strides = (1, 1), kernel_initializer='glorot_uniform', padding='same', use_bias=True))
    FSRCNN.add(LeakyReLU(alpha=0.1))

    FSRCNN.add(Conv2D(filters=12, kernel_size = (3, 3), strides = (1, 1), kernel_initializer='glorot_uniform', padding='same', use_bias=True))
    FSRCNN.add(LeakyReLU(alpha=0.1))

    FSRCNN.add(Conv2D(filters=12, kernel_size = (3, 3), strides = (1, 1), kernel_initializer='glorot_uniform', padding='same', use_bias=True))
    FSRCNN.add(LeakyReLU(alpha=0.1))

    FSRCNN.add(Conv2D(filters=12, kernel_size = (3, 3), strides = (1, 1), kernel_initializer='glorot_uniform', padding='same', use_bias=True))
    FSRCNN.add(LeakyReLU(alpha=0.1))

    FSRCNN.add(Conv2D(filters=12, kernel_size = (3, 3), strides = (1, 1), kernel_initializer='glorot_uniform', padding='same', use_bias=True))
    FSRCNN.add(LeakyReLU(alpha=0.1))

    FSRCNN.add(Conv2D(filters=56, kernel_size = (1, 1), strides = (1, 1), kernel_initializer='glorot_uniform', padding='same', use_bias=True))
    FSRCNN.add(LeakyReLU(alpha=0.1))

    FSRCNN.add(Conv2DTranspose(filters=1, kernel_size = (9, 9), strides = (1, 1), kernel_initializer='glorot_uniform', padding='same', use_bias=True))
    FSRCNN.add(Activation("sigmoid"))

    model = FSRCNN
    dot_img_file = 'Diagram/fsrcnn-anime_model.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, dpi=120)
    print("Saved model diagram.")

    # define optimizer
    adam = Adam(lr=0.003)
    
    # compile model
    FSRCNN.compile(optimizer=adam, loss='mse', metrics=['mean_squared_error'])
    
    return FSRCNN

def read_training_data(file):

    # read training data

    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))

        train_data = np.transpose(data, (0, 2, 3, 1))
        train_label = np.transpose(label, (0, 2, 3, 1))

        return train_data, train_label

def train():

    # ----------Training----------
    
    fsrcnn_model = model()
    #fsrcnn_model.load_weights("model-checkpoint/fsrcnn-anime-tanakitint-weights-improvement-00010.hdf5")
    print(fsrcnn_model.summary())

    DATA_TRAIN = "h5-dataset/train.h5"
    DATA_TEST = "h5-dataset/test.h5"
    CHECKPOINT_PATH = "model-checkpoint/fsrcnn-anime-tanakitint-weights-improvement-{epoch:05d}.hdf5"
    
    ILR_train, HR_train = read_training_data(DATA_TRAIN)
    ILR_test, HR_test = read_training_data(DATA_TEST)

    # checkpoint
    checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor='mean_squared_error', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # fit model
    history = fsrcnn_model.fit(ILR_train, HR_train, epochs=10, batch_size=32, callbacks=callbacks_list, validation_data=(ILR_test, HR_test))

    # save h5 model
    fsrcnn_model.save("my_model-fsrcnn-anime-tanakitint.h5")
    print("Saved h5 model to disk")

    # ----------Visualization----------

    # training visualization
    training_data = history.history
    print(training_data.keys())

    # text file
    f = open('Diagram/training.txt', 'w')
    f.write(str(training_data))
    f.close()

    # json file
    f = open('Diagram/training.json', 'w')
    training_data = str(training_data)
    f.write(str(training_data.replace("\'", "\"")))
    f.close()

    print("Training Data Saved.")

    # summarize history for val_loss
    fig = plt.figure()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('val_loss')
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    # save fig and show
    plt.savefig('Diagram/fsrcnn-anime_model_loss.png', dpi=120)
    plt.show()
    print("Training Fig Saved.")

    # summarize history for val_mean_squared_error
    fig = plt.figure()

    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('val_mean_squared_error')
    plt.ylabel('val_mean_squared_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # save fig and show
    plt.savefig('Diagram/fsrcnn-anime_model_mean_squared_error.png', dpi=120)
    plt.show()
    print("Training Fig Saved.")

if __name__ == "__main__":

    train()

