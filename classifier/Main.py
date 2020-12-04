import tensorflow as tf
import keras
from keras import layers
import numpy as np
from models import CNN, CNN_BN, CNN_Drop

import os

print(os.getcwd())

def CNN_prediction(images):
    ''' 
    param: (list of list of list)/(single nested list) list of 28*28 matrices 
    return: probs (list of lists) list of probability distribution as predicted from the model 
            preds (list) the predicted figure judging by the highest probability 
    '''
    cnn = CNN()
    cnn.load_weights('weights/ckpt_CNN')
    probs = cnn.predict(images)
    preds = np.argmax(probs, axis=1)

    return probs, preds

def CNN_Dropout_prediction(images):
    ''' 
    param: (list of list of list)/(single nested list) list of 28*28 matrices 
    return: probs (list of lists) list of probability distribution as predicted from the model 
            preds (list) the predicted figure judging by the highest probability 
    '''
    cnn_dropout = CNN_Drop()
    cnn_dropout.load_weights('weights/ckpt_CNN_Drop')
    probs = cnn_dropout.predict(images)
    preds = np.argmax(probs, axis=1)
    return probs, preds

def CNN_BatchNromalized_prediction(images):
    ''' 
    param: (list of list of list)/(single nested list) list of 28*28 matrices 
    return: probs (list of lists) list of probability distribution as predicted from the model 
            preds (list) the predicted figure judging by the highest probability 
    '''
    cnn_bn = CNN_BN()
    cnn_bn.load_weights('weights/ckpt_CNN_BN')
    probs = cnn_bn.predict(images)
    preds = np.argmax(probs, axis=1)
    return probs, preds


if __name__ == '__main__':
    num_classes = 10
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (x_train, y_train_), (x_test, y_test_) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    test_data = x_test[:3]
    print("naive CNN:")
    print(CNN_prediction(test_data))
    print('CNN with dropout:')
    print(CNN_Dropout_prediction(test_data))
    print('CNN with batch normalization:')
    print(CNN_BatchNromalized_prediction(test_data))