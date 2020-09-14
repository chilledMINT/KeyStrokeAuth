import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def deep_neural_network_model(X_train, y_train, verbose=False):
    """Accepts the X_train, y_train. Defines, Compiles and Trains a Deep Neural
    Network model. Returns the trained model

    Args:
         X_train (pandas dataframe): The dataframe used for prediction.
         y_train (list): The list containing valid vs invalid user for
         corresponding input.

    Returns:
        Returns a trained keras Sequential model.
    """
    input_dim, output_dim = 31, 1

    # define hyper-parameters
    batch_size = 32
    num_epochs = 30
    layer_zero_size = 256
    layer_one_size = 128
    layer_two_size = 128
    layer_three_size = 64
    layer_four_size = 32
    layer_five_size = 16

    # build the model
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[31,1]))
    model.add(keras.layers.Dense(layer_zero_size, input_dim=input_dim,
                           activation='selu', kernel_initializer='he_normal'))
    model.add(keras.layers.Dense(layer_one_size, activation='selu',
                                 kernel_initializer='he_normal'))
    model.add(keras.layers.Dense(layer_two_size, activation='selu',
                                 kernel_initializer='he_normal'))
    model.add(keras.layers.Dense(layer_three_size, activation='selu',
                                 kernel_initializer='he_normal'))
    model.add(keras.layers.Dense(layer_four_size, activation='selu',
                                 kernel_initializer='he_normal'))
    model.add(keras.layers.Dense(layer_five_size, activation='selu',
                                 kernel_initializer='he_normal'))
    model.add(keras.layers.Dense(output_dim, activation='sigmoid'))

    # add callback for early stopping
    callback = keras.callbacks.EarlyStopping(patience=5,
                                             restore_best_weights=True)
    # compile model
    optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
    model.compile(loss='poisson', optimizer=optimizer,
                  metrics='accuracy')

    # train model
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs,
                        callbacks=[callback], validation_split=0.15)

    # plot the progress
    if verbose:
        pd.DataFrame(history.history).plot(figsize=(12, 9))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
        plt.show()

    return model

def evaluate_model(model, X_test, y_test) -> tuple:
    """Accepts trained model and test set. Evaluates the model on the test set.
    Returns tuple containing accuracy, precision, recall and f1 score.

    Args:
         model (keras.models..) : A keras deep learning model.
         X_test (pandas dataframe): The X_train dataframe.
         y_test (list): Valid vs invalid user for corresponding data.
    """

    y_pred = model.predict(X_test)
    y_pred = y_pred > 0.5
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test)
    recall = recall_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test)

    return accuracy, precision, recall, f1


def run_model(X_train, X_test, y_train, y_test) -> tuple:
    model = deep_neural_network_model(X_train, y_train)
    result = list(evaluate_model(model, X_test, y_test))
    return result