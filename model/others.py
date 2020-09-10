import tensorflow as tf
from tensorflow import keras


def logisticRegressionModel(data):
    """Accepts the train/test data and train/test labels, runs and returns a 
    trained logistic regression model.

    Args:
        data (tuple): xTrain, xTest, yTrain, yTest

    Returns:
        A trained logistic regression model
    """
    xTrain, xTest, yTrain, yTest = data
    input_dim, output_dim = xTrain.shape[1], 2

    # define hyperparameters
    batchSize = 48
    numEpochs = 50

    # build model
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(output_dim,
                           input_dim=input_dim,
                           activation='softmax'))

    # compile model
    model.summary()
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # train model
    model.fit(xTrain,
              yTrain,
              batch_size=batchSize,
              epochs=numEpochs,
              verbose=True,
              validation_data=(xTest, yTest))

    return model


def shallowNeuralNetworkModel(data):
    """Accepts the train/test data and train/test labels, runs, and returns a
    trained shallow neural network model

    Args:
        data (tuple): xTrain, xTest, yTrain, yTest

    Returns:
        A keras shallow neural network model
    """

    xTrain, xTest, yTrain, yTest = data
    input_dim, output_dim = xTrain.shape[1], 2

    # define hyperparameters
    batchSize = 64
    numEpochs = 50
    layer_one_size = 5

    # build model
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(layer_one_size,
                           input_dim=input_dim,
                           activation='tanh'))

    # compile model
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy',
                           keras.metrics.binary_true_positive()])

    # train model
    model.fit(xTrain,
              yTrain,
              batch_size=batchSize,
              epochs=numEpochs,
              verbose=True,
              validataion_data=(xTest, yTest))

    return model
