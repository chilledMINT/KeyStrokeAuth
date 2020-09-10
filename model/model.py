import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from scipy.special import softmax


def prepareDataset(validDataFileName,
                   invalidDataFileName,
                   dataSplit,
                   invalidCount=1000,
                   useValidCount=False) -> tuple:
    """Accepts the file name for the processed valid and invalid data, as well
    as the train-test split values, loads the data, and splits the data.

    Args:
        validDataFileName (string): The name of valid data file.
        invalidDataFileName (string): The name of invalid data file.
        dataSplit (string): Train and test split ratio.
        invalidCount (int, optional): [description]. Defaults to 1000.
        useValidCount (bool, optional): [description]. Defaults to False.

    Returns:
        tuple: returns tuple containing trian data and test data
    """
    assert len(dataSplit) == 2 and dataSplit[0] + dataSplit[1] == 1.0

    # load valid data
    validDataframe = pd.read_csv(validDataFileName)
    validDataframe['label'] = 1
    cols = validDataframe.colums.tolist()
    cols.insert(0, cols.pop(cols.index('label')))
    validDataframe = validDataframe.reindex(columns=cols)
    validDataframe = np.concatenate(
        (validDataframe.as_matrix(), validDataframe.as_matrix(),
         validDataframe.as_matrix(), validDataframe.as_matrix()))

    # load invalid data
    invalidDataframe = pd.read_csv(invalidDataFileName)
    invalidDataframe['label'] = 0
    cols = invalidDataframe.columns.tolist()
    invalidDataframe = invalidDataframe.reindex(columns=cols)
    invalidDataframe = invalidDataframe.as_matrix()
    np.random.shuffle(invalidDataframe)
    if useValidCount:
        invalidDataframe = invalidDataframe[:validDataframe.shape[0], :]
    else:
        invalidDataframe = invalidDataframe[:invalidCount, :]

    # create train/test split
    validTrainData = validDataframe[:int(validDataframe.shape[0] *
                                         dataSplit[0]), :]
    validTestData = validDataframe[int(validDataframe.shape[0] *
                                       dataSplit[0]):, :]
    invalidTrainData = invalidDataframe[:int(invalidDataframe.shape[0] *
                                             dataSplit[0]), :]
    invalidTestData = invalidDataframe[int(invalidDataframe.shape[0] *
                                           dataSplit[0]):, :]
    trainData = np.vstack((validTrainData, invalidTrainData))
    testData = np.vstack((validTestData, invalidTestData))

    # return scrambled data
    np.random.shuffle(trainData)
    np.random.shuffle(testData)
    return (trainData, testData)


def sliceDataset(data) -> tuple:
    """Accepts a tuple of train and test data matrices and organizes the data 
    into train/test data and train/test labels.

    Args:
        data (tuple): A tuple of train/test data

    Returns:
        tuple: returns X_train, X_test, y_train, y_test
    """
    trainData = data[0]
    testData = data[1]

    # organize between labels and data
    xTrain = trainData[:, 1:]
    xTest = testData[:, 1:]
    yTrain = trainData[:, 0]
    yTest = testData[:, 0]

    # one-hot encode data
    output_dim = 2
    yTrain = keras.utils.to_categorical(yTrain, output_dim)
    yTest = keras.utils.to_categorical(yTest, output_dim)
    return (xTrain, xTest, yTrain, yTest)

def deepNeuralNetworkModel(data, verbose=False):
    """Accepts the train/test and data and train/test labels, trains, and
    returns a trained deep neural network model.

    Args:
        data (tuple): xTrain, xTest, yTrain, yTest
        verbose (bool, optional): [description]. Defaults to False.
    """
    xTrain, xTest, yTrain, yTest = data
    input_dim, output_dim = xTrain.shape[1], 2

    # define hyperparameters
    batchSize = 32
    numEpochs = 30
    layer_zero_size = 256
    layer_one_size = 128
    layer_two_size = 128
    layer_three_size = 64
    layer_four_size = 32
    layer_five_size = 16

    # bulid a model
    model = keras.models.Sequential()
    model.add(keras.Dense(layer_zero_size, input_dim=input_dim, activation='selu',
	kernel_initializer='he_normal'))
    model.add(keras.layers.Dropout(0.01))
    model.add(keras.layers.Dense(layer_one_size, activation='selu',
	kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(layer_two_size, activaton='selu',
	kernel_initializer='he_normal'))
    model.add(keras.layers.Dropout(0.01))
    model.add(keras.layers.Dense(layer_three_size, activation='selu',
	kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(layer_four_size, activation='selu',
	kernel_initializer='he_normal'))
    model.add(keras.layers.Dropout(0.01))
    model.add(keras.layers.Dense(layer_five_size, activation='selu',
	kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(output_dim, activation='softmax'))

    # add callback for early-stopping
    stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                         verbose=verbose, mode='min')

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')

    # train model
    history = model.fit(xTrain, yTrain, batch_size=batchSize, epochs=numEpochs,
                        callbacks=[stop], verbose=verbose, validation_split=0.1)

    # plot the progress
    if verbose:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='best')
        plt.show()

    return model


def evaluateModel(model, data, verbose=False) -> tuple:
    """Accepts a keras model and datasets and evaluates custom metrics over the 
    model.
    
    Args:
        model(keras.models...): A keras model
        data (tuple) : xTrain, xTest, yTrain, yTest
        verbose(bool, optional): Defaults to false
    
    Returns:
        tuple: accuracy, recall, precision, F1
    """
    _, xTest, _, yTest = data
    true_positives, true_negatives = 0, 0
    false_positives, false_negatives = 0, 0

    # compute custom metrics
    prediction = model.predict(xTest)
    results = softmax(prediction, axis=0)
    for i in range(results.shape[0]):
        y = int(yTest[i][1])
        yHat = int(results[i][1] > results[i][0])
        if (y and yHat):
            true_positives += 1
        elif (not y and not yHat):
            true_negatives += 1
        elif (y and not yHat):
            false_positives += 1
        else:
            false_negatives += 1

    # calculate metrics
    accuracy = (true_positives + true_negatives) / (true_positives +
                                                    true_negatives +
                                                    false_positives +
                                                    false_negatives)
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    F1 = (2 * true_positives) / (2 * true_positives + false_positives +
                                 false_negatives)

    # print metric results
    if verbose:
        print("\nTrue positives: {} / {}".format(true_positives, true_positives
                                                 + false_negatives))
        print("True negatives: {} / {}".format(true_negatives, true_negatives +
                                               false_positives))
        print("False positives: {} / {}".format(false_positives, true_negatives
                                                + false_positives))
        print("False negatives: {} / {}".format(false_negatives, true_positives
                                                + false_negatives))
        print("\nAccuracy: {}".format(accuracy))
        print("Recall: {}".format(recall))
        print("Precision: {}".format(precision))
        print("F1: {}".format(F1))

    # return results
    return accuracy, recall, precision, F1


def runModel() -> tuple:
	"""Runs the model and retunrs the result metrics

	Returns:
		tuple: Result tuple containing (accuracy, recall, precision, F1)
	"""	
    data = prepareDataset('data/processed-valid-data.csv',
                          'data/processed-invalid-data.csv', (0.80, 0.20), 
						  useValidCount=True)
    data = sliceDataset(data)
    model = deepNeuralNetworkModel(data)
    result = evaluateModel(model, data)
    return result