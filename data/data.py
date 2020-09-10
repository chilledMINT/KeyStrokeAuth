import csv
import os
import numpy as np


def collectLabelledData(rawDataFileName, userId) -> list:
    """Accepts the name of the raw data CSV fild and the ID of the valid user,
    labels the row data correctly, and returns the resuntalt data as a list of
    lists

    Args:
        rawDataFileName (string): Name of the raw Data File
        userId (int): Id of current user

    Returns:
        list: Returns a list of lists containing data marked as valid and invalid
    """
    labelledData = []

    # operate on raw data file
    with open(rawDataFileName, 'r') as rawDataFile:
        rawData = csv.reader(rawDataFile, delimiter=',', lineterminator='\n')

        # iteratively process data
        for rawRow in rawData:
            labelledRow = []

            # label data
            if int(rawRow[0]) == userId:
                labelledRow.append(1)
            else:
                labelledRow.append(0)

            # copy keystroke data
            labelledRow += rawRow[3:]
            labelledData.append(labelledRow)
    return labelledData


def writeCSVData(targetFileName, data) -> None:
    """Accepts a target CSV file name and the data as a list of lists, and
    writes the data contents row-by-row to the target file

    Args:
        targetFileName (string): Name of the file where you want to write data
        data (list of lists): This the a list of lists of data as valid vs
        invalid
    """

    # operate on the target data file
    with open(targetFileName, 'w') as targetFile:
        target = csv.reader(targetFile, delimiter=',', lineterminator='\n')

        # iteratively write data
        for row in data:
            target.writerow(row)


def writeCSVData(targetFileName, data) -> None:
    """Accepts a target CSV file name and the data as a list of lists, and
    writes the data contents row-by-row to the target file.

    Args:
        targetFileName (string): Name of the file where you want to write data
        data (list): A list of lists containign data
    """
    with open(targetFileName, 'w') as targetFile:
        target = csv.writer(targetFile, delimiter=',', lineterminator='\n')

        # iteratively write data
        for row in data:
            target.writerow(row)


def processLabelledData(labelledData, validDataFileName,
                        invalidDataFileName) -> None:
    """Accepts the labelled data, as well as the valid and invalid processed
    data file names, collects the labelled data, processes this data, and writes
    the valid and invalid output

    Args:
        labelledData (list): List containing valid vs invalid data
        validDataFileName (string) The name of valid data file.
        invalidDataFileName (string): The name of invalid data file.
    """
    validLabelledData = []
    invalidLabelledData = []

    # convert data to matrix
    dataMatrix = np.array(labelledData, dtype=np.float32)
    dataMean = dataMatrix.mean(axis=0)[1:]
    dataStd = dataMatrix.std(axis=0)[1:]

    # split labelled data
    for data in labelledData:
        dataArr = np.array(data[1:], dtype=np.float32)

        # normalize the data
        dataArr = (dataArr - dataMean) / dataStd

        # sort the data
        if data[0] == 1:
            validLabelledData.append(dataArr.tolist())
        else:
            invalidLabelledData.append(dataArr.tolist())

    # write data
    writeCSVData(validDataFileName, validLabelledData)
    writeCSVData(invalidDataFileName, invalidLabelledData)


def buildData(validId) -> None:
    """Accepts the ID for the valid user and builds the valid and invalid
    dataset.

    Args:
        validId (int): The id of current(valid) user.
    """
    try:
        os.remove('data/processed-valid-data.csv')
        os.remove('data/processed-invalid-data.csv')
    except:
        pass

    labelledData = collectLabelledData('data/raw-data.csv', validId)
    processLabelledData(labelledData, 'data/processed-valid-data.csv',
                        'data/processed-invalid-data.csv')