import csv
import numpy as np
from sklearn.model_selection import train_test_split


def split_data_to_xy(user_id, test_size=0.15, raw_data_file_name='raw-data.csv') -> tuple:
    """Accepts the user id and returns the X and y values.

    Args:
         user_id (int): The user id of current valid user.
         test_size (float): The fraction of data used for testing.
         raw_data_file_name (string): The name of raw-data-file to extract data from.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train and y_test
    """

    X = []
    y = []

    # prepare valid/invalid output
    with open(raw_data_file_name, 'r') as raw_data_file:
        raw_data = csv.reader(raw_data_file, delimiter=',', lineterminator='\n')

        for row in raw_data:
            if int(row[0]) == user_id:
                y.append(1)
            else:
                y.append(0)
            cur_data = row[3:]
            X.append(cur_data)

    info_matrix = np.array(X, dtype=np.float)
    info_min = info_matrix.min(axis=0)
    info_max = info_matrix.max(axis=0)

    x = []
    for info in info_matrix:
        info_arr = np.array(info, dtype=np.float)

        # normalize the data
        info_arr = (info_arr - info_min) / (info_max - info_min + np.finfo(float).eps)
        x.append(info_arr)

    x = np.array(x)
    # splitting dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def build_data(user_id) -> tuple:
    """Builds and returns the data according to the user_id.

    Args:
         user_id (int): The unique id of certain user.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train and y_test
    """
    return split_data_to_xy(user_id=user_id, test_size=0.15, raw_data_file_name='raw-data.csv')