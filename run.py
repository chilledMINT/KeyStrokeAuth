from data.data import build_data
from model.model import run_model
import numpy as np
import time
import logging

if __name__ == '__main__':
    logging.basicConfig(filename='training-log.log', level=logging.DEBUG, format='%(asctime)s %(message)s', force=True)

    verbose = True

    # take start time
    start = int(round(time.time() * 1000))

    # test each user as valid
    results = []
    for user in range(51):
        logging.info('Testing for user: {}'.format(user))
        X_train, X_test, y_train, y_test = build_data(user_id=user)
        result = list(run_model(X_train, X_test, y_train, y_test))
        results.append(result)

        # log interim results
        if verbose:
            logging.info('Accuracy: {}'.format(result[0]))
            logging.info('Recall: {}'.format(result[1]))
            logging.info('Precision: {}'.format(result[2]))
            logging.info('F1: {}'.format(result[3]))

    # calculate final result
    results = np.array(results, dtype=np.float)
    means = np.mean(results, axis=0)
    stds = np.std(results, axis=0)

    # log final results
    logging.info('Final accuracy mean: {}'.format(means[0]))
    logging.info('Final accuracy std: {}'.format(stds[0]))
    logging.info('Final recall mean: {}'.format(means[1]))
    logging.info('Final recall std: {}'.format(stds[1]))
    logging.info('Final precision mean: {}'.format(means[2]))
    logging.info('Final precision std: {}'.format(stds[2]))
    logging.info('Final F1 mean: {}'.format(means[3]))
    logging.info('Final F1 std: {}'.format(stds[3]))

    # calculate and log total time required to train
    end = int(round(time.time() * 1000))
    logging.info('Total time expended: {}'.format(end - start))
