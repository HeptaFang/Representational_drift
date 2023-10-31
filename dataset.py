import os
import numpy as np
import matplotlib.pyplot as plt
from METAPARAMETERS import *


def main():
    # generate random encoding matrices
    position_encoding = np.random.normal(0, BIN_NUM ** -0.5, (BIN_NUM, 1, HIDDEN_NUM))
    timestamp_encoding = np.random.normal(0, SESSION_NUM ** -0.5, (1, SESSION_NUM, HIDDEN_NUM))
    projection = np.random.normal(0, (HIDDEN_NUM * SPARSENESS) ** -0.5, (HIDDEN_NUM, CELL_NUM))

    sparse_mask = np.random.choice([0, 1], (HIDDEN_NUM, CELL_NUM), p=[1 - SPARSENESS, SPARSENESS])
    projection = projection * sparse_mask

    # generate output
    noise_mul = np.random.normal(0, NOISE_LEVEL, (BIN_NUM, SESSION_NUM, CELL_NUM))
    noise_add = np.random.normal(0, NOISE_LEVEL, (BIN_NUM, SESSION_NUM, CELL_NUM))
    output_mul = (position_encoding * timestamp_encoding) @ projection + noise_mul
    output_add = (position_encoding + timestamp_encoding) @ projection + noise_add

    # generate dataset
    position = np.zeros((BIN_NUM * SESSION_NUM, BIN_NUM))
    timestamp = np.zeros((BIN_NUM * SESSION_NUM, SESSION_NUM))
    activity_mul = np.zeros((BIN_NUM * SESSION_NUM, CELL_NUM))
    activity_add = np.zeros((BIN_NUM * SESSION_NUM, CELL_NUM))
    for i in range(BIN_NUM):
        for j in range(SESSION_NUM):
            position[i * SESSION_NUM + j, i] = 1
            timestamp[i * SESSION_NUM + j, j] = 1
            activity_mul[i * SESSION_NUM + j] = output_mul[i, j]
            activity_add[i * SESSION_NUM + j] = output_add[i, j]

    # save dataset
    path = os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset')
    np.save(os.path.join(path, 'mul_position.npy'), position)
    np.save(os.path.join(path, 'add_position.npy'), position)
    np.save(os.path.join(path, 'mul_timestamp.npy'), timestamp)
    np.save(os.path.join(path, 'add_timestamp.npy'), timestamp)
    np.save(os.path.join(path, 'mul_activity.npy'), activity_mul)
    np.save(os.path.join(path, 'add_activity.npy'), activity_add)


if __name__ == '__main__':
    main()
