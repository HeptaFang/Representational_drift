import numpy as np
import os

from METAPARAMETERS import *


def additive_decompose(matrix, keep_shape=False):
    m, n = matrix.shape
    a = np.mean(matrix, axis=1) - np.mean(matrix)
    b = np.mean(matrix, axis=0) - np.mean(matrix)
    c = np.reshape(np.mean(matrix), (1, 1))
    if keep_shape:
        a = np.reshape(a, (m, 1))
        b = np.reshape(b, (1, n))
        c = c * np.ones((m, n))

    return a, b, c


def multiplicative_decompose(matrix, keep_shape=False):
    m, n = matrix.shape
    a = np.mean(matrix, axis=1)
    if np.mean(a) == 0:
        b = np.zeros(n)
    else:
        b = np.mean(matrix, axis=0) / np.mean(a)
    if keep_shape:
        a = np.reshape(a, (m, 1))
        b = np.reshape(b, (1, n))

    return a, b


def main():
    # for noise_level in NOISE_LEVELS:
    #     for bias in BIAS_LEVELS:
    #         print()
    #         print(f'noise_level: {noise_level}, bias: {bias}')
    #         print('original data')
    #         path = os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset')
    #         activity_mul = np.load(os.path.join(path, f'mul_{noise_level:.1f}_{bias:.1f}_activity.npy'))
    #         # activity_mul = activity_mul.reshape((BIN_NUM, SESSION_NUM, CELL_NUM))
    #         activity_add = np.load(os.path.join(path, f'add_{noise_level:.1f}_{bias:.1f}_activity.npy'))
    #         # activity_add = activity_add.reshape((BIN_NUM, SESSION_NUM, CELL_NUM))
    #
    #         non_zero_count_mul = np.sum(activity_mul != 0)
    #         non_zero_count_add = np.sum(activity_add != 0)
    #
    #         print(f'mul: {non_zero_count_mul}, add: {non_zero_count_add}, size: {activity_mul.size}')
    #         print(f'ratio: {non_zero_count_mul / activity_mul.size:.3f}, {non_zero_count_add / activity_add.size:.3f}')
    #
    #         # pooled data
    #         print('pooled data')
    #         path = os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset')
    #         activity_mul = np.load(os.path.join(path, f'mul_{noise_level:.1f}_{bias:.1f}_activity_pooled.npy'))
    #         # activity_mul = activity_mul.reshape((BIN_NUM, SESSION_NUM, CELL_NUM))
    #         activity_add = np.load(os.path.join(path, f'add_{noise_level:.1f}_{bias:.1f}_activity_pooled.npy'))
    #         # activity_add = activity_add.reshape((BIN_NUM, SESSION_NUM, CELL_NUM))
    #
    #         non_zero_count_mul = np.sum(activity_mul != 0)
    #         non_zero_count_add = np.sum(activity_add != 0)
    #
    #         print(f'mul: {non_zero_count_mul}, add: {non_zero_count_add}, size: {activity_mul.size}')
    #         print(f'ratio: {non_zero_count_mul / activity_mul.size:.3f}, {non_zero_count_add / activity_add.size:.3f}')

    # count the mouse data
    threshold = 5e-2
    for i in range(5):
        print()
        print(f'mouse {i + 1}')
        path = os.path.join(GLOBAL_PATH, 'dataset')
        activity = np.load(os.path.join(path, f'mouse{i + 1}_activity.npy'))
        activity_selected = np.load(os.path.join(path, f'mouse{i + 1}_activity_selected.npy'))
        activity_pooled = np.load(os.path.join(path, f'mouse{i + 1}_activity_selected_pooled.npy'))
        trial_num, cell_num = activity.shape
        trial_num_selected, cell_num_selected = activity_selected.shape
        trial_num_pooled, cell_num_pooled = activity_pooled.shape

        non_zero_count = np.sum(activity > threshold * np.nanmax(activity))
        non_zero_count_selected = np.sum(activity_selected > threshold * np.nanmax(activity_selected))
        non_zero_count_pooled = np.sum(activity_pooled > threshold * np.nanmax(activity_pooled))

        print(
            f'non_zero_count: {non_zero_count}/{trial_num * cell_num}, selected: {non_zero_count_selected}/{trial_num_selected * cell_num_selected}, pooled: {non_zero_count_pooled}/{trial_num_pooled * cell_num_pooled}')
        print(
            f'ratio: {non_zero_count / (trial_num * cell_num):.3f}, selected: {non_zero_count_selected / (trial_num_selected * cell_num_selected):.3f}, pooled: {non_zero_count_pooled / (trial_num_pooled * cell_num_pooled):.3f}')

if __name__ == '__main__':
    main()
