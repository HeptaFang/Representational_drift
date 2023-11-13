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
    for noise_level in NOISE_LEVELS:
        for bias in BIAS_LEVELS:
            print()
            print(f'noise_level: {noise_level}, bias: {bias}')
            path = os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset')
            activity_mul = np.load(os.path.join(path, f'mulnl_{noise_level:.1f}_{bias:.1f}_activity.npy'))
            activity_mul = activity_mul.reshape((BIN_NUM, SESSION_NUM, CELL_NUM))
            activity_add = np.load(os.path.join(path, f'add_{noise_level:.1f}_{bias:.1f}_activity.npy'))
            activity_add = activity_add.reshape((BIN_NUM, SESSION_NUM, CELL_NUM))

            mul_mul_error = 0
            mul_add_error = 0
            add_mul_error = 0
            add_add_error = 0
            for i in range(CELL_NUM):
                a_mul_mul, b_mul_mul = multiplicative_decompose(activity_mul[:, :, i], keep_shape=True)
                a_mul_add, b_mul_add = multiplicative_decompose(activity_add[:, :, i], keep_shape=True)
                a_add_mul, b_add_mul, c_add_mul = additive_decompose(activity_mul[:, :, i], keep_shape=True)
                a_add_add, b_add_add, c_add_add = additive_decompose(activity_add[:, :, i], keep_shape=True)

                mul_mul_error += np.linalg.norm(activity_mul[:, :, i] - a_mul_mul * b_mul_mul)
                mul_add_error += np.linalg.norm(activity_add[:, :, i] - a_mul_add * b_mul_add)
                add_mul_error += np.linalg.norm(activity_mul[:, :, i] - a_add_mul - b_add_mul - c_add_mul)
                add_add_error += np.linalg.norm(activity_add[:, :, i] - a_add_add - b_add_add - c_add_add)

            mul_mul_error /= CELL_NUM * np.sqrt(BIN_NUM * SESSION_NUM)
            mul_add_error /= CELL_NUM * np.sqrt(BIN_NUM * SESSION_NUM)
            add_mul_error /= CELL_NUM * np.sqrt(BIN_NUM * SESSION_NUM)
            add_add_error /= CELL_NUM * np.sqrt(BIN_NUM * SESSION_NUM)

            print(f'dec=mul, data=mul, error: {mul_mul_error}')
            print(f'dec=add, data=mul, error: {add_mul_error}')
            print(f'dec=mul, data=add, error: {mul_add_error}')
            print(f'dec=add, data=add, error: {add_add_error}')


if __name__ == '__main__':
    main()
