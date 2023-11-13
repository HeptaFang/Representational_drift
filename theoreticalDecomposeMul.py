import numpy as np
import os
from matplotlib import pyplot as plt

from METAPARAMETERS import *


def multiplicative_decompose(matrix, keep_shape=False):
    m, n = matrix.shape
    a = np.random.normal(1, 0.1, m)
    b = np.random.normal(1, 0.1, n)
    err = np.zeros(100)
    for i in range(100):
        err[i] = np.linalg.norm(matrix - np.outer(a, b))
        a = np.mean(matrix, axis=1) / np.mean(b)
        b = np.mean(matrix, axis=0) / np.mean(a)
        # print(a, b)
    #
    # plt.plot(err)
    # plt.show()

    return a, b


def multi_with_latent_decompose(matrix, keep_shape=False):
    m, n = matrix.shape
    # err = np.linalg.norm(activity - w @ (a * b))
    a = np.random.normal(1, 0.1, m)
    b = np.random.normal(1, 0.1, n)
    w = np.random.normal(1, 0.1, (m, n))
    err = np.zeros(100)
    for i in range(100):
        err[i] = np.linalg.norm(matrix - w @ np.outer(a,  b))
        a = np.mean(matrix, axis=1) / np.mean(b)
        b = np.mean(matrix, axis=0) / np.mean(a)
        w = matrix / np.mean(np.outer(a, b))
        # print(a, b)

    plt.plot(err)
    plt.show()
    return a, b, w


def main():
    snr = 10000
    # a_original = np.random.normal(1, 0.1, 16)
    # b_original = np.random.normal(1, 0.1, 16)

    a_original = np.random.normal(0, 0.1, 16)
    b_original = np.random.normal(0, 0.1, 16)
    w_original = np.random.normal(0, 0.1, (16, 16))
    noise = np.random.normal(0, 0.1 / snr, (16, 16))
    noise = 0
    mat = w_original @ np.outer(a_original, b_original) + noise
    a, b, w = multi_with_latent_decompose(mat)
    err = np.linalg.norm(mat - np.outer(a, b))
    print(err)


if __name__ == '__main__':
    main()
