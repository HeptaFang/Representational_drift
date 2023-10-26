import numpy as np
import matplotlib.pyplot as plt
import os
from METAPARAMETERS import *


def main():
    # tasks = ['mouse1', 'mouse2', 'mouse3', 'mouse4', 'mouse5']
    tasks = ['mouse1']
    # train_modes = ['Additive', 'Multiplicative', 'MultiWithLatent']
    train_modes = ['AddWithLatent', 'MultiWithLatent']
    # colors = {'Additive': 'r', 'Multiplicative': 'b', 'MultiWithLatent': 'g'}
    colors = {'AddWithLatent': 'r', 'MultiWithLatent': 'g'}
    max_epoch = 3000

    l1_levels = [0, 1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5]
    l2_levels = [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    smooth_levels = [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

    path = os.path.join(GLOBAL_PATH, 'analysis', 'regularization_search')
    fig_path = os.path.join(GLOBAL_PATH, 'image', 'regularization_search')
    file_list = os.listdir(path)

    # overview of all tasks
    for filename in file_list:
        print(filename)
        if 'train_loss' not in filename or 'AddWithLatent' not in filename:
            continue

        test_filename = filename.replace('train_loss', 'test_loss')
        train_multi_filename = filename.replace('AddWithLatent', 'MultiWithLatent')
        test_multi_filename = test_filename.replace('AddWithLatent', 'MultiWithLatent')
        train_data_add = np.load(os.path.join(path, filename))[:, 0]
        test_data_add = np.load(os.path.join(path, test_filename))
        train_data_multi = np.load(os.path.join(path, train_multi_filename))[:, 0]
        test_data_multi = np.load(os.path.join(path, test_multi_filename))
        plt.clf()
        plt.plot(train_data_add, label=f'Additive train', color='r', linestyle='solid')
        plt.plot(test_data_add, label=f'Additive test', color='r', linestyle='dot')
        plt.plot(train_data_multi, label=f'Multiplicative train', color='r', linestyle='solid')
        plt.plot(test_data_multi, label=f'Multiplicative test', color='r', linestyle='dot')
        plt.legend()
        plt.savefig(os.path.join(fig_path,
                                 filename.replace('.npy', '.png').replace('train_loss_', '').replace('_AddWithLatent',
                                                                                                     '')))


if __name__ == '__main__':
    main()
