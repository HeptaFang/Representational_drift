import os

import numpy as np

from METAPARAMETERS import *


def shuffle_dataset():
    dataset_names = [f'mouse{i + 1}' for i in range(5)]
    # dataset_names = ['mul', 'add']
    feature_names = ['position', 'timestamp', 'activity']

    data_suffix = ''
    if SELECTED:
        data_suffix += '_selected'
    if POOLED:
        data_suffix += '_pooled'

    for dataset in dataset_names:
        full_dataset_name = f'{dataset}'
        files = [np.load(
            os.path.join(GLOBAL_PATH, 'dataset',
                         f'{full_dataset_name}_{feature_name}{data_suffix}.npy'))
            for feature_name in feature_names]

        file_length = files[0].shape[0]
        perm = np.random.permutation(file_length)
        print(full_dataset_name, file_length, perm)
        shuffled_files = [file[perm] for file in files]

        for i in range(len(feature_names)):
            np.save(os.path.join(GLOBAL_PATH, 'dataset',
                                 f'{full_dataset_name}_{feature_names[i]}_shuffled{data_suffix}.npy'),
                    shuffled_files[i])


def main():
    np.random.seed(11308)
    shuffle_dataset()


if __name__ == '__main__':
    main()
