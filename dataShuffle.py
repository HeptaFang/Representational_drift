import os

import numpy as np

from METAPARAMETERS import *


def main(noise_level, bias):
    # dataset_names = [f'mouse{i + 1}' for i in range(5)]
    dataset_names = ['mul', 'add']
    feature_names = ['position', 'timestamp', 'activity']
    selected = False

    for dataset in dataset_names:
        full_dataset_name = f'{dataset}_{noise_level:.1f}_{bias:.1f}'
        if selected:
            files = [np.load(
                os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset', f'{full_dataset_name}_{feature_name}_selected.npy'))
                for feature_name in feature_names]
        else:
            files = [
                np.load(os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset', f'{full_dataset_name}_{feature_name}.npy'))
                for feature_name in feature_names]

        file_length = files[0].shape[0]
        perm = np.random.permutation(file_length)
        print(full_dataset_name, file_length, perm)
        shuffled_files = [file[perm] for file in files]

        for i in range(len(feature_names)):
            if selected:
                np.save(os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset',
                                     f'{full_dataset_name}_{feature_names[i]}_shuffled_selected.npy'), shuffled_files[i])
            else:
                np.save(os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset',
                                     f'{full_dataset_name}_{feature_names[i]}_shuffled.npy'), shuffled_files[i])


if __name__ == '__main__':
    np.random.seed(11308)
    for noise_level in NOISE_LEVELS:
        for bias in BIAS_LEVELS:
            main(noise_level, bias)
