import os

import numpy as np

from METAPARAMETERS import *


def main():
    np.random.seed(11308)
    # dataset_names = [f'mouse{i + 1}' for i in range(5)]
    dataset_names = ['mul', 'add']
    feature_names = ['position', 'timestamp', 'activity']
    selected = False

    for dataset in dataset_names:
        if selected:
            files = [np.load(
                os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset', f'{dataset}_{feature_name}_selected.npy'))
                for feature_name in feature_names]
        else:
            files = [
                np.load(os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset', f'{dataset}_{feature_name}.npy'))
                for feature_name in feature_names]

        file_length = files[0].shape[0]
        perm = np.random.permutation(file_length)
        print(dataset, file_length, perm)
        shuffled_files = [file[perm] for file in files]

        for i in range(len(feature_names)):
            if selected:
                np.save(os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset',
                                     f'{dataset}_{feature_names[i]}_shuffled_selected.npy'), shuffled_files[i])
            else:
                np.save(os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset',
                                     f'{dataset}_{feature_names[i]}_shuffled.npy'), shuffled_files[i])


if __name__ == '__main__':
    main()
