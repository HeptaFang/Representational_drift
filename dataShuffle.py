import numpy as np


def main():
    np.random.seed(11308)
    dataset_names = [f'mouse{i + 1}' for i in range(5)]
    feature_names = ['position', 'timestamp', 'activity']

    for dataset in dataset_names:
        files = [np.load(f'dataset\\{dataset}_{feature}_selected.npy') for feature in feature_names]
        file_length = files[0].shape[0]
        perm = np.random.permutation(file_length)
        print(dataset, file_length, perm)
        shuffled_files = [file[perm] for file in files]
        for i in range(len(feature_names)):
            np.save(f'dataset\\{dataset}_{feature_names[i]}_shuffled_selected.npy', shuffled_files[i])


if __name__ == '__main__':
    main()
