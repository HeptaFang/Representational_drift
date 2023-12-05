import os
import numpy as np
import matplotlib.pyplot as plt
from METAPARAMETERS import *


def pooling_dataset(noise_level, bias):
    # dataset_names = [f'mouse{i + 1}' for i in range(5)]
    dataset_names = ['mul', 'add']
    feature_names = ['position', 'timestamp', 'activity']
    selected = False
    pooling_size = 2
    pooling_method = np.mean  # 'ave' , 'max', 'min', 'sum'

    for dataset in dataset_names:
        full_dataset_name = f'{dataset}_{noise_level:.1f}_{bias:.1f}'
        if selected:
            position = np.load(
                os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset',
                             f'{full_dataset_name}_position_selected.npy'))
            timestamp = np.load(
                os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset',
                             f'{full_dataset_name}_timestamp_selected.npy'))
            activity = np.load(
                os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset',
                             f'{full_dataset_name}_activity_selected.npy'))
        else:
            position = np.load(
                os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset', f'{full_dataset_name}_position.npy'))
            timestamp = np.load(
                os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset', f'{full_dataset_name}_timestamp.npy'))
            activity = np.load(
                os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset', f'{full_dataset_name}_activity.npy'))

        # pooling
        position_pooled = np.zeros((position.shape[0] // (pooling_size ** 2), BIN_NUM // pooling_size))
        timestamp_pooled = np.zeros((timestamp.shape[0] // (pooling_size ** 2), SESSION_NUM // pooling_size))
        activity_pooled = np.zeros((BIN_NUM // pooling_size, SESSION_NUM // pooling_size, CELL_NUM))

        for i in range(BIN_NUM // pooling_size):
            for j in range(SESSION_NUM // pooling_size):
                position_pooled[i * (SESSION_NUM // pooling_size) + j, i] = 1
                timestamp_pooled[i * (SESSION_NUM // pooling_size) + j, j] = 1

                # pooling activity
                activity_reshaped = activity.reshape((BIN_NUM, SESSION_NUM, CELL_NUM))

                activity_pooled[i, j, :] = pooling_method(activity_reshaped[i * pooling_size:(i + 1) * pooling_size,
                                                          j * pooling_size:(j + 1) * pooling_size, :], axis=(0, 1))
        activity_pooled = activity_pooled.reshape(((BIN_NUM * SESSION_NUM) // (pooling_size ** 2), CELL_NUM))

        # save
        if selected:
            np.save(os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset',
                                 f'{full_dataset_name}_position_pooled_selected.npy'), position_pooled)
            np.save(os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset',
                                 f'{full_dataset_name}_timestamp_pooled_selected.npy'), timestamp_pooled)
            np.save(os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset',
                                 f'{full_dataset_name}_activity_pooled_selected.npy'), activity_pooled)
        else:
            np.save(os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset',
                                 f'{full_dataset_name}_position_pooled.npy'), position_pooled)
            np.save(os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset',
                                 f'{full_dataset_name}_timestamp_pooled.npy'), timestamp_pooled)
            np.save(os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset',
                                 f'{full_dataset_name}_activity_pooled.npy'), activity_pooled)

        print(position.shape, timestamp.shape, activity.shape)
        print(position_pooled.shape, timestamp_pooled.shape, activity_pooled.shape)

        # plot
        # for i in range(16):
        #     plt.figure(figsize=(10, 10), dpi=200)
        #     plt.imshow(activity[:, i].reshape((BIN_NUM, SESSION_NUM)), aspect='auto')
        #     plt.title(f'cell {i}')
        #     plt.savefig(
        #         os.path.join(GLOBAL_PATH, 'image', 'artificial_dataset', 'cells', f'{full_dataset_name}_cell{i}.png'))
        #     plt.close()
        #
        #     plt.figure(figsize=(10, 10), dpi=200)
        #     plt.imshow(activity_pooled[:, i].reshape((BIN_NUM // pooling_size, SESSION_NUM // pooling_size)),
        #                aspect='auto')
        #     plt.title(f'cell {i}')
        #     plt.savefig(os.path.join(GLOBAL_PATH, 'image', 'artificial_dataset', 'cells',
        #                              f'{full_dataset_name}_cell{i}_pooled.png'))
        #     plt.close()


def main():
    for noise_level in NOISE_LEVELS:
        for bias in BIAS_LEVELS:
            pooling_dataset(noise_level, bias)


if __name__ == '__main__':
    main()
