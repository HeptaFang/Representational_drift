import os
import numpy as np
import matplotlib.pyplot as plt
from METAPARAMETERS import *


def pooling_dataset(plot=False):
    dataset_names = [f'mouse{i + 1}' for i in range(5)]
    # dataset_names = ['mul', 'add']
    pooling_size_position = 3
    pooling_size_timestamp = 1
    pooling_method = np.nanmean  # 'ave' , 'max', 'min', 'sum'

    for dataset in dataset_names:
        full_dataset_name = f'{dataset}'
        if SELECTED:
            position = np.load(
                os.path.join(GLOBAL_PATH, 'dataset', f'{full_dataset_name}_position_selected.npy'))
            timestamp = np.load(
                os.path.join(GLOBAL_PATH, 'dataset', f'{full_dataset_name}_timestamp_selected.npy'))
            activity = np.load(
                os.path.join(GLOBAL_PATH, 'dataset', f'{full_dataset_name}_activity_selected.npy'))
        else:
            position = np.load(
                os.path.join(GLOBAL_PATH, 'dataset', f'{full_dataset_name}_position.npy'))
            timestamp = np.load(
                os.path.join(GLOBAL_PATH, 'dataset', f'{full_dataset_name}_timestamp.npy'))
            activity = np.load(
                os.path.join(GLOBAL_PATH, 'dataset', f'{full_dataset_name}_activity.npy'))
            
        _, bin_num = position.shape
        _, session_num = timestamp.shape
        _, cell_num = activity.shape
               
        # pooling
        position_pooled = np.zeros(
            (position.shape[0] // (pooling_size_position * pooling_size_timestamp), bin_num // pooling_size_position))
        timestamp_pooled = np.zeros((timestamp.shape[0] // (pooling_size_position * pooling_size_timestamp),
                                     session_num // pooling_size_timestamp))
        activity_pooled = np.zeros((bin_num // pooling_size_position, session_num // pooling_size_timestamp, cell_num))

        for i in range(bin_num // pooling_size_position):
            for j in range(session_num // pooling_size_timestamp):
                position_pooled[i * (session_num // pooling_size_position) + j, i] = 1
                timestamp_pooled[i * (session_num // pooling_size_timestamp) + j, j] = 1

                # pooling activity
                activity_reshaped = activity.reshape((bin_num, session_num, cell_num))

                activity_pooled[i, j, :] = pooling_method(
                    activity_reshaped[i * pooling_size_position:(i + 1) * pooling_size_position,
                    j * pooling_size_timestamp:(j + 1) * pooling_size_timestamp, :], axis=(0, 1))
        activity_pooled = activity_pooled.reshape(
            ((bin_num * session_num) // (pooling_size_position * pooling_size_timestamp), cell_num))

        # save
        if SELECTED:
            np.save(os.path.join(GLOBAL_PATH, 'dataset',
                                 f'{full_dataset_name}_position_selected_pooled.npy'), position_pooled)
            np.save(os.path.join(GLOBAL_PATH, 'dataset',
                                 f'{full_dataset_name}_timestamp_selected_pooled.npy'), timestamp_pooled)
            np.save(os.path.join(GLOBAL_PATH, 'dataset',
                                 f'{full_dataset_name}_activity_selected_pooled.npy'), activity_pooled)
        else:
            np.save(os.path.join(GLOBAL_PATH, 'dataset',
                                 f'{full_dataset_name}_position_pooled.npy'), position_pooled)
            np.save(os.path.join(GLOBAL_PATH, 'dataset',
                                 f'{full_dataset_name}_timestamp_pooled.npy'), timestamp_pooled)
            np.save(os.path.join(GLOBAL_PATH, 'dataset',
                                 f'{full_dataset_name}_activity_pooled.npy'), activity_pooled)

        print(position.shape, timestamp.shape, activity.shape)
        print(position_pooled.shape, timestamp_pooled.shape, activity_pooled.shape)

        if plot:
            print('plotting pooling samples')
            # plot
            for i in range(32):
                plt.figure(figsize=(10, 5), dpi=80)

                plt.subplot(1, 2, 1)
                plt.imshow(activity[:, i].reshape((bin_num, session_num)), aspect='auto')
                plt.title(f'original')
                plt.subplot(1, 2, 2)
                plt.imshow(activity_pooled[:, i].reshape(
                    (bin_num // pooling_size_position, session_num // pooling_size_timestamp)),
                    aspect='auto')
                plt.title(f'pooled')

                plt.suptitle(
                    f'cell {i}, pooling size: {(pooling_size_position, pooling_size_timestamp)}, method: {pooling_method.__name__}\n{dataset}')
                plt.savefig(os.path.join(GLOBAL_PATH, 'image', 'cells', f'{full_dataset_name}_cell{i}.png'))
                plt.close()


def main(plot=False):
    # for noise_level in NOISE_LEVELS:
    #     for bias in BIAS_LEVELS:
    #         print(f'noise level: {noise_level}, bias: {bias}')
    #         pooling_dataset(noise_level, bias, plot=plot)
    pooling_dataset(plot=plot)


if __name__ == '__main__':
    main(plot=True)
