import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def main():
    cmaplist = [(1.0, 1.0, 1.0), (0.2, 0.8, 0.2), (0.8, 0.8, 0.2), (0.8, 0.2, 0.2), (0.5, 0.5, 0.5)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, len(cmaplist))

    for mouse_i in range(5):
        activity_raw = np.load(f'dataset\\mouse{mouse_i + 1}_activity_raw.npy')
        confidence = np.load(f'dataset\\mouse{mouse_i + 1}_confidence.npy')
        session_num, bin_num, _, cell_num = activity_raw.shape
        print(activity_raw.shape, confidence.shape)

        nan_mask = np.all(~np.isnan(activity_raw), axis=1)
        # print(nan_mask.shape)
        nan_mask = nan_mask[:, 1:2, :]

        # # hyperparameter search. optimal: threshold=0.70
        for block_length in range(7, 15, 7):
            for threshold in np.arange(0.5, 1.0, 0.05):
                max_available = 0
                max_start = 0
                max_mask = None
                for t_start in range(session_num - block_length):
                    active_ratio = np.mean(nan_mask[t_start:t_start + block_length], axis=0)
                    # print(active_ratio.shape)
                    active_mask = active_ratio > threshold
                    active_count = np.sum(active_mask)

                    if active_count > max_available:
                        max_available = active_count
                        max_start = t_start
                        max_mask = active_mask
                print('threshold={:.2f}, {} {} {}'.format(threshold, block_length, max_available, max_start))

        # threshold = 0.70
        # block_length = 14
        # max_available = 0
        # max_start = 0
        # max_mask = None
        # for t_start in range(session_num - block_length):
        #     active_ratio = np.mean(nan_mask[t_start:t_start + block_length], axis=0)
        #     # print(active_ratio.shape)
        #     active_mask = active_ratio > threshold
        #     active_count = np.sum(active_mask)
        #
        #     if active_count > max_available:
        #         max_available = active_count
        #         max_start = t_start
        #         max_mask = active_mask
        #
        # selected_data = activity_raw[max_start:max_start + block_length, :, :, max_mask[0]]
        # print(
        #     f'selected: {selected_data.shape}, active ratio: {np.mean(nan_mask[max_start:max_start + block_length, :, max_mask[0]])}')
        # np.save(f'dataset\\mouse{mouse_i + 1}_activity_selected_raw.npy', selected_data)
        # np.save(f'dataset\\mouse{mouse_i + 1}_confidence_selected.npy',
        #         confidence[max_start:max_start + block_length, max_mask[0]])
        #


if __name__ == '__main__':
    main()
