import numpy as np


def main():
    # wide type to long type transformation
    # next time use pandas.DataFrame
    for mouse_i in range(5):
        # activity_raw = np.load(f'dataset\\mouse{mouse_i + 1}_activity_raw.npy')
        activity_raw = np.load(f'dataset\\mouse{mouse_i + 1}_activity_selected_raw.npy')
        # confidence = np.load(f'dataset\\mouse{mouse_i + 1}_confidence.npy')
        session_num, bin_num, cue_num, cell_num = activity_raw.shape

        total_size = session_num * bin_num * 2
        position = np.zeros((total_size, bin_num * 2))
        timestamp = np.zeros((total_size, session_num))
        activity = np.zeros((total_size, cell_num))

        for cue in range(2):
            for p in range(bin_num):
                for s in range(session_num):
                    i = cue * bin_num * session_num + p * session_num + s
                    position[i, p + cue * bin_num] = 1
                    timestamp[i, s] = 1
                    activity[i] = activity_raw[s, p, cue+1]

        amplify_multi = 1 / np.nanstd(activity)

        # print(amplify_multi)
        # return

        # np.save(f'dataset\\mouse{mouse_i + 1}_position.npy', position)
        # np.save(f'dataset\\mouse{mouse_i + 1}_timestamp.npy', timestamp)
        # np.save(f'dataset\\mouse{mouse_i + 1}_activity.npy', activity * amplify_multi)
        np.save(f'dataset\\mouse{mouse_i + 1}_position_selected.npy', position)
        np.save(f'dataset\\mouse{mouse_i + 1}_timestamp_selected.npy', timestamp)
        np.save(f'dataset\\mouse{mouse_i + 1}_activity_selected.npy', activity * amplify_multi)


if __name__ == '__main__':
    main()
