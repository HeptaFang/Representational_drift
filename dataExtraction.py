from mat73 import loadmat
import os
import re
import numpy as np


def get_session_path():
    root_path = 'D:\\ExpData\\Binding\\Datasets\\DynamicReorganization\\DataFiles_Driscoll'
    mouse_num = 5
    data_re = re.compile('m[0-9][0-9]_s[0-9][0-9]\\.mat')

    available_sessions = []
    walker = os.walk(root_path)
    for folder in walker:
        for file in folder[2]:
            if data_re.match(file):
                available_sessions.append((folder[0], file))

    sessions = [[] for i in range(5)]
    for session in available_sessions:
        mouse_i = int(session[1][1:3])
        session_i = int(session[1][5:7])
        sessions[mouse_i - 1].append((session_i, f'{session[0]}\\{session[1]}'))

    sessions = [sorted(s, key=lambda x: x[0]) for s in sessions]

    # session number for each mouse and the path
    for i in range(5):
        print(len(sessions[i]), sessions[i])

    return sessions


def main():
    session_path = get_session_path()
    split_cue = True
    filter_correct = True
    for mouse_i in range(5):
        all_activity = []
        all_confidence = []
        for session in session_path[mouse_i]:
            # load data
            mat = loadmat(session[1])
            activity = mat['session_obj']['trials']['spData']
            correct = mat['session_obj']['trials']['correct']
            cue = mat['session_obj']['trials']['trialType']
            confidence = mat['session_obj']['confidenceLabel']

            # # to remove: nan clarify
            # conf_nan = np.isnan(confidence)
            # print(activity[:, :, conf_nan])

            # preprocess
            trial_num, bin_num, cell_num = activity.shape
            activity_filtered = np.zeros((bin_num, 4, cell_num))
            for i in range(4):
                activity_filtered[:, i, :] = np.nanmean(activity[(correct == 1) & (cue == i + 1), :, :], 0)
                print(f'  {session}.{i + 1}: {np.sum(cue == i + 1)}')

            # save
            all_confidence.append(confidence)
            all_activity.append(activity_filtered)
            print(all_activity[-1].shape)
        activity_np = np.array(all_activity)
        confidence_np = np.array(all_confidence)
        print(activity_np.shape)
        np.save(f'dataset/mouse{mouse_i + 1}_activity_raw.npy', activity_np)
        np.save(f'dataset/mouse{mouse_i + 1}_confidence.npy', confidence_np)


if __name__ == '__main__':
    main()
