import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from model import load_model
import torch

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    task_name = 'mouse1'
    train_mode = 'MultiWithLatent'
    model_name = 'Binding'
    epoch = 5000
    conf_cmaplist = [(0.0, 0.0, 0.0), (0.2, 0.8, 0.2), (0.8, 0.8, 0.2), (0.8, 0.2, 0.2), (0.5, 0.5, 0.5)]
    component_cmap = np.random.random((32, 3))
    conf_cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', conf_cmaplist, len(conf_cmaplist))

    # load dataset
    position = np.load('dataset\\' + task_name + '_position_selected.npy')
    timestamp = np.load('dataset\\' + task_name + '_timestamp_selected.npy')
    activity = np.load('dataset\\' + task_name + '_activity_selected.npy')
    raw_activity = np.load('dataset\\' + task_name + '_activity_selected_raw.npy')
    confidence = np.load(f'dataset\\{task_name}_confidence_selected.npy')

    session_num, bin_num, _, cell_num = raw_activity.shape
    cue_num = 2
    print(raw_activity.shape)
    activity = activity.reshape((bin_num, session_num, cue_num, cell_num))
    confidence[np.isnan(confidence)] = 0

    # load model
    model, weight_dict = load_model(bin_num * cue_num, session_num, cell_num, train_mode, model_name, task_name, epoch,
                                    return_weight=True)
    position_tensor = torch.Tensor(position)
    timestamp_tensor = torch.Tensor(timestamp)
    activity_pred = model(position_tensor, timestamp_tensor).detach().numpy()
    activity_pred = activity_pred.reshape((bin_num, session_num, cue_num, cell_num))
    position_encoding = weight_dict['position_encoding.weight'].detach().cpu().numpy()
    timestamp_encoding = weight_dict['timestamp_encoding.weight'].detach().cpu().numpy()
    latent_encoding = weight_dict['latent_projection.weight'].detach().cpu().numpy()
    right_ind = np.argsort(np.argmax(position_encoding[:, :bin_num], 1))
    left_ind = np.argsort(np.argmax(position_encoding[:, bin_num:], 1))
    mix_ind = np.argsort(np.argmax(position_encoding[:, :], 1))

    # plot stuffs
    fig = plt.figure(figsize=[16, 16])
    for i in range(cell_num):
        plt.clf()
        ax_dict = fig.subplot_mosaic(
            [
                ['availability', 'actual_left', 'actual_right', 'pred_left', 'pred_right'],
                ['projection', 'projection', 'projection', 'proj_map_left', 'proj_map_right']
            ],
            width_ratios=[1, 10, 10, 10, 10]
        )
        ax_dict['availability'].imshow(confidence[:, i:i + 1], cmap=conf_cmap, vmin=-0.5, vmax=4.5)
        ax_dict['availability'].tick_params(bottom=False, left=True, labelbottom=False, labelleft=True)
        ax_dict['availability'].spines['top'].set_visible(False)
        ax_dict['availability'].spines['right'].set_visible(False)
        ax_dict['availability'].spines['left'].set_visible(False)
        ax_dict['availability'].spines['bottom'].set_visible(False)
        ax_dict['availability'].title.set_text('Availability')

        ax_dict['actual_left'].imshow(activity[:, :, 1, i].T)
        ax_dict['actual_left'].title.set_text('actual_left')
        ax_dict['actual_right'].imshow(activity[:, :, 0, i].T)
        ax_dict['actual_right'].title.set_text('actual_right')
        ax_dict['pred_left'].imshow(activity_pred[:, :, 1, i].T)
        ax_dict['pred_left'].title.set_text('pred_left')
        ax_dict['pred_right'].imshow(activity_pred[:, :, 0, i].T)
        ax_dict['pred_right'].title.set_text('pred_right')

        timed_encoding = timestamp_encoding[mix_ind, :].T * latent_encoding[i:i + 1, mix_ind]
        ax_dict['projection'].imshow(timed_encoding)
        ax_dict['projection'].title.set_text('Projection map')
        ax_dict['projection'].set_xlabel('latent No.')
        ax_dict['projection'].set_ylabel('day No.')

        ax_dict['proj_map_left'].imshow(timed_encoding @ position_encoding[mix_ind, bin_num:])
        ax_dict['proj_map_right'].imshow(timed_encoding @ position_encoding[mix_ind, :bin_num])

        # ax_dict['proj_map_left'].imshow(position_encoding[mix_ind, bin_num:])
        # ax_dict['proj_map_right'].imshow(position_encoding[mix_ind, :bin_num])

        # plt.show()
        fig.savefig(f'image/cell_activity/{i}.jpg')

        # another figure, showing the components of cell activity
        plt.clf()
        # # find the most active day
        # most_active = np.argmax(np.var(activity[:, :, 0, i], 1), 0)

        # find two days with the most different activity
        max_diff = 0
        chosen_day = None
        for j in range(session_num):
            for k in range(j + 1, session_num):
                if np.linalg.norm(activity_pred[:, j, 0, i] - activity_pred[:, k, 0, i]) > max_diff:
                    max_diff = np.linalg.norm(activity_pred[:, j, 0, i] - activity_pred[:, k, 0, i])
                    chosen_day = (j, k)

        # calculate the contribution of each component to the difference
        latent_components_diff = ((latent_encoding[i, :] * timestamp_encoding[:, chosen_day[0]]).reshape(
            (32, 1)) * position_encoding[:, :21]) - (
                                         (latent_encoding[i, :] * timestamp_encoding[:, chosen_day[1]]).reshape(
                                             (32, 1)) * position_encoding[:, :21])
        # print(latent_components.shape)
        active_components_idx = np.argsort(-np.mean(np.abs(latent_components_diff), 1))

        for day_idx in range(2):
            day = chosen_day[day_idx]
            plt.subplot(2, 2, day_idx + 1)
            plt.plot(activity[:, day, 0, i].T, color='blue', alpha=0.5, marker='o')
            plt.plot(activity_pred[:, day, 0, i].T, color='red', alpha=0.5, marker='o')
            latent_components = (latent_encoding[i, :] * timestamp_encoding[:, day]).reshape(
                (32, 1)) * position_encoding[:, :21]
            plt.legend(['actual', 'pred'])
            plt.title(f'cell {i}, day {day}')

            plt.subplot(2, 2, day_idx + 3)
            for j in range(5):
                plt.plot(latent_components[active_components_idx[j], :],
                         color=component_cmap[active_components_idx[j], :])

            plt.title(f'latent components')
        fig.savefig(f'image/components/{i}.jpg')


if __name__ == '__main__':
    main()
