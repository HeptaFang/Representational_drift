import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from model import load_model
import torch

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    task_names = ['mouse1', 'mouse2', 'mouse3', 'mouse4', 'mouse5']
    # task_name = 'mouse1'
    # train_mode = 'MultiWithLatent'
    model_name = 'Binding'
    epoch = 1000

    for task_name in task_names:
        # load dataset
        position = np.load('dataset\\' + task_name + '_position_selected.npy')
        timestamp = np.load('dataset\\' + task_name + '_timestamp_selected.npy')
        activity = np.load('dataset\\' + task_name + '_activity_selected.npy')
        raw_activity = np.load('dataset\\' + task_name + '_activity_selected_raw.npy')

        session_num, bin_num, _, cell_num = raw_activity.shape
        cue_num = 2
        print(raw_activity.shape)
        activity = activity.reshape((bin_num, session_num, cue_num, cell_num))

        # load 3 models for comparison
        models = [load_model(bin_num * cue_num, session_num, cell_num, 'Additive', model_name, task_name, epoch,
                             reg=True),
                  load_model(bin_num * cue_num, session_num, cell_num, 'Multiplicative', model_name, task_name, epoch,
                             reg=True),
                  load_model(bin_num * cue_num, session_num, cell_num, 'MultiWithLatent', model_name, task_name, epoch,
                             reg=True)]

        position_tensor = torch.Tensor(position)
        timestamp_tensor = torch.Tensor(timestamp)
        activity_preds = []
        for i in range(3):
            model = models[i]
            activity_pred = model(position_tensor, timestamp_tensor).detach().numpy()
            activity_pred = activity_pred.reshape((bin_num, session_num, cue_num, cell_num))
            activity_pred[np.isnan(activity)] = np.nan
            activity_preds.append(activity_pred)

        # plot stuffs
        legends = ['Add', 'Mul', 'Mul+Latent']
        fig = plt.figure(figsize=[16, 6])
        for i in range(cell_num):
            plt.clf()
            ax_dict = fig.subplot_mosaic(
                [
                    ['actual_left', 'actual_right', 'pred_left_1', 'pred_right_1'],
                    ['pred_left_2', 'pred_right_2', 'pred_left_3', 'pred_right_3'],
                ]
            )
            global_vmin = 100000
            global_vmax = 0
            global_vmax = max(global_vmax, np.nanmax(activity[:, :, 0:2, i]))
            global_vmin = min(global_vmin, np.nanmin(activity[:, :, 0:2, i]))
            for j in range(3):
                global_vmax = max(global_vmax, np.nanmax(activity_preds[j][:, :, 0:2, i]))
                global_vmin = min(global_vmin, np.nanmin(activity_preds[j][:, :, 0:2, i]))

            img = ax_dict['actual_left'].imshow(activity[:, :, 1, i].T, vmin=global_vmin, vmax=global_vmax)
            ax_dict['actual_left'].title.set_text('actual left')
            ax_dict['actual_right'].imshow(activity[:, :, 0, i].T, vmin=global_vmin, vmax=global_vmax)
            ax_dict['actual_right'].title.set_text('actual right')
            for j in range(3):
                ax_dict[f'pred_left_{j + 1}'].imshow(activity_preds[j][:, :, 1, i].T, vmin=global_vmin, vmax=global_vmax)
                ax_dict[f'pred_left_{j + 1}'].title.set_text(f'{legends[j]} left')
                ax_dict[f'pred_right_{j + 1}'].imshow(activity_preds[j][:, :, 0, i].T, vmin=global_vmin, vmax=global_vmax)
                ax_dict[f'pred_right_{j + 1}'].title.set_text(f'{legends[j]} right')

            fig.colorbar(img, ax=ax_dict.values(), shrink=0.6)
            fig.suptitle(f'{task_name} cell {i}')
            fig.savefig(f'image/cell_activity/{task_name}_{i}.jpg')

            plt.clf()


if __name__ == '__main__':
    main()
