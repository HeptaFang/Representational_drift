import numpy as np
import matplotlib.pyplot as plt
import os
from METAPARAMETERS import *


def main():
    # tasks = ['mouse1', 'mouse2', 'mouse3', 'mouse4', 'mouse5']
    tasks = ['mouse1']
    # train_modes = ['Additive', 'Multiplicative', 'MultiWithLatent']
    train_modes = ['AddWithLatent', 'MultiWithLatent']
    # colors = {'Additive': 'r', 'Multiplicative': 'b', 'MultiWithLatent': 'g'}
    colors = {'AddWithLatent': 'r', 'MultiWithLatent': 'g'}
    max_epoch = 3000

    l1_levels = [0, 1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5]
    l2_levels = [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    smooth_levels = [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

    # l1_str = [f'{l1:.0e}' for l1 in l1_levels]
    # l2_str = [f'{l2:.0e}' for l2 in l2_levels]
    # smooth_str = [f'{smooth:.0e}' for smooth in smooth_levels]

    path = os.path.join(GLOBAL_PATH, 'analysis', 'regularization_search')
    fig_path = os.path.join(GLOBAL_PATH, 'image', 'regularization_search')
    file_list = os.listdir(path)

    grid_fig = plt.figure(figsize=(10, 10), dpi=100)
    training_curve_fig = plt.figure(figsize=(10, 10), dpi=100)

    # overview of all tasks
    for task_name in tasks:
        for l1_idx in range(len(l1_levels)):
            # overall grid plot
            # (l2, smooth, train_mode, train/test, epoch)
            all_loss = np.zeros((len(l2_levels), len(smooth_levels), len(train_modes), 2, max_epoch))

            for l2_idx in range(len(l2_levels)):
                for smooth_idx in range(len(smooth_levels)):
                    # basic plot for comparison
                    l1 = l1_levels[l1_idx]
                    l2 = l2_levels[l2_idx]
                    smooth = smooth_levels[smooth_idx]
                    reg_marker = f'{l1}_{l2}_{smooth}'
                    reg_marker_str = f'{l1:.0e}_{l2:.0e}_{smooth:.0e}'

                    training_curve_fig.clear()
                    ax = training_curve_fig.add_subplot(111)
                    for train_mode in train_modes:
                        train_filename = os.path.join(path,
                                                      f'train_loss_{task_name}_{train_mode}_{reg_marker}_{0}_{max_epoch}.npy')
                        test_filename = os.path.join(path,
                                                     f'test_loss_{task_name}_{train_mode}_{reg_marker}_{0}_{max_epoch}.npy')
                        try:
                            train_loss = np.load(train_filename)[:, 0]
                            test_loss = np.load(test_filename)
                            all_loss[l2_idx, smooth_idx, train_modes.index(train_mode), 0, :] = train_loss
                            all_loss[l2_idx, smooth_idx, train_modes.index(train_mode), 1, :] = test_loss
                            ax.plot(train_loss, label=f'{train_mode} train', color=colors[train_mode],
                                    linestyle='solid')
                            ax.plot(test_loss, label=f'{train_mode} test', color=colors[train_mode], linestyle='dotted')
                        except FileNotFoundError:
                            continue
                    ax.set_title(f'{task_name} {reg_marker_str}')
                    ax.set_xlabel('epoch')
                    ax.set_ylabel('loss')
                    ax.legend()
                    training_curve_fig.savefig(os.path.join(fig_path, f'{task_name}_{reg_marker_str}.png'))

            # grid plot
            grid_fig.clear()
            axs = grid_fig.subplots(2, 2, sharex=True, sharey=True)
            min_loss = np.min(all_loss, axis=4)
            vmin = np.min(min_loss)
            vmax = np.max(min_loss)
            img = axs[0][0].imshow(min_loss[:, :, 0, 0], vmin=vmin, vmax=vmax)  # train loss, Additive
            axs[0][1].imshow(min_loss[:, :, 0, 1], vmin=vmin, vmax=vmax)  # test loss, Additive
            axs[1][0].imshow(min_loss[:, :, 1, 0], vmin=vmin, vmax=vmax)  # train loss, Multiplicative
            axs[1][1].imshow(min_loss[:, :, 1, 1], vmin=vmin, vmax=vmax)  # test loss, Multiplicative
            axs[0][0].set_title('Minimum train loss')
            axs[0][1].set_title('Minimum test loss')
            axs[0][0].set_ylabel('Additive')
            axs[1][0].set_ylabel('Multiplicative')
            grid_fig.suptitle(f'{task_name} l1={l1_levels[l1_idx]:.0e}')
            grid_fig.supxlabel('smooth regularization')
            grid_fig.supylabel('l2 regularization')
            grid_fig.colorbar(img, ax=axs.ravel().tolist(), shrink=0.6)
            grid_fig.savefig(os.path.join(fig_path, f'{task_name}_l1level={l1_idx}.png'))


if __name__ == '__main__':
    main()
