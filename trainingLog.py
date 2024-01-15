import numpy as np
import matplotlib.pyplot as plt

from METAPARAMETERS import *


def main():
    separate_task = False
    tasks = ['mouse1', 'mouse2', 'mouse3', 'mouse4', 'mouse5']
    # tasks = ['mouse1']
    # tasks = ['mul', 'add']
    train_modes = ['MultiWithLatent', 'AddWithLatent']

    # colors = {'Additive': 'r', 'Multiplicative': 'b', 'MultiWithLatent': 'g'}
    colors = {'AddWithLatent': 'b', 'MultiWithLatent': 'r'}
    # colors = {'mul-MultiWithLatent': 'r', 'add-AddWithLatent': 'b',
    #           'mul-AddWithLatent': 'g', 'add-MultiWithLatent': 'y'}
    reg_epoch = 200
    max_epoch = 1000

    for task_name in tasks:
        fig_basic = plt.figure(figsize=(12, 8), dpi=100)
        ax_basic = fig_basic.add_subplot(121)
        ax_regularize = fig_basic.add_subplot(122)
        # fig_regularize = plt.figure(figsize=(6, 8), dpi=100)
        # ax_regularize = fig_regularize.add_subplot(111)
        x_basic = np.arange(0, reg_epoch, 1)
        x_extent = np.arange(reg_epoch, max_epoch, 1)
        max_loss = 0
        min_loss = 100000

        for train_mode in train_modes:
            all_basic_train_loss = []
            all_basic_test_loss = []
            all_extent_train_loss = []
            all_extent_test_loss = []
            all_regularize_train_loss = []
            all_regularize_test_loss = []

            for test_idx in range(16):
                basic_train_loss = np.load(
                    f'analysis\\train_loss_{task_name}_{train_mode}_0_{reg_epoch}_pool={POOLED}_{test_idx}.npy')
                basic_test_loss = np.load(
                    f'analysis\\test_loss_{task_name}_{train_mode}_0_{reg_epoch}_pool={POOLED}_{test_idx}.npy')
                extent_train_loss = np.load(
                    f'analysis\\train_loss_{task_name}_{train_mode}_{reg_epoch}_{max_epoch}_pool={POOLED}_{test_idx}.npy')
                extent_test_loss = np.load(
                    f'analysis\\test_loss_{task_name}_{train_mode}_{reg_epoch}_{max_epoch}_pool={POOLED}_{test_idx}.npy')
                regularize_train_loss = np.load(
                    f'analysis\\train_loss_{task_name}_{train_mode}_{reg_epoch}_{max_epoch}_regularize_pool={POOLED}_{test_idx}.npy')
                regularize_test_loss = np.load(
                    f'analysis\\test_loss_{task_name}_{train_mode}_{reg_epoch}_{max_epoch}_regularize_pool={POOLED}_{test_idx}.npy')

                all_basic_train_loss.append(basic_train_loss)
                all_basic_test_loss.append(basic_test_loss)
                all_extent_train_loss.append(extent_train_loss)
                all_extent_test_loss.append(extent_test_loss)
                all_regularize_train_loss.append(regularize_train_loss)
                all_regularize_test_loss.append(regularize_test_loss)

            basic_train_loss = np.mean(np.array(all_basic_train_loss), axis=0)
            basic_test_loss = np.mean(np.array(all_basic_test_loss), axis=0)
            extent_train_loss = np.mean(np.array(all_extent_train_loss), axis=0)
            extent_test_loss = np.mean(np.array(all_extent_test_loss), axis=0)
            regularize_train_loss = np.mean(np.array(all_regularize_train_loss), axis=0)
            regularize_test_loss = np.mean(np.array(all_regularize_test_loss), axis=0)
            basic_train_std = np.std(np.array(all_basic_train_loss), axis=0)
            basic_test_std = np.std(np.array(all_basic_test_loss), axis=0)
            extent_train_std = np.std(np.array(all_extent_train_loss), axis=0)
            extent_test_std = np.std(np.array(all_extent_test_loss), axis=0)
            regularize_train_std = np.std(np.array(all_regularize_train_loss), axis=0)
            regularize_test_std = np.std(np.array(all_regularize_test_loss), axis=0)

            ax_basic.plot(x_basic, basic_train_loss[:, 0], label=f'{train_mode} train', color=colors[train_mode],
                          linestyle='solid', )
            ax_basic.plot(x_basic, basic_test_loss, label=f'{train_mode} test', color=colors[train_mode],
                          linestyle='dashed')
            ax_basic.plot(x_extent, extent_train_loss[:, 0], color=colors[train_mode], linestyle='solid')
            ax_basic.plot(x_extent, extent_test_loss, color=colors[train_mode], linestyle='dashed')
            ax_regularize.plot(x_basic, basic_train_loss[:, 0], label=f'{train_mode} train', color=colors[train_mode],
                               linestyle='solid')
            ax_regularize.plot(x_basic, basic_test_loss, label=f'{train_mode} test', color=colors[train_mode],
                               linestyle='dashed')
            ax_regularize.plot(x_extent, regularize_train_loss[:, 0], color=colors[train_mode], linestyle='solid')
            ax_regularize.plot(x_extent, regularize_test_loss, color=colors[train_mode], linestyle='dashed')

            ax_basic.fill_between(x_basic, basic_train_loss[:, 0] - basic_train_std[:, 0],
                                  basic_train_loss[:, 0] + basic_train_std[:, 0], alpha=0.3, color=colors[train_mode])
            ax_basic.fill_between(x_basic, basic_test_loss - basic_test_std, basic_test_loss + basic_test_std,
                                  alpha=0.3,
                                  color=colors[train_mode])
            ax_basic.fill_between(x_extent, extent_train_loss[:, 0] - extent_train_std[:, 0],
                                  extent_train_loss[:, 0] + extent_train_std[:, 0], alpha=0.3, color=colors[train_mode])
            ax_basic.fill_between(x_extent, extent_test_loss - extent_test_std, extent_test_loss + extent_test_std,
                                  alpha=0.3, color=colors[train_mode])
            ax_regularize.fill_between(x_basic, basic_train_loss[:, 0] - basic_train_std[:, 0],
                                       basic_train_loss[:, 0] + basic_train_std[:, 0], alpha=0.3,
                                       color=colors[train_mode])
            ax_regularize.fill_between(x_basic, basic_test_loss - basic_test_std, basic_test_loss + basic_test_std,
                                       alpha=0.3, color=colors[train_mode])
            ax_regularize.fill_between(x_extent, regularize_train_loss[:, 0] - regularize_train_std[:, 0],
                                       regularize_train_loss[:, 0] + regularize_train_std[:, 0], alpha=0.3,
                                       color=colors[train_mode])
            ax_regularize.fill_between(x_extent, regularize_test_loss - regularize_test_std,
                                       regularize_test_loss + regularize_test_std, alpha=0.3,
                                       color=colors[train_mode])

            if np.max(basic_test_loss) > max_loss:
                max_loss = np.max(basic_test_loss)
            if np.min(regularize_train_loss) < min_loss:
                min_loss = np.min(regularize_train_loss)

        ax_basic.legend()
        ax_basic.set_ylim(-0.1, 1.5)
        ax_basic.set_xlabel('Epoch')
        ax_basic.set_ylabel('Loss')
        ax_basic.set_title(f'{task_name} No Regularization')
        ax_regularize.legend()
        ax_regularize.set_ylim(-0.1, 1.5)
        ax_regularize.vlines(reg_epoch, 0, 1.4, color='k', linestyle='dashed')
        ax_regularize.set_xlabel('Epoch')
        ax_regularize.set_ylabel('Loss')
        ax_regularize.set_title(f'{task_name} With Regularization')
        fig_basic.savefig(f'image\\pooling\\{task_name}_2k_pool={POOLED}_full.png')
        # fig_regularize.savefig(f'image\\{task_name}_regularize_2k.png')


if __name__ == '__main__':
    main()
