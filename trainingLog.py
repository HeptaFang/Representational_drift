import numpy as np
import matplotlib.pyplot as plt


def main():
    # tasks = ['mouse1', 'mouse2', 'mouse3', 'mouse4', 'mouse5']
    tasks = ['mouse1']
    train_modes = ['Additive', 'Multiplicative', 'MultiWithLatent']
    train_modes = ['AddWithLatent', 'MultiWithLatent']

    # colors = {'Additive': 'r', 'Multiplicative': 'b', 'MultiWithLatent': 'g'}
    colors = {'AddWithLatent': 'r', 'MultiWithLatent': 'g'}
    reg_epoch = 500
    max_epoch = 5000

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
            basic_train_loss = np.load(f'analysis\\train_loss_{task_name}_{train_mode}_0_{reg_epoch}.npy')
            basic_test_loss = np.load(f'analysis\\test_loss_{task_name}_{train_mode}_0_{reg_epoch}.npy')
            extent_train_loss = np.load(f'analysis\\train_loss_{task_name}_{train_mode}_{reg_epoch}_{max_epoch}.npy')
            extent_test_loss = np.load(f'analysis\\test_loss_{task_name}_{train_mode}_{reg_epoch}_{max_epoch}.npy')
            regularize_train_loss = np.load(
                f'analysis\\train_loss_{task_name}_{train_mode}_{reg_epoch}_{max_epoch}_regularize.npy')
            regularize_test_loss = np.load(
                f'analysis\\test_loss_{task_name}_{train_mode}_{reg_epoch}_{max_epoch}_regularize.npy')

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
        fig_basic.savefig(f'image\\{task_name}_5k_new.png')
        # fig_regularize.savefig(f'image\\{task_name}_regularize_2k.png')


if __name__ == '__main__':
    main()
