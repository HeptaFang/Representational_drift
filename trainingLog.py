import numpy as np
import matplotlib.pyplot as plt


def main():
    # tasks = ['mouse1', 'mouse2', 'mouse3', 'mouse4', 'mouse5']
    tasks = ['mouse1']
    train_modes = ['AddWithLatent', 'MultiWithLatent']

    # colors = {'Additive': 'r', 'Multiplicative': 'b', 'MultiWithLatent': 'g'}
    # colors = {'AddWithLatent': 'r', 'MultiWithLatent': 'g'}
    colors = {0: 'r', 0.5: 'g', 1: 'b', -1: 'y'}
    init_label = {0: 'exp=0', 0.5: 'exp=0.5', 1: 'exp=1', -1: 'zero'}
    max_epoch = 1000

    for task_name in tasks:
        fig = plt.figure(figsize=(12, 8), dpi=100)
        ax_add = fig.add_subplot(121)
        ax_multi = fig.add_subplot(122)
        # fig_regularize = plt.figure(figsize=(6, 8), dpi=100)
        # ax_regularize = fig_regularize.add_subplot(111)
        max_loss = 0
        min_loss = 100000

        for init_mode in [0, 0.5, 1, -1]:
            add_train_loss = np.load(
                f'analysis\\train_loss_{task_name}_AddWithLatent_0_{max_epoch}_init={init_mode}.npy')
            add_test_loss = np.load(f'analysis\\test_loss_{task_name}_AddWithLatent_0_{max_epoch}_init={init_mode}.npy')
            multi_train_loss = np.load(
                f'analysis\\train_loss_{task_name}_MultiWithLatent_0_{max_epoch}_init={init_mode}.npy')
            multi_test_loss = np.load(
                f'analysis\\test_loss_{task_name}_MultiWithLatent_0_{max_epoch}_init={init_mode}.npy')

            ax_add.plot(add_train_loss[:, 0], label=f'{init_label[init_mode]} train', color=colors[init_mode],
                        linestyle='solid', )
            ax_add.plot(add_test_loss, label=f'{init_label[init_mode]} test', color=colors[init_mode],
                        linestyle='dotted', )
            ax_multi.plot(multi_train_loss[:, 0], label=f'{init_label[init_mode]} train', color=colors[init_mode],
                          linestyle='solid', )
            ax_multi.plot(multi_test_loss, label=f'{init_label[init_mode]} test', color=colors[init_mode],
                          linestyle='dotted', )

        ax_add.legend()
        ax_add.set_ylim(-0.1, 1.5)
        ax_add.set_xlabel('Epoch')
        ax_add.set_ylabel('Loss')
        ax_add.set_title(f'{task_name} Additive')
        ax_multi.legend()
        ax_multi.set_ylim(-0.1, 1.5)
        ax_multi.set_xlabel('Epoch')
        ax_multi.set_ylabel('Loss')
        ax_multi.set_title(f'{task_name} Multiplicative')

        fig.savefig(f'image\\{task_name}_init_test.png')
        # fig_regularize.savefig(f'image\\{task_name}_regularize_2k.png')


if __name__ == '__main__':
    main()
