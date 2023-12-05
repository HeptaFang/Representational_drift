import numpy as np
import matplotlib.pyplot as plt

from METAPARAMETERS import *


def main(noise_level, bias, fit_mode):
    separate_task = False
    # tasks = ['mouse1', 'mouse2', 'mouse3', 'mouse4', 'mouse5']
    tasks = ['mul', 'add']
    train_modes = ['MultiWithLatent', 'AddWithLatent']

    # colors = {'Additive': 'r', 'Multiplicative': 'b', 'MultiWithLatent': 'g'}
    # colors = {'AddWithLatent': 'r', 'MultiWithLatent': 'g'}
    colors = {'mul-MultiWithLatent': 'r', 'add-AddWithLatent': 'b',
              'mul-AddWithLatent': 'g', 'add-MultiWithLatent': 'y'}
    max_epoch = 1200

    fig = plt.figure(figsize=(16, 16), dpi=100)
    axs = []

    for pool_mode in [False, True]:
        ax = fig.add_subplot(2, 2, 2 if pool_mode else 1)
        axs.append(ax)
        for task_name in tasks:
            full_task_name = f'{task_name}_{noise_level:.1f}_{bias:.1f}'

            for train_mode in train_modes:
                all_train_loss = np.zeros((max_epoch, 4, N_REPEAT))
                all_test_loss = np.zeros((max_epoch, N_REPEAT))
                for seed in range(1):
                    all_train_loss[:, :, seed] = np.load(
                        f'analysis\\artificial_dataset\\train_loss_{full_task_name}_{train_mode}_{seed}_0_{max_epoch}_fit={fit_mode}_pool={pool_mode}.npy')
                    all_test_loss[:, seed] = np.load(
                        f'analysis\\artificial_dataset\\test_loss_{full_task_name}_{train_mode}_{seed}_0_{max_epoch}_fit={fit_mode}_pool={pool_mode}.npy')

                label = f'{task_name}-{train_mode}'
                ax.plot(np.mean(all_train_loss, axis=2)[:, 0], label=f'model={train_mode[:3]}, task={task_name} train',
                        color=colors[label],
                        linestyle='solid')
                ax.plot(np.mean(all_test_loss, axis=1), label=f'model={train_mode[:3]}, task={task_name} test',
                        color=colors[label],
                        linestyle='dashed', )

        ax.legend()
        if pool_mode:
            ax.set_title(f'Pooled')
        else:
            ax.set_title(f'Original')
        # ax.set_ylim(-0.1, 1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

        # ratios
        ax = fig.add_subplot(2, 2, 4 if pool_mode else 3)
        axs.append(ax)
        for task_name in tasks:
            full_task_name = f'{task_name}_{noise_level:.1f}_{bias:.1f}'

            for train_mode in train_modes:
                all_train_loss = np.zeros((max_epoch, 4, N_REPEAT))
                all_test_loss = np.zeros((max_epoch, N_REPEAT))
                for seed in range(1):
                    all_train_loss[:, :, seed] = np.load(
                        f'analysis\\artificial_dataset\\train_loss_{full_task_name}_{train_mode}_{seed}_0_{max_epoch}_fit={fit_mode}_pool={pool_mode}.npy')
                    all_test_loss[:, seed] = np.load(
                        f'analysis\\artificial_dataset\\test_loss_{full_task_name}_{train_mode}_{seed}_0_{max_epoch}_fit={fit_mode}_pool={pool_mode}.npy')

                label = f'{task_name}-{train_mode}'
                ax.plot(np.mean(all_test_loss, axis=1) / np.mean(all_train_loss, axis=2)[:, 0],
                        label=f'model={train_mode[:3]}, task={task_name}',
                        color=colors[label],
                        linestyle='solid')

        ax.legend()
        if pool_mode:
            ax.set_title(f'Pooled loss ratio')
        else:
            ax.set_title(f'Original loss ratio')
        # ax.set_ylim(-0.1, 1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('test loss / train loss')

    # same y axis for each pair of plots
    max_ylim = [min(axs[0].get_ylim()[0], axs[2].get_ylim()[0]),
                max(axs[0].get_ylim()[1], axs[2].get_ylim()[1])]
    axs[0].set_ylim(max_ylim)
    axs[2].set_ylim(max_ylim)
    max_ylim = [min(axs[1].get_ylim()[0], axs[3].get_ylim()[0]),
                max(axs[1].get_ylim()[1], axs[3].get_ylim()[1])]
    axs[1].set_ylim(max_ylim)
    axs[3].set_ylim(max_ylim)

    fig.suptitle(f'Noise level: {noise_level}, Bias: {bias}, Fit mode: {fit_mode}')
    fig.savefig(
        f'image\\artificial_dataset\\training\\artificial_{noise_level:.1f}_{bias:.1f}_fit={fit_mode}.png')

    plt.close(fig)


if __name__ == '__main__':
    for noise_level in [0.0, 0.5]:
        for bias in [-2.0, -1.0]:
            for fit_order in [None, 3, 7]:
                # for fit_pool in [(3, True), (None, True)]:
                main(noise_level, bias, fit_order)
