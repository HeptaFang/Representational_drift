import numpy as np
import matplotlib.pyplot as plt

from METAPARAMETERS import *


def main(noise_level, bias, fit_mode, pool_mode):
    # tasks = ['mouse1', 'mouse2', 'mouse3', 'mouse4', 'mouse5']
    tasks = ['mul', 'add']
    train_modes = ['MultiWithLatent', 'AddWithLatent']

    # colors = {'Additive': 'r', 'Multiplicative': 'b', 'MultiWithLatent': 'g'}
    # colors = {'AddWithLatent': 'r', 'MultiWithLatent': 'g'}
    colors = {'mul-MultiWithLatent': 'r', 'add-AddWithLatent': 'b',
              'mul-AddWithLatent': 'g', 'add-MultiWithLatent': 'y'}
    max_epoch = 1200

    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111)

    for task_name in tasks:
        full_task_name = f'{task_name}_{noise_level:.1f}_{bias:.1f}'
        max_loss = 0
        min_loss = 100000

        for train_mode in train_modes:
            all_train_loss = np.zeros((max_epoch, 4, N_REPEAT))
            all_test_loss = np.zeros((max_epoch, N_REPEAT))
            for seed in range(1):
                all_train_loss[:, :, seed] = np.load(
                    f'analysis\\artificial_dataset\\train_loss_{full_task_name}_{train_mode}_{seed}_0_{max_epoch}_fit={fit_mode}_pool={pool_mode}.npy')
                all_test_loss[:, seed] = np.load(
                    f'analysis\\artificial_dataset\\test_loss_{full_task_name}_{train_mode}_{seed}_0_{max_epoch}_fit={fit_mode}_pool={pool_mode}.npy')

            label = f'{task_name}-{train_mode}'
            # plot train loss and std
            ax.plot(np.mean(all_train_loss, axis=2)[:, 0], label=f'model={train_mode[:3]}, task={task_name} train',
                    color=colors[label],
                    linestyle='solid')
            # ax.fill_between(range(max_epoch),
            #                 np.mean(all_train_loss, axis=2)[:, 0] - np.std(all_train_loss, axis=2)[:, 0],
            #                 np.mean(all_train_loss, axis=2)[:, 0] + np.std(all_train_loss, axis=2)[:, 0],
            #                 alpha=0.2, color=colors[label])
            # plot test loss and std
            ax.plot(np.mean(all_test_loss, axis=1), label=f'model={train_mode[:3]}, task={task_name} test',
                    color=colors[label],
                    linestyle='dashed', )
            # ax.fill_between(range(max_epoch), np.mean(all_test_loss, axis=1) - np.std(all_test_loss, axis=1),
            #                 np.mean(all_test_loss, axis=1) + np.std(all_test_loss, axis=1),
            #                 alpha=0.2, color=colors[label])

    ax.legend()
    ax.set_title(f'Noise level: {noise_level:.1f}, Bias: {bias:.1f}\nFit mode: {fit_mode}, Pool mode: {pool_mode}')
    # ax.set_ylim(-0.1, 1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    fig.savefig(
        f'image\\artificial_dataset\\training\\artificial_{noise_level:.1f}_{bias:.1f}_fit={fit_mode}_pool={pool_mode}.png')


if __name__ == '__main__':
    for noise_level in [0.0, 0.5]:
        for bias in [-2.0, -1.0]:
            # for fit_pool in [(3, True), (7, True), (3, False), (7, False), (None, False), (None, True)]:
            for fit_pool in [(3, True)]:
                main(noise_level, bias, fit_pool[0], fit_pool[1])
