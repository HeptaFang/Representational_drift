import numpy as np
import matplotlib.pyplot as plt


def main():
    # tasks = ['mouse1', 'mouse2', 'mouse3', 'mouse4', 'mouse5']
    tasks = ['mul', 'add']
    train_modes = ['MultiWithLatent', 'AddWithLatent']

    # colors = {'Additive': 'r', 'Multiplicative': 'b', 'MultiWithLatent': 'g'}
    # colors = {'AddWithLatent': 'r', 'MultiWithLatent': 'g'}
    colors = {'mul-MultiWithLatent': 'r', 'add-AddWithLatent': 'b',
              'mul-AddWithLatent': 'g', 'add-MultiWithLatent': 'y'}
    max_epoch = 1000
    
    fig = plt.figure(figsize=(6, 8), dpi=100)
    ax = fig.add_subplot(111)

    for task_name in tasks:
        max_loss = 0
        min_loss = 100000

        for train_mode in train_modes:
            train_loss = np.load(
                f'analysis\\train_loss_{task_name}_{train_mode}_0_{max_epoch}.npy')
            test_loss = np.load(
                f'analysis\\test_loss_{task_name}_{train_mode}_0_{max_epoch}.npy')

            label = f'{task_name}-{train_mode}'
            ax.plot(train_loss[:, 0], label=f'{label} train', color=colors[label],
                        linestyle='solid', )
            ax.plot(test_loss, label=f'{label} test', color=colors[label],
                        linestyle='dashed', )

    ax.legend()
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    fig.savefig(f'image\\artificial.png')


if __name__ == '__main__':
    main()
