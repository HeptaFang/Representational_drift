import numpy as np
import torch
import os

from training import train_model
from METAPARAMETERS import *


def main():
    np.random.seed(113)
    torch.manual_seed(308)
    # tasks = ['mouse1', 'mouse2', 'mouse3', 'mouse4', 'mouse5']
    tasks = ['mouse1']
    train_modes = ['AddWithLatent', 'MultiWithLatent']
    max_epoch = 1000

    l1_levels = [0, 1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6]
    l2_levels = [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    smooth_levels = [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

    l1 = 1e-6
    l2 = 1e-4
    smooth = 2e-4

    for task_name in tasks:
        for train_mode in train_modes:
            print(f'Training {task_name} with {train_mode} Phase Basic')
            regularization_paras = {'lambda_position': l2, 'lambda_timestamp': l2,
                                    'lambda_position_smooth': smooth, 'lambda_timestamp_smooth': smooth,
                                    'lambda_latent_l1': l1, 'lambda_latent_l2': l2, }
            train_loss, test_loss = train_model(task_name, train_mode, from_epoch=0, to_epoch=max_epoch,
                                                regularization_paras=regularization_paras)
            np.save(os.path.join(GLOBAL_PATH, 'analysis',
                                 f'train_loss_{task_name}_{train_mode}_{0}_{max_epoch}.npy'), train_loss)
            np.save(os.path.join(GLOBAL_PATH, 'analysis',
                                 f'test_loss_{task_name}_{train_mode}_{0}_{max_epoch}.npy'), test_loss)


if __name__ == '__main__':
    main()
