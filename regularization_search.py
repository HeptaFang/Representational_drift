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
