import numpy as np
import torch
import os
import sys

from training import train_model
from METAPARAMETERS import *


def main(l1_idx, smooth_idx):
    np.random.seed(113)
    torch.manual_seed(308)
    # tasks = ['mouse1', 'mouse2', 'mouse3', 'mouse4', 'mouse5']
    tasks = ['mouse1']
    train_modes = ['MultiWithLatent', 'AddWithLatent']
    max_epoch = 1200

    l1_levels = [0, 1e-7, 1e-5, 1e-3, 1e-1]
    l2_levels = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    smooth_levels = [0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # l1_levels = [0]
    # l2_levels = [0]
    # smooth_levels = [0]

    smooth = smooth_levels[smooth_idx]
    l1 = l1_levels[l1_idx]
    for l2 in l2_levels:
        for task_name in tasks:
            for train_mode in train_modes:
                reg_marker = f'{l1}_{l2}_{smooth}'
                print(f'Training {task_name} with {train_mode} Phase Basic, reg: {reg_marker}')
                regularization_paras = {'lambda_position': l2, 'lambda_timestamp': l2,
                                        'lambda_position_smooth': smooth, 'lambda_timestamp_smooth': smooth,
                                        'lambda_latent_l1': l1, 'lambda_latent_l2': l2, }
                train_loss, test_loss = train_model(task_name, train_mode, from_epoch=0, to_epoch=max_epoch,
                                                    regularization_paras=regularization_paras, reg_marker=reg_marker)
                np.save(os.path.join(GLOBAL_PATH, 'analysis',
                                     f'train_loss_{task_name}_{train_mode}_{reg_marker}_{0}_{max_epoch}.npy'),
                        train_loss)
                np.save(os.path.join(GLOBAL_PATH, 'analysis',
                                     f'test_loss_{task_name}_{train_mode}_{reg_marker}_{0}_{max_epoch}.npy'), test_loss)


if __name__ == '__main__':
    main(0, 0)
