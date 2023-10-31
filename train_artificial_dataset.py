import os
import time

import numpy as np
import torch

from training import train_model
from METAPARAMETERS import *


def main():
    np.random.seed(113)
    torch.manual_seed(308)
    tasks = ['mul', 'add']
    train_modes = ['MultiWithLatent', 'AddWithLatent']
    max_epoch = 1000

    for task_name in tasks:
        for train_mode in train_modes:
            print(f'Training {task_name} with {train_mode}')
            regularization_paras = {'lambda_position': 0.0, 'lambda_timestamp': 0.0,
                                    'lambda_position_smooth': 0.0, 'lambda_timestamp_smooth': 0.0,
                                    'lambda_latent_l1': 0.0, 'lambda_latent_l2': 0.0, }
            train_loss, test_loss = train_model(task_name, train_mode, from_epoch=0, to_epoch=max_epoch,
                                                regularization_paras=regularization_paras,
                                                folder=os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset'),
                                                model_name='Artificial',)
            np.save(os.path.join(GLOBAL_PATH, 'analysis',
                                 f'train_loss_{task_name}_{train_mode}_{0}_{max_epoch}.npy'), train_loss)
            np.save(os.path.join(GLOBAL_PATH, 'analysis',
                                 f'test_loss_{task_name}_{train_mode}_{0}_{max_epoch}.npy'), test_loss)


if __name__ == '__main__':
    main()
