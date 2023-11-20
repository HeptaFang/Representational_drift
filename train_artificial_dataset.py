import os
import time

import numpy as np
import torch

from training import train_model
from METAPARAMETERS import *


def main(noise_level, bias, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # tasks = ['mul', 'add']
    # train_modes = ['MultiWithLatent', 'AddWithLatent']
    tasks = ['mul']
    train_modes = ['MultiWithLatent']
    max_epoch = 1000

    for task_name in tasks:
        for train_mode in train_modes:
            full_task_name = f'{task_name}_{noise_level:.1f}_{bias:.1f}'
            print(f'Training {task_name} with {train_mode}')
            regularization_paras = {'lambda_position': 0.0, 'lambda_timestamp': 0.0,
                                    'lambda_position_smooth': 0.0, 'lambda_timestamp_smooth': 0.0,
                                    'lambda_latent_l1': 0.0, 'lambda_latent_l2': 0.0, 'lambda_bias': 0.0}
            # regularization_paras = {'lambda_position': 5e-3, 'lambda_timestamp': 5e-3,
            #                         'lambda_position_smooth': 0.0, 'lambda_timestamp_smooth': 0.0,
            #                         'lambda_latent_l1': 0.0, 'lambda_latent_l2': 5e-3, }
            train_loss, test_loss = train_model(full_task_name, train_mode,
                                                from_epoch=2000, to_epoch=3000,
                                                regularization_paras=regularization_paras,
                                                folder=os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset'),
                                                model_name='Artificial', log_level=2, bias_mode='train', reconstruction=False)
            # train_loss, test_loss = train_model(full_task_name, train_mode,
            #                                     from_epoch=200, to_epoch=400,
            #                                     regularization_paras=regularization_paras,
            #                                     folder=os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset'),
            #                                     model_name='Artificial', log_level=2, bias_mode='fixed')
            # train_loss, test_loss = train_model(full_task_name, train_mode,
            #                                     from_epoch=400, to_epoch=600,
            #                                     regularization_paras=regularization_paras,
            #                                     folder=os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset'),
            #                                     model_name='Artificial', log_level=2, bias_mode='train')
            # regularization_paras = {'lambda_position': 1e-2, 'lambda_timestamp': 1e-2,
            #                         'lambda_position_smooth': 0.0, 'lambda_timestamp_smooth': 0.0,
            #                         'lambda_latent_l1': 0.0, 'lambda_latent_l2': 1e-2, }
            # train_loss, test_loss = train_model(full_task_name, train_mode,
            #                                     from_epoch=200, to_epoch=max_epoch,
            #                                     regularization_paras=regularization_paras,
            #                                     folder=os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset'),
            #                                     model_name='Artificial', log_level=2)
            # np.save(os.path.join(GLOBAL_PATH, 'analysis',
            #                      f'train_loss_{full_task_name}_{train_mode}_{seed}_{0}_{max_epoch}.npy'), train_loss)
            # np.save(os.path.join(GLOBAL_PATH, 'analysis',
            #                      f'test_loss_{full_task_name}_{train_mode}_{seed}_{0}_{max_epoch}.npy'), test_loss)


if __name__ == '__main__':
    for noise_level in [0.0]:
        for bias in [-1.0]:
            for seed in range(1):
                print(noise_level, bias, seed)
                main(noise_level, bias, seed)
