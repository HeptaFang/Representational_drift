import os
import time

import numpy as np
import torch

from training import train_model
from METAPARAMETERS import *


def train_artificial(noise_level, bias, seed, override_fit_order=-1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # tasks = ['mul', 'add']
    train_modes = ['MultiWithLatent', 'AddWithLatent']
    tasks = ['mul', 'add']
    # train_modes = ['MultiWithLatent']
    max_epoch = 1200

    for task_name in tasks:
        for train_mode in train_modes:
            single_start_time = time.time()
            full_task_name = f'{task_name}_{noise_level:.1f}_{bias:.1f}'
            print(f'Training {task_name} with {train_mode}')
            regularization_paras = {'lambda_position': 0.0, 'lambda_timestamp': 0.0,
                                    'lambda_position_smooth': 0.0, 'lambda_timestamp_smooth': 0.0,
                                    'lambda_latent_l1': 0.0, 'lambda_latent_l2': 0.0, 'lambda_bias': 0.0}
            # regularization_paras = {'lambda_position': 5e-3, 'lambda_timestamp': 5e-3,
            #                         'lambda_position_smooth': 0.0, 'lambda_timestamp_smooth': 0.0,
            #                         'lambda_latent_l1': 0.0, 'lambda_latent_l2': 5e-3, }
            train_loss, test_loss = train_model(full_task_name, train_mode,
                                                from_epoch=0, to_epoch=max_epoch,
                                                regularization_paras=regularization_paras,
                                                folder=os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset'),
                                                model_name='Artificial', log_level=1, bias_mode='train',
                                                reconstruction=False)
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
            if override_fit_order != -1:
                fit_order = override_fit_order
            else:
                fit_order = FIT_ORDER
            np.save(os.path.join(GLOBAL_PATH, 'analysis', 'artificial_dataset',
                                 f'train_loss_{full_task_name}_{train_mode}_{seed}_{0}_{max_epoch}_fit={fit_order}_pool={POOLED}.npy'),
                    train_loss)
            np.save(os.path.join(GLOBAL_PATH, 'analysis', 'artificial_dataset',
                                 f'test_loss_{full_task_name}_{train_mode}_{seed}_{0}_{max_epoch}_fit={fit_order}_pool={POOLED}.npy'),
                    test_loss)
            print(f'--- {time.time() - single_start_time} seconds ---')


def main(override_fit_order=-1):
    for noise_level in [0.0, 0.5]:
        for bias in [-2.0, -1.0]:
            for seed in range(1):
                start_time = time.time()
                print(noise_level, bias, seed)
                train_artificial(noise_level, bias, seed, override_fit_order=override_fit_order)
                print(f'--- {time.time() - start_time} seconds ---')


if __name__ == '__main__':
    main()
