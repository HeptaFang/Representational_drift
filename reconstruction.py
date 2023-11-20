import torch
import numpy as np
import os
from model import load_model

from METAPARAMETERS import *


def main():
    noise_level = 0.5
    bias = -1.0
    model_name = 'Artificial'
    task_name = 'mul'
    train_mode = 'MultiWithLatent'
    from_epoch = 0
    bias_mode = 'fixed'
    folder = os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset')

    path = os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset')
    position_encoding = np.load(os.path.join(path, f'mul_{noise_level:.1f}_{bias:.1f}_position_encoding.npy'))
    timestamp_encoding = np.load(os.path.join(path, f'mul_{noise_level:.1f}_{bias:.1f}_timestamp_encoding.npy'))
    projection = np.load(os.path.join(path, f'mul_{noise_level:.1f}_{bias:.1f}_projection.npy'))
    position = np.load(os.path.join(path, f'mul_{noise_level:.1f}_{bias:.1f}_position.npy'))
    timestamp = np.load(os.path.join(path, f'mul_{noise_level:.1f}_{bias:.1f}_timestamp.npy'))
    activity = np.load(os.path.join(path, f'mul_{noise_level:.1f}_{bias:.1f}_activity.npy'))
    position_tensor = torch.from_numpy(position).float()
    timestamp_tensor = torch.from_numpy(timestamp).float()

    position_encoding = position_encoding.reshape((32, 32)).T
    timestamp_encoding = timestamp_encoding.reshape((32, 32)).T
    projection = projection.T

    # load weights
    model = load_model(BIN_NUM, SESSION_NUM, CELL_NUM, train_mode, model_name, task_name, epoch=from_epoch,
                       bias_mode=bias_mode)

    position_encoding_tensor = torch.from_numpy(position_encoding).float()
    timestamp_encoding_tensor = torch.from_numpy(timestamp_encoding).float()
    projection_tensor = torch.from_numpy(projection).float()
    # bias: ones, size=(CELL_NUM,)
    bias_tensor = torch.ones(CELL_NUM).float() * bias
    weights = {'position_encoding.weight': position_encoding_tensor,
               'timestamp_encoding.weight': timestamp_encoding_tensor,
               'latent_projection.weight': projection_tensor, 'bias': bias_tensor}
    model.load_state_dict(weights)
    model.set_bias(bias)

    # save weights
    torch.save(model.state_dict(), os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset', f'mul_0.0_{bias}_reconstruction.m'))

    activity_pred = model(position_tensor, timestamp_tensor).detach().cpu().numpy()
    print(activity_pred.shape)
    print(activity.shape)
    print(np.mean(activity), np.mean(activity_pred), np.mean(np.abs(activity_pred - activity)))


if __name__ == '__main__':
    main()
