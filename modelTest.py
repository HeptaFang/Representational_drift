import numpy as np
import torch
from model import BindingModel
from matplotlib import pyplot as plt


def main():
    task_name = 'mouse1'
    train_mode = 'MultiWithLatent'
    model_name = 'Binding'
    epoch_no = 1000

    position = np.load('dataset\\' + task_name + '_position.npy')
    timestamp = np.load('dataset\\' + task_name + '_timestamp.npy')
    activity = np.load('dataset\\' + task_name + '_activity.npy')

    bin_num = position.shape[1]
    session_num = timestamp.shape[1]
    cell_num = activity.shape[1]

    if train_mode == 'Additive':
        model = BindingModel(bin_num, session_num, cell_num, binding_mode='add')
    elif train_mode == 'Multiplicative':
        model = BindingModel(bin_num, session_num, cell_num, binding_mode='mul')
    elif train_mode == 'MultiWithLatent':
        model = BindingModel(bin_num, session_num, cell_num, binding_mode='mul', latent_size=64)
    else:
        raise ValueError('Invalid train mode.')

    model.load_state_dict(torch.load(f'model\\{model_name}_{train_mode}_{task_name}_{epoch_no}.m'))

    print(model.position_encoding.weight.size)
    print(model.timestamp_encoding.weight.size)
    print(model.latent_projection.weight.size)

    plt.imshow(model.position_encoding.weight.detach())
    plt.show()
    plt.imshow(model.timestamp_encoding.weight.detach())
    plt.show()
    plt.imshow(model.latent_projection.weight.detach())
    plt.show()


if __name__ == '__main__':
    main()
