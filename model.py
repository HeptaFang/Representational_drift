import torch
from torch import nn
from scipy.special import erfinv
import numpy as np
import os
from functools import partial

from METAPARAMETERS import *


class BindingModel(nn.Module):
    """
    input: position, timestamp
    output: predicted neuron response
    """

    def __init__(self, position_size, timestamp_size, output_size, latent_size=None, binding_mode='mul',
                 bias_mode='train'):
        super(BindingModel, self).__init__()
        self.position_size = position_size
        self.timestamp_size = timestamp_size
        self.output_size = output_size
        self.use_latent = True
        self.latent_size = latent_size
        self.binding_mode = binding_mode
        self.bias_mode = bias_mode
        print(bias_mode)

        self.sparseness_map = None
        self.bias_z_map = None

        if latent_size is None:
            self.use_latent = False
            self.latent_size = output_size

        self.position_encoding = nn.Linear(self.position_size, self.latent_size, bias=False)
        self.timestamp_encoding = nn.Linear(self.timestamp_size, self.latent_size, bias=False)
        self.latent_projection = nn.Linear(self.latent_size, self.output_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.output_size))
        if self.bias_mode == 'train':
            pass
        elif self.bias_mode == 'fixed':
            # set bias to -2
            self.bias.data = torch.ones(self.output_size) * (-2)
            self.bias.requires_grad = False
        elif self.bias_mode == 'sparseness':
            pass
        else:
            self.bias = None
        # self.activation = partial(nn.functional.leaky_relu, negative_slope=0.01)
        self.activation = nn.functional.relu

        # weight initialization
        # nn.init.normal_(self.position_encoding.weight, mean=0, std=1)
        # nn.init.normal_(self.timestamp_encoding.weight, mean=0, std=1)
        # nn.init.normal_(self.latent_projection.weight, mean=0, std=1 / (self.latent_size ** 0.5))
    def set_bias(self, bias=None):
        if bias is None:
            if self.bias_mode == 'fixed':
                self.bias.data = torch.ones(self.output_size) * (-2)
            return

        self.bias.data = torch.ones(self.output_size) * bias

    def load_sparseness_map(self, sparseness_map):
        # sparseness_map should be 1D: cell_num
        self.sparseness_map = sparseness_map
        assert self.sparseness_map.shape[0] == self.output_size and len(self.sparseness_map.shape) == 1
        self.bias_z_map = 2 ** 0.5 * erfinv(2 * self.sparseness_map - 1)

    def forward(self, position, timestamp):
        # encoding
        position_code = self.position_encoding(position)
        timestamp_code = self.timestamp_encoding(timestamp)

        # binding
        if self.binding_mode == 'mul':
            binding_code = position_code * timestamp_code
        elif self.binding_mode == 'add':
            binding_code = (position_code + timestamp_code) * (2 ** -0.5)  # normalize output variance
        else:
            raise ValueError('Invalid binding mode')

        # latent space projection
        if self.use_latent:
            binding_code_projected = self.latent_projection(binding_code)
        else:
            binding_code_projected = binding_code

        # bias
        if self.bias_mode == 'sparseness':
            self.bias = self.bias_z_map * torch.std(binding_code_projected, dim=1, keepdim=True) + torch.mean(
                binding_code_projected, dim=1, keepdim=True)

        # activation
        output = self.activation(binding_code_projected + self.bias)

        return output


def load_model(bin_num, session_num, cell_num, train_mode, model_name=None, task_name=None, epoch=0,
               return_weight=False, bias_mode='train',reconstruction=False):
    if train_mode == 'Additive':
        model = BindingModel(bin_num, session_num, cell_num, binding_mode='add')
    elif train_mode == 'Multiplicative':
        model = BindingModel(bin_num, session_num, cell_num, binding_mode='mul')
    elif train_mode == 'AddWithLatent':
        model = BindingModel(bin_num, session_num, cell_num, binding_mode='add', latent_size=MODEL_HIDDEN_NUM, bias_mode=bias_mode)
    elif train_mode == 'MultiWithLatent':
        model = BindingModel(bin_num, session_num, cell_num, binding_mode='mul', latent_size=MODEL_HIDDEN_NUM, bias_mode=bias_mode)
    elif train_mode == 'MultiWithLatentMedium':
        model = BindingModel(bin_num, session_num, cell_num, binding_mode='mul', latent_size=128)
    elif train_mode == 'MultiWithLatentLarge':
        model = BindingModel(bin_num, session_num, cell_num, binding_mode='mul', latent_size=256)
    else:
        raise ValueError('Invalid train mode.')

    weight_dict = None
    if epoch != 0:
        model_path = os.path.join(GLOBAL_PATH, 'model',
                                  model_name + '_' + train_mode + '_' + task_name + '_' + str(epoch) + '.m')
        weight_dict = torch.load(model_path)
        model.load_state_dict(weight_dict)

    if reconstruction:
        weight_dict = torch.load(os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset', 'mul_0.0_-2.0_reconstruction.m'))
        model.load_state_dict(weight_dict)

    model.set_bias()

    if return_weight:
        return model, weight_dict
    else:
        return model


def main():
    # model = GRUNet(2, 64, 1)
    # dim_in = 2
    # dim_hidden = 128
    # dim_out = 1
    # model = nn.Sequential(nn.GRU(input_size=dim_in, hidden_size=dim_hidden, batch_first=True),
    #                       nn.Linear(in_features=dim_hidden, out_features=dim_out))
    model_name = 'GRU64_Single'
    # torch.save(model.state_dict(), GLOBAL_PATH +'model\\' + model_name + '.m')


if __name__ == '__main__':
    main()
