import torch
from torch import nn


class BindingModel(nn.Module):
    """
    input: position, timestamp
    output: predicted neuron response
    """

    def __init__(self, position_size, timestamp_size, output_size, latent_size=None, binding_mode='mul'):
        super(BindingModel, self).__init__()
        self.position_size = position_size
        self.timestamp_size = timestamp_size
        self.output_size = output_size
        self.use_latent = True
        self.latent_size = latent_size
        self.binding_mode = binding_mode

        if latent_size is None:
            self.use_latent = False
            self.latent_size = output_size

        self.position_encoding = nn.Linear(self.position_size, self.latent_size, bias=False)
        self.timestamp_encoding = nn.Linear(self.timestamp_size, self.latent_size, bias=False)
        self.latent_projection = nn.Linear(self.latent_size, self.output_size)
        self.activation = nn.functional.relu

    def forward(self, position, timestamp):
        # encoding
        position_code = self.position_encoding(position)
        timestamp_code = self.timestamp_encoding(timestamp)

        # binding
        if self.binding_mode == 'mul':
            binding_code = position_code * timestamp_code
        elif self.binding_mode == 'add':
            binding_code = position_code + timestamp_code
        else:
            raise ValueError('Invalid binding mode')

        # latent space projection
        if self.use_latent:
            binding_code_projected = self.latent_projection(binding_code)
        else:
            binding_code_projected = binding_code

        # activation
        output = self.activation(binding_code_projected)

        return output


def load_model(bin_num, session_num, cell_num, train_mode, model_name=None, task_name=None, epoch=0,
               return_weight=False, reg=False):
    if train_mode == 'Additive':
        model = BindingModel(bin_num, session_num, cell_num, binding_mode='add')
    elif train_mode == 'Multiplicative':
        model = BindingModel(bin_num, session_num, cell_num, binding_mode='mul')
    elif train_mode == 'AddWithLatent':
        model = BindingModel(bin_num, session_num, cell_num, binding_mode='add', latent_size=32)
    elif train_mode == 'MultiWithLatent':
        model = BindingModel(bin_num, session_num, cell_num, binding_mode='mul', latent_size=32)
    elif train_mode == 'MultiWithLatentMedium':
        model = BindingModel(bin_num, session_num, cell_num, binding_mode='mul', latent_size=128)
    elif train_mode == 'MultiWithLatentLarge':
        model = BindingModel(bin_num, session_num, cell_num, binding_mode='mul', latent_size=256)
    else:
        raise ValueError('Invalid train mode.')

    weight_dict = None
    if epoch != 0:
        model_path = f'model\\{model_name}_{train_mode}_{task_name}_{epoch}.m'
        if reg:
            model_path = f'model\\{model_name}_{train_mode}_{task_name}_{epoch}_reg.m'
        weight_dict = torch.load(model_path)
        model.load_state_dict(weight_dict)

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
    # torch.save(model.state_dict(), 'model\\' + model_name + '.m')


if __name__ == '__main__':
    main()
