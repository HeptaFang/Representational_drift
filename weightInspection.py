import numpy as np
import torch
from model import BindingModel
from matplotlib import pyplot as plt


def main():
    # train_mode = 'Multiplicative'
    train_mode = 'MultiWithLatent'
    mouse_i = 1
    training_epoch = 5000
    model_path = f'model\\Binding_{train_mode}_mouse{mouse_i}_{training_epoch}.m'

    weight_dict = torch.load(model_path)

    position_num, _ = weight_dict['position_encoding.weight'].size()
    timestamp_num, _ = weight_dict['timestamp_encoding.weight'].size()
    cell_num, latent_num = weight_dict['latent_projection.weight'].size()

    # model = BindingModel(position_size=position_num, timestamp_size=timestamp_num,
    #                      output_size=cell_num, latent_size=latent_num,
    #                      binding_mode='mul')
    #
    # model.load_state_dict(weight_dict)

    position_encoding = weight_dict['position_encoding.weight'].detach().cpu().numpy()
    timestamp_encoding = weight_dict['timestamp_encoding.weight'].detach().cpu().numpy()
    latent_encoding = weight_dict['latent_projection.weight'].detach().cpu().numpy()
    print(position_encoding.shape)
    ind = np.argsort(np.argmax(position_encoding[:, :21], 1))
    # ind = np.argsort(np.argmax(position_encoding[:, 21:], 1))
    # ind = np.argsort(np.argmax(position_encoding[:, :], 1))
    position_encoding = position_encoding[ind, :]
    plt.imshow(position_encoding)
    plt.colorbar()
    plt.show()

    # all cell in day1
    # for i in range(14):
    #     img = latent_encoding[i * 100:(i + 1) * 100, :]
    #     img_range = np.max(np.abs(img))
    #     plt.imshow(img, cmap='bwr', vmin=-img_range, vmax=img_range)
    #     plt.colorbar()
    #     plt.show()

    # weight by time
    # for t in range(59):
    #     timed_weight = latent_encoding * timestamp_encoding[:, t]
    #     i = 0
    #     img = timed_weight[i * 100:(i + 1) * 100, :]
    #     img_range = np.max(np.abs(img))
    #     plt.imshow(img, cmap='bwr', vmin=-img_range, vmax=img_range)
    #     plt.colorbar()
    #     plt.show()


if __name__ == '__main__':
    main()
