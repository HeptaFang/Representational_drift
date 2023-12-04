from functools import partial
import os
import torch
import numpy as np
import json
from torch.utils.data import TensorDataset, DataLoader

from model import BindingModel, load_model
from METAPARAMETERS import *

EPOCH_NUM = 1000


def binding_regularization(model, lambda_position=1e-4, lambda_timestamp=0.0,
                           lambda_position_smooth=1e-4, lambda_timestamp_smooth=0.0,
                           lambda_latent_l1=1e-6, lambda_latent_l2=0.0, lambda_bias=0.0):
    # return 0, 0, 0
    # L2 regularization
    if lambda_position != 0.0:
        position_l2_regularization = lambda_position * torch.linalg.vector_norm(model.position_encoding.weight, 2)
    else:
        position_l2_regularization = 0
    if lambda_timestamp != 0.0:
        timestamp_l2_regularization = lambda_timestamp * torch.linalg.vector_norm(model.timestamp_encoding.weight, 2)
    else:
        timestamp_l2_regularization = 0

    if model.use_latent and lambda_latent_l2 != 0.0:
        latent_l2_regularization = lambda_latent_l2 * torch.linalg.vector_norm(model.latent_projection.weight, 2)
    else:
        latent_l2_regularization = 0

    # smooth regularization for position & timestamp encoding
    if lambda_position_smooth != 0.0:
        position_diff_weight = torch.diff(model.position_encoding.weight, dim=1)
        position_smooth_regularization = lambda_position_smooth * torch.linalg.vector_norm(position_diff_weight[:20], 2)
        position_smooth_regularization += lambda_position_smooth * torch.linalg.vector_norm(position_diff_weight[-20:], 2)
    else:
        position_smooth_regularization = 0
    if lambda_timestamp_smooth != 0.0:
        timestamp_diff_weight = torch.diff(model.timestamp_encoding.weight, dim=1)
        timestamp_smooth_regularization = lambda_timestamp_smooth * torch.linalg.vector_norm(timestamp_diff_weight, 2)
    else:
        timestamp_smooth_regularization = 0

    # L1 regularization for latent space transformation
    if model.use_latent:
        latent_l1_regularization = lambda_latent_l1 * torch.linalg.vector_norm(model.latent_projection.weight, 1)
    else:
        latent_l1_regularization = 0

    # bias regularization
    if lambda_bias != 0.0:
        bias_regularization = lambda_bias * torch.sum(model.bias)
    else:
        bias_regularization = 0

    l2 = position_l2_regularization + timestamp_l2_regularization + latent_l2_regularization
    smooth = position_smooth_regularization + timestamp_smooth_regularization
    l1 = latent_l1_regularization + bias_regularization

    return l2, smooth, l1


def nan_MSEloss(y_pred, y):
    mask = torch.isnan(y)
    y = torch.where(mask, torch.tensor(0.0), y)
    y_pred = torch.where(mask, torch.tensor(0.0), y_pred)
    compensate = 1 - torch.sum(mask) / torch.numel(y)

    loss = (y_pred - y) ** 2
    return torch.mean(loss) / compensate


def train_model(task_name, train_mode, from_epoch=0, to_epoch=1000, regularization_paras=None, full_batch=False,
                log_level=0, save_interval=100, folder='dataset', model_name='Default', bias_mode='train', reconstruction=False):
    use_selected_cell = False
    # load dataset
    if use_selected_cell:
        position = np.load(os.path.join(GLOBAL_PATH, folder, task_name + '_position_shuffled_selected.npy'))
        timestamp = np.load(os.path.join(GLOBAL_PATH, folder, task_name + '_timestamp_shuffled_selected.npy'))
        activity = np.load(os.path.join(GLOBAL_PATH, folder, task_name + '_activity_shuffled_selected.npy'))
    else:
        position = np.load(os.path.join(GLOBAL_PATH, folder, task_name + '_position_shuffled.npy'))
        timestamp = np.load(os.path.join(GLOBAL_PATH, folder, task_name + '_timestamp_shuffled.npy'))
        activity = np.load(os.path.join(GLOBAL_PATH, folder, task_name + '_activity_shuffled.npy'))

    bin_num = position.shape[1]
    session_num = timestamp.shape[1]
    cell_num = activity.shape[1]
    trial_num = activity.shape[0]

    test_num = trial_num // 16
    train_num = trial_num - test_num

    position_train = position[:train_num]
    timestamp_train = timestamp[:train_num]
    activity_train = activity[:train_num]
    position_test = position[train_num:]
    timestamp_test = timestamp[train_num:]
    activity_test = activity[train_num:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # load model, create new if from_epoch=0
    model = load_model(bin_num, session_num, cell_num, train_mode, model_name, task_name, epoch=from_epoch, bias_mode=bias_mode, reconstruction=reconstruction)

    training_epoch = to_epoch - from_epoch
    model.to(device)

    # create dataloader
    tensor_position_train = torch.Tensor(position_train)
    tensor_timestamp_train = torch.Tensor(timestamp_train)
    tensor_activity_train = torch.Tensor(activity_train)
    tensor_position_test = torch.Tensor(position_test)
    tensor_timestamp_test = torch.Tensor(timestamp_test)
    tensor_activity_test = torch.Tensor(activity_test)
    dataset = TensorDataset(tensor_position_train, tensor_timestamp_train, tensor_activity_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nan_MSEloss
    regularization_fn = binding_regularization
    if regularization_paras is not None:
        regularization_fn = partial(binding_regularization, **regularization_paras)

    train_loss = np.zeros((training_epoch, 4))
    test_loss = np.zeros(training_epoch)

    for i in range(training_epoch):
        train_loss[i] = train_loop(dataloader, model, loss_fn, regularization_fn, optimizer, device, full_batch)
        test_loss[i] = test(model, tensor_position_test, tensor_timestamp_test, tensor_activity_test, loss_fn, device)
        if (i + 1) % save_interval == 0:
            torch.save(model.state_dict(),
                       os.path.join(GLOBAL_PATH, 'model',
                                    f'{model_name}_{train_mode}_{task_name}_{i + 1 + from_epoch}.m'))
            if log_level == 1:
                print(
                    f"Epoch {i + 1 + from_epoch}/{to_epoch}\ntrain_loss: {np.sum(train_loss[i]):>5f} = {train_loss[i][0]:>5f} + {train_loss[i][1]:>5f} + {train_loss[i][2]:>5f} + {train_loss[i][3]:>5f}\ntest_loss: {test_loss[i]:>7f}")

        if log_level == 2:
            print(
                f"Epoch {i + 1 + from_epoch}/{to_epoch}\ntrain_loss: {np.sum(train_loss[i]):>5f} = {train_loss[i][0]:>5f} + {train_loss[i][1]:>5f} + {train_loss[i][2]:>5f} + {train_loss[i][3]:>5f}\ntest_loss: {test_loss[i]:>7f}")

    # plt.plot(train_loss)
    # plt.show()
    return train_loss, test_loss


def train_loop(dataloader, model, loss_fn, regularization_fn, optimizer, device, full_batch=False, save_interval=100):
    size = len(dataloader.dataset)
    all_loss = []
    optimizer.zero_grad()
    total_loss = 0

    for batch, (position, timestamp, activity) in enumerate(dataloader):
        position, timestamp, activity = position.to(device), timestamp.to(device), activity.to(device)
        activity_pred = model(position, timestamp)
        activity_pred_filtered = activity_pred[:, ~torch.any(activity.isnan(), dim=0)]
        activity_filtered = activity[:, ~torch.any(activity.isnan(), dim=0)]

        task_loss = loss_fn(activity_pred, activity)
        position_reg, timestamp_reg, latent_reg = regularization_fn(model)
        loss = task_loss + position_reg + timestamp_reg + latent_reg
        if not full_batch:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            total_loss += loss
        all_loss.append((float(task_loss), float(position_reg), float(timestamp_reg), float(latent_reg)))

    # full batch training
    if full_batch:
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # average loss
    return np.mean(np.array(all_loss), 0)


def test(model, position, timestamp, activity, loss_fn, device):
    position, timestamp, activity = position.to(device), timestamp.to(device), activity.to(device)
    activity_pred = model(position, timestamp)
    # activity_pred_filtered = activity_pred[:, ~torch.any(activity.isnan(), dim=0)]
    # activity_filtered = activity[:, ~torch.any(activity.isnan(), dim=0)]

    return loss_fn(activity_pred.detach(), activity.detach())


def main():
    np.random.seed(113)
    torch.manual_seed(308)
    # tasks = ['mouse1', 'mouse2', 'mouse3', 'mouse4', 'mouse5']
    tasks = ['mouse1']
    train_modes = ['MultiWithLatent', 'AddWithLatent']
    max_epoch = 1000

    for task_name in tasks:
        for train_mode in train_modes:
            for init_mode in [0, 0.5, 1, -1]:
                print(f'Training {task_name} with {train_mode} init={init_mode}')
                regularization_paras = {'lambda_position': 0.0, 'lambda_timestamp': 0.0,
                                        'lambda_position_smooth': 0.0, 'lambda_timestamp_smooth': 0.0,
                                        'lambda_latent_l1': 0.0, 'lambda_latent_l2': 0.0, }
                train_loss, test_loss = train_model(task_name, train_mode, from_epoch=0, to_epoch=max_epoch,
                                                    regularization_paras=regularization_paras, init_mode=init_mode)
                np.save(os.path.join(GLOBAL_PATH, 'analysis',
                                     f'train_loss_{task_name}_{train_mode}_{0}_{max_epoch}_init={init_mode}.npy'),
                        train_loss)
                np.save(os.path.join(GLOBAL_PATH, 'analysis',
                                     f'test_loss_{task_name}_{train_mode}_{0}_{max_epoch}_init={init_mode}.npy'),
                        test_loss)

            # regularization_paras = {'lambda_position': 1e-3, 'lambda_timestamp': 1e-3,
            #                         'lambda_position_smooth': 2e-3, 'lambda_timestamp_smooth': 0.0,
            #                         'lambda_latent_l1': 3e-5, 'lambda_latent_l2': 1e-3, }


if __name__ == '__main__':
    main()
