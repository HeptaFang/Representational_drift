from functools import partial
import matplotlib.pyplot as plt
import torch
from model import BindingModel, load_model
import numpy as np
import json
from torch.utils.data import TensorDataset, DataLoader

EPOCH_NUM = 1000


def binding_regularization(model, lambda_position=1e-4, lambda_timestamp=0.0,
                           lambda_position_smooth=1e-4, lambda_timestamp_smooth=0.0,
                           lambda_latent=1e-6):
    # return 0, 0, 0
    # L2 regularization for position & timestamp
    position_l2_regularization = lambda_position * torch.linalg.vector_norm(model.position_encoding.weight, 2)
    timestamp_l2_regularization = lambda_timestamp * torch.linalg.vector_norm(model.timestamp_encoding.weight, 2)

    # smooth regularization for position & timestamp encoding
    position_diff_weight = torch.diff(model.position_encoding.weight, dim=1)
    position_smooth_regularization = lambda_position_smooth * torch.linalg.vector_norm(position_diff_weight[:20], 2)
    position_smooth_regularization += lambda_position_smooth * torch.linalg.vector_norm(position_diff_weight[-20:], 2)
    timestamp_diff_weight = torch.diff(model.timestamp_encoding.weight, dim=1)
    timestamp_smooth_regularization = lambda_timestamp_smooth * torch.linalg.vector_norm(timestamp_diff_weight, 2)

    # L1 regularization for latent space transformation
    if model.use_latent:
        latent_l1_regularization = lambda_latent * torch.linalg.vector_norm(model.latent_projection.weight, 1)
    else:
        latent_l1_regularization = 0

    l2 = position_l2_regularization + timestamp_l2_regularization
    smooth = position_smooth_regularization + timestamp_smooth_regularization
    l1 = latent_l1_regularization

    return l2, smooth, l1


def nan_MSEloss(y_pred, y):
    mask = torch.isnan(y)
    y = torch.where(mask, torch.tensor(0.0), y)
    y_pred = torch.where(mask, torch.tensor(0.0), y_pred)
    compensate = 1 - torch.sum(mask) / torch.numel(y)

    loss = (y_pred - y) ** 2
    return torch.mean(loss) / compensate


def train_model(task_name, train_mode, from_epoch=0, to_epoch=1000, regularization_paras=None, full_batch=False):
    use_selected_cell = True
    # load dataset
    if use_selected_cell:
        position = np.load('dataset\\' + task_name + '_position_shuffled_selected.npy')
        timestamp = np.load('dataset\\' + task_name + '_timestamp_shuffled_selected.npy')
        activity = np.load('dataset\\' + task_name + '_activity_shuffled_selected.npy')
    else:
        position = np.load('dataset\\' + task_name + '_position_shuffled.npy')
        timestamp = np.load('dataset\\' + task_name + '_timestamp_shuffled.npy')
        activity = np.load('dataset\\' + task_name + '_activity_shuffled.npy')

    bin_num = position.shape[1]
    session_num = timestamp.shape[1]
    cell_num = activity.shape[1]
    trial_num = activity.shape[0]

    train_num = int(trial_num * 0.95)
    test_num = trial_num - train_num

    position_train = position[:train_num]
    timestamp_train = timestamp[:train_num]
    activity_train = activity[:train_num]
    position_test = position[train_num:]
    timestamp_test = timestamp[train_num:]
    activity_test = activity[train_num:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # load model, create new if from_epoch=0
    model_name = 'Binding'
    model = load_model(bin_num, session_num, cell_num, train_mode, model_name, task_name, epoch=from_epoch)

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

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nan_MSEloss
    regularization_fn = binding_regularization
    if regularization_paras is not None:
        regularization_fn = partial(binding_regularization, **regularization_paras)

    train_loss = np.zeros((training_epoch, 4))
    test_loss = np.zeros(training_epoch)

    for i in range(training_epoch):
        train_loss[i] = train_loop(dataloader, model, loss_fn, regularization_fn, optimizer, device, full_batch)
        test_loss[i] = test(model, tensor_position_test, tensor_timestamp_test, tensor_activity_test, loss_fn, device)
        if (i + 1) % 10 == 0:
            if regularization_paras['lambda_latent'] > 0:
                torch.save(model.state_dict(), f'model\\{model_name}_{train_mode}_{task_name}_{i + 1 + from_epoch}_reg.m')
            else:
                torch.save(model.state_dict(), f'model\\{model_name}_{train_mode}_{task_name}_{i + 1 + from_epoch}.m')
        print(
            f"Epoch {i + 1 + from_epoch}/{to_epoch}\ntrain_loss: {np.sum(train_loss[i]):>5f} = {train_loss[i][0]:>5f} + {train_loss[i][1]:>5f} + {train_loss[i][2]:>5f} + {train_loss[i][3]:>5f}\ntest_loss: {test_loss[i]:>7f}")

        # if i % 10 == 0:
        #     idx = np.random.randint(0, 4095)
        #     test_x = x[idx, :, :]
        #     test_y = y[idx, :, :]
        #     y_pred = model(torch.Tensor(test_x).to(device))
        #     plt.clf()
        #     plt.plot(test_x[:, 0])
        #     plt.plot(test_y[:, 0])
        #     plt.plot(y_pred[:, 0].cpu().detach().numpy())
        #     plt.legend(['x', 'y', 'pred'])
        #     plt.savefig(f'image\\{task_name}\\{i}.jpg')

    # plt.plot(train_loss)
    # plt.show()
    return train_loss, test_loss


def train_loop(dataloader, model, loss_fn, regularization_fn, optimizer, device, full_batch=False):
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


# def main():
#     train_loss = np.zeros((5, 3, EPOCH_NUM, 4))
#     total_train_loss = np.zeros((5, 3, EPOCH_NUM))
#     test_loss = np.zeros((5, 3, EPOCH_NUM))
#     modes = ['MultiWithLatent', 'Additive', 'Multiplicative']
#     # modes = ['MultiWithLatent', 'MultiWithLatentMedium', 'MultiWithLatentLarge']
#     data_label = 'MultiEncoding'
#     # data_label = 'MultiSizeLatent'
#     for i in range(5):
#         for j in range(3):
#             train_mode = modes[j]
#             name = f'mouse{i + 1}'
#             train_loss[i, j], test_loss[i, j] = train_model(name, train_mode)
#
#         total_train_loss[i] = np.sum(train_loss[i], 2)
#         plt.clf()
#         colors = 'rgb'
#         for j in range(3):
#             plt.plot(total_train_loss[i, j], f'{colors[j]}-')
#         for j in range(3):
#             plt.plot(test_loss[i, j], f'{colors[j]}:')
#         plt.title(f'mouse{i + 1}')
#         plt.xlabel('training progress (epoch)')
#         plt.ylabel('loss')
#
#         plt.legend(
#             ['MultiWithLatent_train', 'Additive_train', 'Multiplicative_train',
#              'MultiWithLatent_test', 'Additive_test', 'Multiplicative_test'])
#         # plt.legend(['64_train', '128_train', '256_train', '64_test', '128_test', '256_test'])
#
#         plt.savefig(f'image\\trainingLoss_{data_label}_mouse{i + 1}')
#
#     np.save(f'analysis\\train_loss_{data_label}.npy', train_loss)
#     np.save(f'analysis\\test_loss_{data_label}.npy', test_loss)

def main():
    np.random.seed(113)
    torch.manual_seed(308)
    tasks = ['mouse1', 'mouse2', 'mouse3', 'mouse4', 'mouse5']
    train_modes = ['Additive', 'Multiplicative', 'MultiWithLatent']
    reg_epoch = 500
    max_epoch = 5000

    for task_name in tasks:
        for train_mode in train_modes:
            print(f'Training {task_name} with {train_mode} Phase Basic')
            regularization_paras = {'lambda_position': 0.0, 'lambda_timestamp': 0.0,
                                    'lambda_position_smooth': 0.0, 'lambda_timestamp_smooth': 0.0,
                                    'lambda_latent': 0.0}
            train_loss, test_loss = train_model(task_name, train_mode, from_epoch=0, to_epoch=reg_epoch,
                                                regularization_paras=regularization_paras)
            np.save(f'analysis\\train_loss_{task_name}_{train_mode}_{0}_{reg_epoch}.npy', train_loss)
            np.save(f'analysis\\test_loss_{task_name}_{train_mode}_{0}_{reg_epoch}.npy', test_loss)

            print(f'Training {task_name} with {train_mode} Phase Extend')
            train_loss, test_loss = train_model(task_name, train_mode, from_epoch=reg_epoch, to_epoch=max_epoch,
                                                regularization_paras=regularization_paras)
            np.save(f'analysis\\train_loss_{task_name}_{train_mode}_{reg_epoch}_{max_epoch}.npy', train_loss)
            np.save(f'analysis\\test_loss_{task_name}_{train_mode}_{reg_epoch}_{max_epoch}.npy', test_loss)

            regularization_paras = {'lambda_position': 1e-3, 'lambda_timestamp': 1e-3,
                                    'lambda_position_smooth': 2e-3, 'lambda_timestamp_smooth': 0.0,
                                    'lambda_latent': 3e-5}
            # regularization_paras = {'lambda_position': 2e-4, 'lambda_timestamp': 2e-4,
            #                         'lambda_position_smooth': 2e-3, 'lambda_timestamp_smooth': 0.0,
            #                         'lambda_latent': 1e-5}
            print(f'Training {task_name} with {train_mode} Phase Regularize')
            train_loss, test_loss = train_model(task_name, train_mode, from_epoch=reg_epoch, to_epoch=max_epoch,
                                                regularization_paras=regularization_paras)
            np.save(f'analysis\\train_loss_{task_name}_{train_mode}_{reg_epoch}_{max_epoch}_regularize.npy', train_loss)
            np.save(f'analysis\\test_loss_{task_name}_{train_mode}_{reg_epoch}_{max_epoch}_regularize.npy', test_loss)


if __name__ == '__main__':
    main()
