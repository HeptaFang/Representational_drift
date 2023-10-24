from model import load_model
import numpy as np
from matplotlib import pyplot as plt
import torch


def main():
    task_name = 'mouse1'
    train_mode = 'MultiWithLatent'
    model_name = 'Binding'

    # load dataset
    position = np.load('dataset\\' + task_name + '_position.npy')
    timestamp = np.load('dataset\\' + task_name + '_timestamp.npy')
    activity = np.load('dataset\\' + task_name + '_activity.npy')
    raw_activity = np.load('dataset\\' + task_name + '_activity_raw.npy')

    print(raw_activity.shape)

    bin_num = position.shape[1]
    session_num = timestamp.shape[1]
    cell_num = activity.shape[1]
    trial_num = activity.shape[0]

    # # search for identical cells
    # for i in range(14):
    #     plt.imshow(raw_activity[0, :, 1, i * 100:(i + 1) * 100])
    #     plt.colorbar()
    #     plt.title(f'{i * 100}to{(i + 1) * 100}')
    #     plt.show()

    # compare actual & predict
    cell_marked = np.array(
        [2, 11, 91, 144, 162, 179, 180, 181, 182, 188, 189, 288, 294, 347, 321, 365, 466, 454, 543, 565])
    cell_marked_num = len(cell_marked)
    cell_activity = raw_activity[0, :, 1, cell_marked]
    cell_activity_predicted = np.zeros((cell_marked_num, 21))
    print(cell_activity.shape)
    print(cell_activity_predicted.shape)

    model = load_model(bin_num, session_num, cell_num, train_mode, model_name, task_name, epoch=200)

    for position_num in range(21):
        position_input = np.zeros(42)
        position_input[position_num] = 1
        timestamp_input = np.zeros(59)
        timestamp_input[0] = 1

        position_input_tensor = torch.Tensor(position_input)
        timestamp_input_tensor = torch.Tensor(timestamp_input)

        predict = model(position_input_tensor, timestamp_input_tensor).detach().numpy()
        print(predict.shape)

        cell_activity_predicted[:, position_num] = predict[cell_marked]

    for i in range(cell_marked_num):
        plt.plot(cell_activity[i, :] * 18.87046138258291)  # amplify factor to be fixed
        plt.plot(cell_activity_predicted[i, :])
        plt.title(f'Cell {cell_marked[i]}')
        plt.legend(['actual', 'predicted'])
        plt.show()


if __name__ == '__main__':
    main()
