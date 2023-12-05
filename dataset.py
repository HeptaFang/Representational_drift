import os
import numpy as np
import matplotlib.pyplot as plt
from METAPARAMETERS import *


def generate_dataset(noise_level, bias, plot=False, override_fit_order=-1):
    # generate random encoding matrices
    position_encoding = np.random.normal(0, 1, (BIN_NUM, 1, HIDDEN_NUM))
    timestamp_encoding = np.random.normal(0, 1, (1, SESSION_NUM, HIDDEN_NUM))

    if override_fit_order != -1:
        fit_order = override_fit_order
    else:
        fit_order = FIT_ORDER

    # fit the encoding with polynomial
    if fit_order is not None:
        position_x = np.linspace(-1, 1, BIN_NUM)
        timestamp_x = np.linspace(-1, 1, SESSION_NUM)
        position_encoding_fit = np.zeros((BIN_NUM, 1, HIDDEN_NUM))
        timestamp_encoding_fit = np.zeros((1, SESSION_NUM, HIDDEN_NUM))
        for i in range(HIDDEN_NUM):  # for each hidden unit
            position_encoding_fit[:, 0, i] = np.polyval(np.polyfit(position_x, position_encoding[:, 0, i], fit_order),
                                                        position_x)
            timestamp_encoding_fit[0, :, i] = np.polyval(
                np.polyfit(timestamp_x, timestamp_encoding[0, :, i], fit_order),
                timestamp_x)
        # normalize fitting variance to 1
        position_encoding_fit = position_encoding_fit / np.sqrt(np.var(position_encoding_fit))
        timestamp_encoding_fit = timestamp_encoding_fit / np.sqrt(np.var(timestamp_encoding_fit))

        if plot:
            # plot the encoding
            print('plotting encoding')
            for i in range(HIDDEN_NUM):
                plt.figure()
                plt.subplot(2, 1, 1)
                plt.plot(position_x, position_encoding[:, 0, i])
                plt.plot(position_x, position_encoding_fit[:, 0, i])
                plt.legend(['original', 'fit'])
                plt.title('position encoding')
                plt.subplot(2, 1, 2)
                plt.plot(timestamp_x, timestamp_encoding[0, :, i])
                plt.plot(timestamp_x, timestamp_encoding_fit[0, :, i])
                plt.legend(['original', 'fit'])
                plt.title('timestamp encoding')
                # plt.show()
                plt.suptitle(f'encoding_dim{i}_{noise_level:.1f}_{bias:.1f}')
                plt.savefig(os.path.join(GLOBAL_PATH, 'image', 'artificial_dataset', 'dataset_detail',
                                         f'encoding_dim{i}_{noise_level:.1f}_{bias:.1f}.png'))
                plt.close()

            print('plotting distribution')
            # plot value distribution, range from -4 to 4, show standard normal distribution
            x = np.linspace(-4, 4, 100)
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.hist(position_encoding.reshape(-1), bins=100, range=(-4, 4), density=True)
            plt.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi))
            plt.title('position encoding')
            plt.subplot(2, 2, 2)
            plt.hist(timestamp_encoding.reshape(-1), bins=100, range=(-4, 4), density=True)
            plt.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi))
            plt.title('timestamp encoding')
            plt.subplot(2, 2, 3)
            plt.hist(position_encoding_fit.reshape(-1), bins=100, range=(-4, 4), density=True)
            plt.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi))
            plt.title('position encoding fit')
            plt.subplot(2, 2, 4)
            plt.hist(timestamp_encoding_fit.reshape(-1), bins=100, range=(-4, 4), density=True)
            plt.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi))
            plt.title('timestamp encoding fit')
            # plt.show()
            plt.suptitle(f'encoding distribution_{noise_level:.1f}_{bias:.1f}')
            plt.savefig(os.path.join(GLOBAL_PATH, 'image', 'artificial_dataset', 'dataset_detail',
                                     f'encoding_distribution_{noise_level:.1f}_{bias:.1f}.png'))

        position_encoding = position_encoding_fit
        timestamp_encoding = timestamp_encoding_fit

    print(f'encoding var: {np.var(position_encoding)}, {np.var(timestamp_encoding)}')
    print(f'encoding mean: {np.mean(position_encoding)}, {np.mean(timestamp_encoding)}')
    projection = np.random.normal(0, 1 / (np.sqrt(HIDDEN_NUM * SPARSENESS)), (HIDDEN_NUM, CELL_NUM))

    sparse_mask = np.random.choice([0, 1], (HIDDEN_NUM, CELL_NUM), p=[1 - SPARSENESS, SPARSENESS])
    projection = projection * sparse_mask

    binding_encoding_mul = position_encoding * timestamp_encoding
    binding_encoding_add = (position_encoding + timestamp_encoding) * (2 ** -0.5)
    print(f'binding var: {np.var(binding_encoding_mul)}, {np.var(binding_encoding_add)}')
    print(f'binding mean: {np.mean(binding_encoding_mul)}, {np.mean(binding_encoding_add)}')

    projected_mul = binding_encoding_mul @ projection
    projected_add = binding_encoding_add @ projection
    print(f'projected var: {np.var(projected_mul)}, {np.var(projected_add)}')
    print(f'projected mean: {np.mean(projected_mul)}, {np.mean(projected_add)}')

    # generate output
    noise_mul = np.random.normal(0, noise_level, (BIN_NUM, SESSION_NUM, CELL_NUM))
    noise_add = np.random.normal(0, noise_level, (BIN_NUM, SESSION_NUM, CELL_NUM))
    output_mul = projected_mul + noise_mul + bias
    output_add = projected_add + noise_add + bias
    print(f'output var: {np.var(output_mul)}, {np.var(output_add)}')
    print(f'output mean: {np.mean(output_mul)}, {np.mean(output_add)}')

    # activation function
    output_mul[output_mul < 0] = 0
    output_add[output_add < 0] = 0
    print(f'activation var: {np.var(output_mul)}, {np.var(output_add)}')
    print(f'activation mean: {np.mean(output_mul)}, {np.mean(output_add)}')

    # normalize
    print('factors:', np.std(output_mul), np.std(output_add))
    # output_mul = output_mul / np.std(output_mul)
    # output_add = output_add / np.std(output_add)
    # print('normalize:', np.var(output_mul), np.var(output_add))
    # print('mean square', np.mean(output_mul * output_mul), np.mean(output_add * output_add))

    # generate dataset
    position = np.zeros((BIN_NUM * SESSION_NUM, BIN_NUM))
    timestamp = np.zeros((BIN_NUM * SESSION_NUM, SESSION_NUM))
    activity_mul = np.zeros((BIN_NUM * SESSION_NUM, CELL_NUM))
    activity_add = np.zeros((BIN_NUM * SESSION_NUM, CELL_NUM))
    #
    # for i in range(CELL_NUM):
    #     plt.imshow(output_mul[:, :, i])
    #     plt.title(f'mul, Cell {i}')
    #     plt.show()

    for i in range(BIN_NUM):
        for j in range(SESSION_NUM):
            position[i * SESSION_NUM + j, i] = 1
            timestamp[i * SESSION_NUM + j, j] = 1
            activity_mul[i * SESSION_NUM + j] = output_mul[i, j]
            activity_add[i * SESSION_NUM + j] = output_add[i, j]

    # save dataset
    path = os.path.join(GLOBAL_PATH, 'dataset', 'artificial_dataset')
    np.save(os.path.join(path, f'mul_{noise_level:.1f}_{bias:.1f}_position.npy'), position)
    np.save(os.path.join(path, f'mul_{noise_level:.1f}_{bias:.1f}_timestamp.npy'), timestamp)
    np.save(os.path.join(path, f'mul_{noise_level:.1f}_{bias:.1f}_activity.npy'), activity_mul)
    np.save(os.path.join(path, f'mul_{noise_level:.1f}_{bias:.1f}_position_encoding.npy'), position_encoding)
    np.save(os.path.join(path, f'mul_{noise_level:.1f}_{bias:.1f}_timestamp_encoding.npy'), timestamp_encoding)
    np.save(os.path.join(path, f'mul_{noise_level:.1f}_{bias:.1f}_projection.npy'), projection)
    np.save(os.path.join(path, f'add_{noise_level:.1f}_{bias:.1f}_position.npy'), position)
    np.save(os.path.join(path, f'add_{noise_level:.1f}_{bias:.1f}_timestamp.npy'), timestamp)
    np.save(os.path.join(path, f'add_{noise_level:.1f}_{bias:.1f}_activity.npy'), activity_add)
    np.save(os.path.join(path, f'add_{noise_level:.1f}_{bias:.1f}_position_encoding.npy'), position_encoding)
    np.save(os.path.join(path, f'add_{noise_level:.1f}_{bias:.1f}_timestamp_encoding.npy'), timestamp_encoding)
    np.save(os.path.join(path, f'add_{noise_level:.1f}_{bias:.1f}_projection.npy'), projection)


def main(plot=False, override_fit_order=-1):
    np.random.seed(17)
    for noise_level in NOISE_LEVELS:
        for bias in BIAS_LEVELS:
            print()
            print(f'noise_level: {noise_level}, bias: {bias}')
            generate_dataset(noise_level, bias, plot=plot, override_fit_order=override_fit_order)


if __name__ == '__main__':
    main(plot=False)
