import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from model import load_model
import torch

from METAPARAMETERS import *

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    for noise_level in NOISE_LEVELS:
        for bias in BIAS_LEVELS:

            # check path, create if not exist
            folder_path = f'image\\artificial_dataset\\cell_profile\\{noise_level:.1f}_{bias:.1f}'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # load dataset
            add_activity = np.load(
                'dataset\\artificial_dataset\\' + f'add_{noise_level:.1f}_{bias:.1f}' + '_activity.npy')
            mul_activity = np.load(
                'dataset\\artificial_dataset\\' + f'mul_{noise_level:.1f}_{bias:.1f}' + '_activity.npy')
            add_activity = add_activity.reshape((BIN_NUM, SESSION_NUM, CELL_NUM))
            mul_activity = mul_activity.reshape((BIN_NUM, SESSION_NUM, CELL_NUM))

            # plot stuffs
            fig = plt.figure(figsize=[14, 6])
            for i in range(CELL_NUM):
                print(f'{noise_level:.1f}_{bias:.1f} cell {i}...')
                plt.clf()
                ax_dict = fig.subplot_mosaic(
                    [
                        ['mul', 'add'], ],
                )

                im = ax_dict['mul'].imshow(mul_activity[:, :, i].T)
                ax_dict['mul'].title.set_text(f'Cell {i} mul')
                fig.colorbar(im)

                im = ax_dict['add'].imshow(add_activity[:, :, i].T)
                ax_dict['add'].title.set_text(f'Cell {i} add')
                fig.colorbar(im)

                # plt.show()
                fig.savefig(f'{folder_path}\\{i}.jpg')


if __name__ == '__main__':
    main()
