import time
import sys

from METAPARAMETERS import *
from dataset import main as create_dataset
from pooling import main as pooling
from dataShuffle import main as shuffle
from train_artificial_dataset import main as train_artificial_dataset


def generate_dataset(override_fit_order=None):
    # Create artificial dataset
    create_dataset(override_fit_order=override_fit_order)
    if POOLED:
        pooling()
    shuffle()


def main():
    """
    Hub script. Run all the processes in proper order.
    """

    for override_fit_order in [None, 3, 7]:
        generate_dataset(override_fit_order=override_fit_order)
        train_artificial_dataset(override_fit_order=override_fit_order)
    # Training


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time: {end - start}')
