import time
import sys

from train_artificial_dataset import main as train_artificial_dataset
from METAPARAMETERS import *
from dataset import main as create_dataset
from pooling import main as pooling
from dataShuffle import main as shuffle


def generate_dataset():
    # Create artificial dataset
    create_dataset()
    if POOLED:
        pooling()
    shuffle()


def main():
    """
    Hub script. Run all the processes in proper order.
    """

    generate_dataset()

    # Training


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time: {end - start}')
