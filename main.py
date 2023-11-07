import time
import sys

from train_artificial_dataset import main as train_artificial_dataset
from METAPARAMETERS import *


def main():
    """
    Hub script. Run all the processes in proper order.
    """
    noise_idx, bias_idx, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    train_artificial_dataset(NOISE_LEVELS[noise_idx], BIAS_LEVELS[bias_idx], seed)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time: {end - start}')
