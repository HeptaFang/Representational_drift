import time
import sys

from path_check import main as path_check
from regularization_search import main as regularization_search


def main():
    """
    Hub script. Run all the processes in proper order.
    """
    path_check()
    l1_idx, l2_idx = int(sys.argv[1]), int(sys.argv[2])
    regularization_search(l1_idx, l2_idx)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time: {end - start}')
