import time

from path_check import main as path_check
from regularization_search import main as regularization_search


def main():
    """
    Hub script. Run all the processes in proper order.
    """

    path_check()
    regularization_search()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time: {end - start}')
