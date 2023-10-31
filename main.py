import time

# from path_check import main as path_check
# from regularization_search import main as regularization_search
from dataset import main as dataset
from dataShuffle import main as dataShuffle
from train_artificial_dataset import main as train_artificial_dataset


def main():
    """
    Hub script. Run all the processes in proper order.
    """
    # path_check()
    # regularization_search()
    dataset()
    dataShuffle()
    train_artificial_dataset()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time: {end - start}')
