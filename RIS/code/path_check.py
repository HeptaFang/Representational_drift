from METAPARAMETERS import *
import os


def main():
    """
    Create necessary folders.
    Run this before any other script.
    """
    folders = ['model', 'dataset', 'image', 'analysis']
    print(os.listdir(GLOBAL_PATH))
    for folder in folders:
        if folder not in os.listdir(GLOBAL_PATH):
            os.mkdir(GLOBAL_PATH + folder)


if __name__ == '__main__':
    main()
