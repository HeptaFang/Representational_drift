import os


def main():
    """
    Create necessary folders.
    Run this before any other script.
    """
    folders = ['model', 'dataset', 'image', 'analysis']
    print(os.listdir())
    for folder in folders:
        if folder not in os.listdir():
            os.mkdir(folder)


if __name__ == '__main__':
    main()
