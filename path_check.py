import os


def main():
    folders = ['model', 'dataset', 'image', 'analysis']
    print(os.listdir())
    for folder in folders:
        if folder not in os.listdir():
            os.mkdir(folder)


if __name__ == '__main__':
    main()
