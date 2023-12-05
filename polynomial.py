import numpy as np
from matplotlib import pyplot as plt


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def main():
    # generate 5 random 4th order polynomial lines
    for i in range(10):
        x = np.linspace(-3, 3, 100)
        y = np.random.normal(0, 2, 5)
        print(y)
        plt.plot(x,
                 y[0] * x ** 4 / factorial(4) + y[1] * x ** 3 / factorial(3) + y[2] * x ** 2 / factorial(2) + y[3] * x +
                 y[4])
    plt.show()


if __name__ == '__main__':
    main()
