# start with a simple nonlinear function y = 2*x1**2 + x2**2
import numpy
import random


def target_func(x1, x2):
    output = 2 * x1**2 + x2**2
    return output


# training datafile path
file_path = 'training_dataset.csv'
nolines = 2000


with open(file_path, 'w') as fp:
    for i in range(nolines):
        a = 1000 * random.random() - 500
        b = 1000 * random.random() - 500
        y = target_func(a, b)
        fp.write(str(y))
        fp.write(',')
        fp.write(str(a))
        fp.write(',')
        fp.write(str(b))
        fp.write('\n')
        pass
    pass
