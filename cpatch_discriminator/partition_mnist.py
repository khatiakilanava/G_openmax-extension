from utils import mnist_reader
from utils.download import download
import random
import math
import pickle
import json


def main():
    download(directory="mnist", url="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", extract_gz=True)
    download(directory="mnist", url="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", extract_gz=True)
    download(directory="mnist", url="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", extract_gz=True)
    download(directory="mnist", url="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", extract_gz=True)

    folds = 5

    # Split mnist into 5 folds:
    mnist = items_train = mnist_reader.Reader('mnist', train=True, test=True).items
    class_bins = {}
    random.shuffle(mnist)

    for x in mnist:    # separates the classes into separate bins and then partitions the same number of different classes
        if x[0] not in class_bins: #into 5 folds. So in each fold for instance we have 10 samples of 
            class_bins[x[0]] = []#class 0 and 20 samples class 9 etc and 40 from class 5...
        class_bins[x[0]].append(x)
    mnist_folds = [[] for _ in range(folds)]
    for _class, data in class_bins.items():
        count = len(data)
        print("Class %d count: %d" % (_class, count))
        count_per_fold = count // folds
        for i in range(folds):
            mnist_folds[i] += data[i * count_per_fold: (i + 1) * count_per_fold]
    print("Folds sizes:")

    for i in range(len(mnist_folds)):
        print(len(mnist_folds[i]))
        output = open('data_fold_%d.pkl' % i, 'wb')
        pickle.dump(mnist_folds[i], output)
        output.close()

if __name__ == '__main__':
    main()
