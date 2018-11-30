import os, time
import matplotlib.pyplot as plt
import pickle
import imageio
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import scipy.misc
from net import *
import math
import json
import random

#total class count, plus unseen
TOTAL_CLASS_COUNT = 10

use_cuda = torch.cuda.is_available()

FloatTensor = torch.FloatTensor
IntTensor = torch.IntTensor
LongTensor = torch.LongTensor
torch.set_default_tensor_type('torch.FloatTensor')

if use_cuda:
    device = torch.cuda.current_device()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FloatTensor = torch.cuda.FloatTensor
    IntTensor = torch.cuda.IntTensor
    LongTensor = torch.cuda.LongTensor
    print("Running on ", torch.cuda.get_device_name(device))


def setup(x):
    if use_cuda:
        return x.cuda()
    else:
        return x.cpu()


def SetGrad(p, x):
    for param in p:
        param.requires_grad = x


def numpy2torch(x):
    return setup(torch.from_numpy(x))


def computeAccuracy(probabilities, y):
    predicted = np.argmax(probabilities, 1)
    correct = np.equal(y, predicted)
    return np.sum(correct) / y.shape[0]


def computePerCalssAccuracy(probabilities, y):
    class_correct = list(0. for i in range(TOTAL_CLASS_COUNT))
    class_total = list(0. for i in range(TOTAL_CLASS_COUNT))

    predicted = np.argmax(probabilities, 1)

    correct = np.equal(y, predicted)

    for i in range(y.shape[0]):
        label = y[i]
        class_correct[label] += correct[i]
        class_total[label] += 1

    for i in range(TOTAL_CLASS_COUNT):
        if class_total[i] > 0:
            print('Accuracy of %d : %2d %%' % (
                i, 100 * class_correct[i] / class_total[i]))

    return np.sum(correct) / y.shape[0]


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def Test(data, labels, batch_size, C, train_classes):

    C.eval()
    with torch.no_grad():
        result = np.zeros((len(data), len(train_classes) + 1))

        for it in range(len(data) // batch_size):
            x_ = data[it * batch_size:(it + 1) * batch_size, :, :]
            x_ = numpy2torch(x_) / 255.0
            x_.sub_(0.5).div_(0.5)
            x_ = Variable(x_)
            x_ = x_.view(-1, 1, 32, 32)
            r = C(x_).squeeze().cpu().data.numpy()
            result[it * batch_size:(it + 1) * batch_size] = r

        accuracy = computePerCalssAccuracy(result, labels)
        print("accuracy: %f" % accuracy)


def show_train_hist(hist, show = False, save = False, path='Train_hist.png'):
    x = range(len(hist['C_losses']))

    yc1 = hist['C_losses']

    plt.plot(x, yc1, label='C_losses')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def main(folding_id, class_fold, folds=5):
    # training parameters
    batch_size = 512
    lr = 0.001
    train_epoch = 50

    mnist_train = []
    mnist_valid = []

    class_data = json.load(open('class_table_fold_%d.txt' % class_fold))

    train_classes = class_data[0]["train"]

    for i in range(folds):
        if i != folding_id:
            with open('data_fold_%d.pkl' % i, 'rb') as pkl:
                fold = pickle.load(pkl)
            if len(mnist_valid) == 0:
                mnist_valid = fold
            else:
                mnist_train += fold

    random.shuffle(mnist_train)
    random.shuffle(mnist_valid)

    #keep only train classes
    mnist_train = [x for x in mnist_train if x[0] in train_classes]
    mnist_valid = [x for x in mnist_valid if x[0] in train_classes]

    fake_class_id = len(train_classes)

    with open('data_fakes_%d.pkl' % folding_id, 'rb') as pkl:
        _fake_mnist = pickle.load(pkl)

    fake_count = int(min(_fake_mnist.shape[0], len(mnist_train) / 1.3))

    print("Fake set size: ", fake_count)

    print("Train set size: ", len(mnist_train))
    print("Valid set size: ", len(mnist_valid))

    for i in range(fake_count):
        x_fake = _fake_mnist[i, 0, :, :]
        x_fake *= 0.5
        x_fake += 0.5
        x_fake *= 255.0
        mnist_train.append((-1, x_fake))
        mnist_valid.append((-1, x_fake))

    print("Train set size: ", len(mnist_train))
    print("Valid set size: ", len(mnist_valid))

    def pad(array, padding_size):
        if len(array) % padding_size != 0:
            padding = padding_size - len(array) % padding_size
            array += array[:padding]

    pad(mnist_train, 1024)
    pad(mnist_valid, 1024)

    print("After padding, train set size:", len(mnist_train))

    mnist_train_x = np.asarray([x[1] for x in mnist_train], np.float32)
    mnist_train_y = np.asarray([x[0] for x in mnist_train], np.int)

    mnist_valid_x = np.asarray([x[1] for x in mnist_valid], np.float32)
    mnist_valid_y = np.asarray([x[0] for x in mnist_valid], np.int)

    map = lambda x: train_classes.index(x) if x in train_classes else fake_class_id
    vfunc = np.vectorize(map)

    mnist_train_y = vfunc(mnist_train_y)
    mnist_valid_y = vfunc(mnist_valid_y)

    # network
    C = ClassifierBMVC(len(train_classes) + 1, False, 128)
    setup(C)

    C_optimizer = optim.Adam(C.parameters(), lr=lr, betas=(0.5, 0.999))

    # results save folder
    if not os.path.isdir('MNIST_BMVC_GUN_results'):
        os.mkdir('MNIST_BMVC_GUN_results')

    train_hist = {}
    train_hist['C_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    print('training start!')
    start_time = time.time()

    for epoch in range(train_epoch):
        C_losses = []

        # learning rate decay
        if (epoch+1) % 20 == 0:
            C_optimizer.param_groups[0]['lr'] /= 5
            print("learning rate change!")

        epoch_start_time = time.time()

        mnist_train_x, mnist_train_y = shuffle_in_unison(mnist_train_x, mnist_train_y)

        it = 0

        def GetBatch(it):
            x_ = mnist_train_x[it * batch_size:(it + 1) * batch_size, :, :]
            y_ = mnist_train_y[it * batch_size:(it + 1) * batch_size]
            y_ = numpy2torch(y_).type(LongTensor)
            x_ = numpy2torch(x_) / 255.0
            x_.sub_(0.5).div_(0.5)
            x_ = Variable(x_)
            y_ = Variable(y_)
            x_ = x_.view(-1, 1, 32, 32)
            return x_, y_, it + 1

        C.train()
        while True:
            if it == len(mnist_train_x) // batch_size:
                break

            x_ , y_, it = GetBatch(it)

            C.zero_grad()

            result = C(x_).squeeze()

            C_loss = F.nll_loss(result, y_)

            C_loss.backward()

            C_optimizer.step()

            C_losses.append(C_loss)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - ptime: %.2f, loss: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime,
                                                                  torch.mean(torch.FloatTensor(C_losses))))

        train_hist['C_losses'].append(torch.mean(torch.FloatTensor(C_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        #if epoch % 4 == 0:
        Test(mnist_valid_x, mnist_valid_y, batch_size, C, train_classes)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
    print("Training finish!... save training results")
    torch.save(C.state_dict(), "c_GBMVCcloseset_param_fold_%d.pkl" % folding_id)

    with open('MNIST_BMVC_GUN_results/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path='MNIST_BMVC_GUN_results/MNIST_classifier_train_hist.png')

if __name__ == '__main__':
    main(0, 0)
