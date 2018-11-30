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
import libmr
import scipy.spatial.distance as spd

#total class count, plus unseen

np.seterr(all='raise')

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


def numpy2torch(x):
    return setup(torch.from_numpy(x))

fold_id = -1
fake_class_id = -1

def ExtractFeatures(data, train_classes, check_correctness=True):
    data_x = np.asarray([x[1] for x in data], np.float32)

    C = ClassifierBMVC(len(train_classes), True, 128)
    C.load_state_dict(torch.load("c_GBMVCcloseset_param_fold_%d.pkl" % fold_id))
    setup(C)
    C.eval()
    with torch.no_grad():
        result = np.zeros((len(data_x), len(train_classes)))

        batch_size = 128
        for it in range(len(data_x) // batch_size + (len(data_x) % batch_size != 0)):
            batch = batch_size
            if (it + 1) * batch_size > len(data_x):
                batch = len(data_x) - it * batch_size

            x_ = data_x[it * batch_size: it * batch_size + batch, :, :]

            x_ = numpy2torch(x_) / 255.0
            x_.sub_(0.5).div_(0.5)
            x_ = Variable(x_)
            x_ = x_.view(-1, 1, 32, 32)
            r = C(x_).squeeze().cpu().data.numpy()
            result[it * batch_size: it * batch_size + batch] = r

    if check_correctness:
        data_y = np.asarray([x[0] for x in data], np.float32)
        correct = np.equal(np.argmax(result, 1),  data_y)
        return result, correct
    else:
        return result


def distance(a, b):
    return spd.euclidean(a, b) / 200. + spd.cosine(a, b)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def recalibrate_scores(weibull_model, train_classes_and_fake, activations, mav):
    ranked_list = activations.argsort()[::-1]
    alpha = len(train_classes_and_fake)

    ranked_alpha = np.zeros(activations.shape[0])
    for i in range(len(train_classes_and_fake)):
        ranked_alpha[ranked_list[i]] = (alpha - i) / float(alpha)

    modified_activations = np.zeros(activations.shape[0])

    for i in range(len(train_classes_and_fake)):
        mr = weibull_model[i]
        d = distance(mav[i], activations)
        wscore = mr.w_score(d)
        modified_activations[i] = activations[i] * (1 - wscore * ranked_alpha[i])

    #unknown = np.sum(activations[:-1] - modified_activations[:-1])

    #modified_activations[-1] = unknown

    probabilities = softmax(modified_activations)
    return probabilities


def ComputeMAVAndMR(train_set, train_classes_and_fake, tail_size):
    f, correct = ExtractFeatures(train_set, train_classes_and_fake, True)

    mav = np.zeros((len(train_classes_and_fake), f.shape[1]))

    count = np.zeros(len(train_classes_and_fake))
    for i in range(len(train_set)):
        if correct[i]:
            id = train_set[i][0]
            mav[id] += f[i]
            count[id] += 1

    for i in range(len(train_classes_and_fake)):
        mav[i] /= count[i]

    distances = []

    for i in range(len(train_classes_and_fake)):
        distances_class = []
        for j in range(len(train_set)):
            id = train_set[j][0]

            if id == i and correct[i]:
                query = f[j]
                query_distance = distance(mav[i], query)
                distances_class += [query_distance]

        distances += [distances_class]

    weibull_model = {}

    for i in range(len(train_classes_and_fake)):
        mr = libmr.MR()
        tailtofit = sorted(distances[i])[-tail_size:]
        mr.fit_high(tailtofit, len(tailtofit))
        weibull_model[i] = mr

    return mav, weibull_model


# def GetF1(true_positive, false_positive, false_negative):
#     precision = true_positive / (true_positive + false_positive)
#     recall = true_positive / (true_positive + false_negative)
#     return 2.0 * precision * recall / (precision + recall)

def GetF1(precision, recall):
    if precision == 0.0 or recall == 0.0:
        return 0
    return 2.0 * precision * recall / (precision + recall)


def ComputeResult(dataset, train_classes_and_fake, mr, mav, threshhold=None):
    features = ExtractFeatures(dataset, train_classes_and_fake, False)

    results = []

    for i in range(len(dataset)):
        label = dataset[i][0]

        known = label != fake_class_id

        activations = features[i]
        probabilities = recalibrate_scores(mr, train_classes_and_fake, activations, mav)[:len(train_classes_and_fake) - 1]

        max_value = np.max(probabilities)

        predicted_class = np.argmax(probabilities)

        results.append((known, max_value, predicted_class, label))

    def ComputeF1(t):
        true_positive = 0
        false_positive = 0
        false_negative = 0

        known_samples = 0
        classified_as_known = 0

        for i in range(len(dataset)):
            known, max_value, predicted_class, label = results[i]

            if max_value > t:
                correct = predicted_class == label
                classified_as_known += 1
            else:
                correct = not known

            known_samples += known

            true_positive += known and correct
            false_positive += (not known) and (not correct)
            false_negative += known and (not correct)

        if classified_as_known == 0:
            return 0

        recall = true_positive / known_samples#(true_positive + false_negative)
        precision = true_positive / classified_as_known#(true_positive + false_positive)

        F1 = GetF1(precision, recall)
        return F1

    if not threshhold is None:
        return ComputeF1(threshhold), threshhold
    else:
        best_f1 = 0
        best_th_l = 0
        best_th_u = 0
        best_i_l = 0
        best_i_u = 0
        print("Start threshhold search")
        for i in range(-8000, 8000, 200):
            threshhold = 1.0 / (1.0 + np.exp(-i / 500.0))
            f = ComputeF1(threshhold)
            if f > best_f1:
                best_f1 = f
                best_th_l = threshhold
                best_i_l = i
            if f == best_f1:
                best_th_u = threshhold
                best_i_u = i

        mid_i = (best_i_u + best_i_l) // 2
        print(best_i_u, best_i_l, mid_i)
        best_f1 = 0

        for i in range(mid_i - 500, mid_i + 500, 10):
            threshhold = 1.0 / (1.0 + np.exp(-i / 500.0))
            f = ComputeF1(threshhold)
            if f > best_f1:
                best_f1 = f
                best_th_l = threshhold
                best_i_l = i
            if f == best_f1:
                best_th_u = threshhold
                best_i_u = i

        mid_i = (best_i_u + best_i_l) // 2
        print(best_i_u, best_i_l, mid_i)
        best_f1 = 0

        for i in range(mid_i - 50, mid_i + 50):
            threshhold = 1.0 / (1.0 + np.exp(-i / 500.0))
            f = ComputeF1(threshhold)
            if f > best_f1:
                best_f1 = f
                best_th_l = threshhold
                best_i_l = i
            if f == best_f1:
                best_th_u = threshhold
                best_i_u = i

        mid_i = (best_i_u + best_i_l) // 2
        print(best_i_u, best_i_l, mid_i)

        best_th = (best_th_l + best_th_u) / 2.0

        print("Best threshhold: ", best_th)
        return best_f1, best_th


def main(folding_id, opennessid, class_fold, tail_size, folds=5):
    global fold_id
    global fake_class_id

    fold_id = folding_id

    class_data = json.load(open('class_table_fold_%d.txt' % class_fold))

    train_classes = class_data[0]["train"]
    test_classes = class_data[opennessid]["test_target"]

    openness = 1.0 - math.sqrt(2 * len(train_classes) / (len(train_classes) + len(test_classes)))
    print("\tOpenness: %f" % openness)

    mnist_train = []
    mnist_valid = []

    for i in range(folds):
        if i != folding_id:
            with open('data_fold_%d.pkl' % i, 'rb') as pkl:
                fold = pickle.load(pkl)
            if len(mnist_valid) == 0:
                mnist_valid = fold
            else:
                mnist_train += fold

    with open('data_fold_%d.pkl' % folding_id, 'rb') as pkl:
        mnist_test = pickle.load(pkl)

    random.shuffle(mnist_train)
    random.shuffle(mnist_valid)

    # add k + 1, unknown class
    fake_class_id = len(train_classes)
    train_classes_and_fake = train_classes + [fake_class_id]

    mapClassID = lambda x: train_classes.index(x) if x in train_classes else fake_class_id

    #keep only train classes
    mnist_train = [(mapClassID(x[0]), x[1]) for x in mnist_train if x[0] in train_classes]

    #keep only test classes
    mnist_test = [(mapClassID(x[0]), x[1]) for x in mnist_test if x[0] in test_classes]
    mnist_valid = [(mapClassID(x[0]), x[1]) for x in mnist_valid if x[0] in test_classes]


    with open('data_fakes_%d.pkl' % folding_id, 'rb') as pkl:
        fake_mnist = pickle.load(pkl)

    for i in range(fake_mnist.shape[0]):
        x_fake = fake_mnist[i, 0, :, :]
        x_fake *= 0.5
        x_fake += 0.5
        x_fake *= 255.0
        mnist_train.append((fake_class_id, x_fake))

    mav, mr = ComputeMAVAndMR(mnist_train, train_classes_and_fake, tail_size)

    _, best_th = ComputeResult(mnist_valid, train_classes_and_fake, mr, mav)

    F1, _ = ComputeResult(mnist_test, train_classes_and_fake, mr, mav, best_th)

    print("F1: %f" % (F1))
    return F1, best_th

if __name__ == '__main__':
    main(0, 1, 0, 500)
    main(0, 2, 0, 500)
    main(0, 3, 0, 500)
    main(0, 4, 0, 500)
