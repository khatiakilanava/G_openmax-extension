import os, time
import matplotlib.pyplot as plt
import pickle
import imageio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import scipy.misc
from net import *  #imports discriminator and generator
import math
import json
import random

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


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter must have dimensions: HxW or HxWx3 or HxWx4')


fixed_z_ = numpy2torch(np.random.normal(0.0, 0.5, (60, 100)).astype(np.float32))
fixed_z_ = Variable(fixed_z_.view(-1, 100, 1, 1), volatile=True)

fixed_y_ = numpy2torch(np.repeat(np.arange(0, 6, dtype=np.int).reshape([6, 1]), 10, 0)).type(LongTensor)
fixed_y_label_ = torch.zeros(60, 6)
fixed_y_label_.scatter_(1, fixed_y_, 1.0)
fixed_y_label_ = Variable(setup(fixed_y_label_).view(-1, 6, 1, 1), volatile=True)


def show_result(G, path='result.png'):
    G.eval()
    images = G(fixed_z_, fixed_y_label_)
    G.train()

    images = images.cpu().data.view(-1, 32, 32, 1).numpy()

    images += 0.5

    image = np.squeeze(merge(images, (6, 10)))
    scipy.misc.imsave(path, image)


def show_train_hist(hist, show = False, save = False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    yd = hist['D_losses']
    yg = hist['G_losses']

    plt.plot(x, yd, label='D_loss')
    plt.plot(x, yg, label='G_loss')

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
    batch_size = 64
    lr = 0.0002
    train_epoch = 7

    mnist_train = []

    class_data = json.load(open('class_table_fold_%d.txt' % class_fold))

    train_classes = class_data[0]["train"]

    for i in range(folds):
        if i != folding_id:
            with open('data_fold_%d.pkl' % i, 'rb') as pkl:
                fold = pickle.load(pkl)
            mnist_train += fold

    #keep only train classes
    mnist_train = [x for x in mnist_train if x[0] in train_classes]

    print("Train set size:", len(mnist_train))

    def pad(array, padding_size):
        if len(array) % padding_size != 0:
            padding = padding_size - len(array) % padding_size
            array += array[:padding]

    pad(mnist_train, 1024)

    print("After padding, train set size:", len(mnist_train))

    mapClassID = lambda x: train_classes.index(x)

    mnist_train_x = np.asarray([x[1] for x in mnist_train], np.float32)
    mnist_train_y = np.asarray([mapClassID(x[0]) for x in mnist_train], np.int)

    # network
    G = CGenerator(len(train_classes))
    D = PatchDiscriminator()

    G.weight_init(mean=0, std=0.02)
    D.weight_init(mean=0, std=0.02)
    setup(G)
    setup(D)

    # Binary Cross Entropy loss
    BCE_loss = nn.BCELoss()

    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # results save folder
    if not os.path.isdir('MNIST_BMVC_GUN_results'):
        os.mkdir('MNIST_BMVC_GUN_results')
    if not os.path.isdir('MNIST_BMVC_GUN_results/Fixed_results'):
        os.mkdir('MNIST_BMVC_GUN_results/Fixed_results')

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    print('training start!')
    start_time = time.time()

    y_real_ = torch.ones(batch_size)
    y_fake_ = torch.zeros(batch_size)

    y_real_, y_fake_ = Variable(y_real_), Variable(y_fake_)


    def shuffle_in_unison(a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

    onehot = torch.zeros(len(train_classes), len(train_classes))
    onehot = onehot.scatter_(1, LongTensor([x for x in range(len(train_classes))]).view(len(train_classes), 1), 1).view(len(train_classes), len(train_classes), 1, 1)

    fill = torch.zeros([6, 6, 32, 32])
    for i in range(6):
        fill[i, i, :, :] = 1

    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []

        # learning rate decay
        if (epoch+1) % 20 == 0:
            G_optimizer.param_groups[0]['lr'] /= 2
            D_optimizer.param_groups[0]['lr'] /= 2
            print("learning rate change!")

        epoch_start_time = time.time()

        mnist_train_x, mnist_train_y = shuffle_in_unison(mnist_train_x, mnist_train_y)

        for it in range(len(mnist_train_x) // batch_size):

            x_ = mnist_train_x[it * batch_size:(it + 1) * batch_size, :, :]
            y_ = mnist_train_y[it * batch_size:(it + 1) * batch_size]

            x_ = numpy2torch(x_) / 255.0
            x_.sub_(0.5).div_(0.5)
            x_ = Variable(x_)
            x_ = x_.view(-1, 1, 32, 32)
            y_ = numpy2torch(y_).type(LongTensor)
            y_fill_ = Variable(fill[y_])

            mini_batch = x_.size()[0]

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = Variable(z_)
            y_c = (torch.rand(mini_batch, 1) * len(train_classes)).type(LongTensor).squeeze()
            y_label_ = Variable(onehot[y_c])

            x_fake = Variable(G(z_, y_label_).data)

            D.zero_grad()

            D_result = D(x_).squeeze() #y_fill_
            D_real_loss = BCE_loss(D_result, y_real_)

            y_fill_ = Variable(fill[y_c])

            D_result = D(x_fake).squeeze()  #y_fill_

            D_fake_loss = BCE_loss(D_result, y_fake_)

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            DLoss = D_train_loss.item()

            D_losses.append(DLoss)

            # train generator G
            G.zero_grad()

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = Variable(z_)
            y_c = (torch.rand(mini_batch, 1) * len(train_classes)).type(LongTensor).squeeze()
            y_label_ = Variable(onehot[y_c])

            y_fill_ = Variable(fill[y_c])

            x_fake = G(z_, y_label_)
            D_result = D(x_fake).squeeze() #y_fill_

            G_train_loss = BCE_loss(D_result, y_real_)

            G_train_loss.backward()
            G_optimizer.step()

            G_losses.append(G_train_loss.item())

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time


        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime,
                                                                  torch.mean(torch.FloatTensor(D_losses)),
                                                                  torch.mean(torch.FloatTensor(G_losses))))
        fixed_p = 'MNIST_BMVC_GUN_results/Fixed_results/MNIST_bmvcgun_' + str(epoch + 1) + '.png'
        show_result(G, path=fixed_p)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
    print("Training finish!... save training results")
    torch.save(G.state_dict(), "MNIST_BMVC_GUN_results/generator_param.pkl")
    torch.save(D.state_dict(), "MNIST_BMVC_GUN_results/discriminator1_param.pkl")

    torch.save(G.state_dict(), "generator_BMVC_param_fold_%d.pkl" % folding_id)

    with open('MNIST_BMVC_GUN_results/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path='MNIST_BMVC_GUN_results/MNIST_cGAN_train_hist.png')

    images = []
    for e in range(train_epoch):
        img_name = 'MNIST_BMVC_GUN_results/Fixed_results/MNIST_bmvcgun_' + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave('MNIST_BMVC_GUN_results/generation_animation.gif', images, fps=5)


if __name__ == '__main__':
    main(0, 0)
    main(1, 0)
    main(2, 0)
    main(3, 0)
    main(4, 0)


