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
from net import *
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


def main(folding_id):
    G = Generator(128)
    G.load_state_dict(torch.load("generator_mod1_param_fold_%d.pkl" % folding_id))
    setup(G)
    G.eval()

    fake_mnist = []
    batch_size = 128
    with torch.no_grad():
        for i in range(32768 * 4 // batch_size):
            z_ = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
            z_ = Variable(z_)

            x_fake = G(z_).squeeze().view(-1, 1, 32, 32)

            for j in range(batch_size):
                fake_mnist.append(x_fake[j])

    print("Fake set size: %d" % len(fake_mnist))

    fake_mnist = np.stack(fake_mnist)

    output = open('data_fakes_%d.pkl' % folding_id, 'wb')
    pickle.dump(fake_mnist, output)
    output.close()

    image = np.squeeze(merge(fake_mnist[:10000, :, :].reshape(-1, 32, 32, 1), (100, 100)))
    scipy.misc.imsave("fakes_mod1.png", image)

if __name__ == '__main__':
    main(0)
