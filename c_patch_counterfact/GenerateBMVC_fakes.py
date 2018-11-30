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

def get_slice(data, it, batch_size):
    return data[it * batch_size:it * batch_size + batch_size]

def shuffle_numpy(x):
    return np.take(x,np.random.permutation(x.shape[0]), axis=0)

def main(networks,folding_id,class_fold):    #     TO DOO0000000000000O need to pass the networks when calling this file build_netrowrks() punqcia am generateis gamodzaxebisas !!!!!
    mnist_train=[]
    class_data = json.load(open('class_table_fold_%d.txt' % class_fold))

    train_classes = class_data[0]["train"]
    train_classes_count=len(train_classes)
    for i in range(5):
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

    random.shuffle(mnist_train)

    mapClassID = lambda x: train_classes.index(x)
    mnist_train_x = np.asarray([x[1] for x in mnist_train], np.float32)
    gan_scale=4 
    C = ClassifierBMVC(train_classes_count, True, 64)
    C.load_state_dict(torch.load("c_closeset_param_fold_%d.pkl" % folding_id))
    setup(C)
    C.eval()
    netG = networks['generator']
    netG.load_state_dict(torch.load("generator_BMVC_param_fold_%d.pkl" % folding_id))
    setup(netG)
    netG.eval()
    netE = networks['encoder']
    setup(netE)
    netE.eval()
    fake_mnist = []
    batch_size = 128  # To doooooooooooooooooooo re run everything you have run so far for batch_size 64 afterwards  
    classification_threshold = .5

    for i in range(10):
        with torch.no_grad():
            mnist_train_x = shuffle_numpy(mnist_train_x)
            for it in range(len(mnist_train_x) // (2 * batch_size)):
                # TO DOOO000000000000000000O select 2 training samples from training fold!!!
                start_images = get_slice(mnist_train_x, it * 2, batch_size)

                end_images = get_slice(mnist_train_x, it * 2 + 1, batch_size)

                start_images=Variable(torch.from_numpy(start_images))
                start_images=start_images.unsqueeze_(1)
                start_images=start_images.repeat(1,3,1,1)
                start_images=start_images.type(FloatTensor)
                #start_images=start_images.squeeze()


                end_images=Variable(torch.from_numpy(end_images))
                
                end_images=end_images.unsqueeze_(1)
                end_images=end_images.repeat(1,3,1,1)
                end_images=end_images.type(FloatTensor)
                #end_images=end_images.squeeze()

                

                z_0 = netE(start_images, gan_scale)
                z_1 = netE(end_images, gan_scale)
                theta = np.random.uniform(size=len(z_0))
                theta = Variable(torch.FloatTensor(theta)).cuda()
                theta = theta.unsqueeze(-1)
                z_interp = theta * z_0 + (1 - theta) * z_1

                images = netG(z_interp, gan_scale).squeeze().view(-1, 1, 32, 32)
                preds = F.softmax(C(images).squeeze())
                confidence = preds.max(dim=1)[0]
                images = images.data.cpu().numpy()
                for idx, conf in enumerate(confidence.data.cpu().numpy()):
                    if conf < classification_threshold:
                        fake_mnist.append(images[idx])
            

    print("Fake set size: %d", len(fake_mnist))

    fake_mnist = np.stack(fake_mnist)

    output = open('data_fakes_%d.pkl' % folding_id, 'wb')
    pickle.dump(fake_mnist, output)
    output.close()

    image = np.squeeze(merge(fake_mnist[:10000, :, :].reshape(-1, 32, 32, 1), (100, 100)))
    scipy.misc.imsave("fakesBMVC.png", image)

if __name__ == '__main__':
    main(0)
