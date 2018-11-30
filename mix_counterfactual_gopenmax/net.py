import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.misc

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if images.shape[3] in (3,4):
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
      img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter must have dimensions: HxW or HxWx3 or HxWx4')


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper


class Generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(10, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*2, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):#, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))

        return x


class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(1, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d // 2, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.sigmoid(self.conv4(x))
        return x


class AvgDiscriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(AvgDiscriminator, self).__init__()
        self.avg = torch.nn.AvgPool2d(4, 2, 1, count_include_pad=False)
        self.conv1 = nn.Conv2d(1, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d // 2, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, 1, 4, 2, 0)

        # self.avg = torch.nn.AvgPool2d(4, 2, 1, count_include_pad=False)
        # self.conv1 = nn.Conv2d(1, d//2, 4, 2, 1)
        # self.conv2 = nn.Conv2d(d // 2, d*2, 4, 2, 1)
        # self.conv2_bn = nn.BatchNorm2d(d*2)
        # self.conv3 = nn.Conv2d(d*2, 1, 4, 2, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):#, label):
        avg = self.avg(input)

        #image = np.squeeze(merge(avg.cpu().data.view(-1, 16, 16, 1).numpy() + 0.5, (16, 8)))
        #scipy.misc.imsave("avg.png", image)

        x = F.leaky_relu(self.conv1(avg), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.sigmoid(self.conv3(x))

        return x


class ClassifierBMVC(nn.Module):
    # initializers
    def __init__(self, c, return_penultimate=False, d=128):
        super(ClassifierBMVC, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 1024, 8, 8, 0)
        self.conv4 = nn.Conv2d(1024, c, 1, 1, 0)

        self.return_penultimate = return_penultimate

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.max_pool2d(x, 2, 2, 0)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.max_pool2d(x, 2, 2, 0)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.dropout(x, 0.5, self.training)
        x = self.conv4(x)

        if self.return_penultimate:
            return x
        return F.log_softmax(x, 1)


class CGenerator(nn.Module):
    # initializers
    def __init__(self, c, d=128):
        super(CGenerator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(c, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))

        return x

class CDiscriminator(nn.Module):
    # initializers
    def __init__(self, c,  d=128):
        super(CDiscriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(1, d//2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(c, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.sigmoid(self.conv4(x))
        return x


class Classifier(nn.Module):
    # initializers
    def __init__(self, c, return_penultimate=False, d=128):
        super(Classifier, self).__init__()
        self.conv1_1 = nn.Conv2d(1, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d // 2, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, c, 4, 1, 0)
        self.conv5 = nn.Conv2d(c, d//4, 1, 1, 0)
        self.conv4_bn = nn.BatchNorm2d(c)
        self.conv5_bn = nn.BatchNorm2d(d//4)
        self.conv6 = nn.Conv2d(d//4, 1, 1, 1, 0)
        self.return_penultimate = return_penultimate

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv4(x)
        if self.return_penultimate:
            return x
        return F.log_softmax(x)


class ClassifierWithRejection(nn.Module):
    # initializers
    def __init__(self, c, d=128):
        super(ClassifierWithRejection, self).__init__()
        self.conv1_1 = nn.Conv2d(1, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d // 2, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, c, 4, 1, 0)
        self.conv5 = nn.Conv2d(c, d//16, 1, 1, 0)
        self.conv4_bn = nn.BatchNorm2d(c)
        self.conv5_bn = nn.BatchNorm2d(d//16)
        self.conv6 = nn.Conv2d(d//16, 1, 1, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.leaky_relu(x, 0.2)
        x1 = self.conv4(x)

        x = self.conv5(x1)
        x = self.conv5_bn(x)
        x = self.conv6(F.leaky_relu(x, 0.2))
        return F.log_softmax(x1), F.sigmoid(x)


class PatchDiscriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(PatchDiscriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(1, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d // 2, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, 1, 4, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):#, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        batch_size = x.size()[0]

        x = self.conv3(x)

        x = F.sigmoid(x).view(batch_size, -1).mean(1)
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
