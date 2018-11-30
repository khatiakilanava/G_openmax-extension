import time
import os
import torch
import random
import torch.nn.functional as F
from torch.autograd import Variable
import json
from vector import make_noise
import imutil
from logutil import TimeSeries
import pickle
from gradient_penalty import calc_gradient_penalty
import numpy as np
import scipy.misc

log = TimeSeries('Training GAN')
log = TimeSeries('Training GAN')
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

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def train_gan(networks, optimizers, fold_id, class_fold):
    for net in networks.values():
        net.train()
    netE = networks['encoder']
    netE=setup(netE)
    netD = networks['discriminator']
    netD=setup(netD)
    netG = networks['generator']
    netG=setup(netG)
    netC = networks['classifier_k']
    optimizerE = optimizers['encoder']
    optimizerD = optimizers['discriminator']
    optimizerG = optimizers['generator']
    optimizerC = optimizers['classifier_k']
    batch_size = 64
    latent_size = 100
    mnist_train = []
    epoch=7
    folds=5
    # results save folder
    if not os.path.isdir('MNIST_BMVC_GUN_results'):
        os.mkdir('MNIST_BMVC_GUN_results')
    if not os.path.isdir('MNIST_BMVC_GUN_results/Fixed_results'):
        os.mkdir('MNIST_BMVC_GUN_results/Fixed_results')
   
    class_data = json.load(open('class_table_fold_%d.txt' % class_fold))

    train_classes = class_data[0]["train"]
    folds=5
    for i in range(folds):
        if i != fold_id:
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

    images= np.asarray([x[1] for x in mnist_train], np.float32)
    labels= np.asarray([mapClassID(x[0]) for x in mnist_train], np.int)
    #images = Variable(mnist_train_x)
    #labels = Variable(mnist_train_y)
    #images=Variable(torch.from_numpy(images))
    #labels=Variable(torch.from_numpy(labels))
    #ac_scale = random.choice([1, 2, 4, 8])
    ac_scale = 4
    sample_scale = 4
    for it in range(len(images) // batch_size):
        x = images[it * batch_size:(it + 1) * batch_size, :, :]
        x=Variable(torch.from_numpy(x))
        x=x.unsqueeze_(1)
        x=x.repeat(1,3,1,1)
        x=x.type(FloatTensor)

        ############################
        # Discriminator Updates
        ###########################
        netD.zero_grad()

        # Classify sampled images as fake
        noise = make_noise(batch_size, latent_size, sample_scale)
        fake_images = netG(noise, sample_scale)
        logits = netD(fake_images)[:,0]
        loss_fake_sampled = F.softplus(logits).mean()
        log.collect('Discriminator Sampled', loss_fake_sampled)
        loss_fake_sampled.backward()

        """
        # Classify autoencoded images as fake
        more_images, more_labels = dataloader.get_batch()
        more_images = Variable(more_images)
        fake_images = netG(netE(more_images, ac_scale), ac_scale)
        logits_fake = netD(fake_images)[:,0]
        #loss_fake_ac = F.softplus(logits_fake).mean() * options['discriminator_weight']
        loss_fake_ac = logits_fake.mean() * options['discriminator_weight']
        log.collect('Discriminator Autoencoded', loss_fake_ac)
        loss_fake_ac.backward()
        """
    
        # Classify real examples as real
        logits = netD(x)[:,0]

        loss_real = F.softplus(-logits).mean() * 0.01  # 0.01 is discriminator weight
        #loss_real = -logits.mean() * options['discriminator_weight']
        loss_real.backward()
        log.collect('Discriminator Real', loss_real)

        gp = calc_gradient_penalty(netD, x.data, fake_images.data)
        gp.backward()
        log.collect('Gradient Penalty', gp)

        optimizerD.step()

        ############################

        ############################
        # Generator Update
        ###########################
        netG.zero_grad()

        """
        # Minimize fakeness of sampled images
        noise = make_noise(batch_size, latent_size, sample_scale)
        fake_images_sampled = netG(noise, sample_scale)
        logits = netD(fake_images_sampled)[:,0]
        errSampled = F.softplus(-logits).mean() * options['generator_weight']
        errSampled.backward()
        log.collect('Generator Sampled', errSampled)
        """

        # Minimize fakeness of autoencoded images
        fake_images = netG(netE(x, ac_scale), ac_scale)
        logits = netD(fake_images)[:,0]
        #errG = F.softplus(-logits).mean() * options['generator_weight']
        errG = -logits.mean() * 0.01 # 0.01 is generator weight
        errG.backward()
        log.collect('Generator Autoencoded', errG)

        optimizerG.step()

        ############################
        # Autoencoder Update
        ###########################
        netG.zero_grad()
        netE.zero_grad()

        # Minimize reconstruction loss
        reconstructed = netG(netE(x, ac_scale), ac_scale)
        err_reconstruction = torch.mean(torch.abs(x- reconstructed)) 
        err_reconstruction.backward()
        log.collect('Pixel Reconstruction Loss', err_reconstruction)

        optimizerE.step()
        optimizerG.step()
    ###########################

    ############################
    # Classifier Update
    ############################
    #netC.zero_grad()                                                                  # TOOOASK DO I NEEED THIS TEEEP THE CLASSIFIER HERE????????????????

    # Classify real examples into the correct K classes with hinge loss
    #classifier_logits = netC(images)
    #errC = F.softplus(classifier_logits * -labels).mean()
    #errC.backward()
    #log.collect('Classifier Loss', errC)

   # optimizerC.step()
    ############################

    # Keep track of accuracy on positive-labeled examples for monitoring
    #log.collect_prediction('Classifier Accuracy', netC(images), labels)
    #log.collect_prediction('Discriminator Accuracy, Real Data', netD(images), labels)
 
    #log.print_every()
    #SAVE RESULTS
    torch.save(netG.state_dict(), "MNIST_BMVC_GUN_results/generator_param.pkl")
    torch.save(netD.state_dict(), "MNIST_BMVC_GUN_results/discriminator1_param.pkl")

    torch.save(netG.state_dict(), "generator_BMVC_param_fold_%d.pkl" % fold_id)
    #if i % 100 == 1:
     #   fixed_noise = make_noise(batch_size, latent_size, sample_scale, fixed_seed=42)
     #   demo(networks, images, fixed_noise, ac_scale, sample_scale, result_dir, epoch, i)
    return True


def demo(networks, images, fixed_noise, ac_scale, sample_scale, result_dir, epoch=0, idx=0):
    netE = networks['encoder']
    netG = networks['generator']

    def image_filename(*args):
        image_path = os.path.join(result_dir, 'images')
        name = '_'.join(str(s) for s in args)
        name += '_{}'.format(int(time.time() * 1000))
        return os.path.join(image_path, name) + '.jpg'

    demo_fakes = netG(fixed_noise, sample_scale)
    img = demo_fakes.data[:16]

    filename = image_filename('samples', 'scale', sample_scale)
    caption = "S scale={} epoch={} iter={}".format(sample_scale, epoch, idx)
    imutil.show(img, filename=filename, resize_to=(256,256), caption=caption)

    aac_before = images[:8]
    aac_after = netG(netE(aac_before, ac_scale), ac_scale)
    img = torch.cat((aac_before, aac_after))

    filename = image_filename('reconstruction', 'scale', ac_scale)
    caption = "R scale={} epoch={} iter={}".format(ac_scale, epoch, idx)
    imutil.show(img, filename=filename, resize_to=(256,256), caption=caption)


