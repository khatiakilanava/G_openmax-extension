#!/usr/bin/env python
import argparse
import os
import sys
from pprint import pprint
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from training import train_gan
from networks import build_networks, save_networks, get_optimizers
networks = build_networks(10)  # we have 10 classes
train_epoch=7
optimizers = get_optimizers(networks)
print('GAN training start!')

for epoch in range(train_epoch):
    train_gan(networks, optimizers, fold, class_fold)
    #generate_counterfactual(networks, dataloader, **options)
    
    