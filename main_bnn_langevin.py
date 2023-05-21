'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import sys
sys.path.append('..')
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from torch.autograd import Variable
import numpy as np
import random
import utils
import samplers_bnn

parser = argparse.ArgumentParser(description='CIFAR10 Training')
parser.add_argument(
    "--dataset", type=str, default="CIFAR10", help="dataset name: CIFAR10/100")
parser.add_argument('--sampler', type=str, default='o_langevin')    
parser.add_argument('--dir', type=str, default='./checkpoints', help='path to save checkpoints (default: None)')
parser.add_argument('--data_path', type=str, default='../data', metavar='PATH',
                    help='path to datasets location (default: None)')               
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.1)')                    
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay')                           
parser.add_argument('--momentum', type=float, default=0,
                    help='0: SGLD; 0<m<1: SGHMC')
parser.add_argument('--temperature', type=float, default=1.,
                    help='Temperature')  
parser.add_argument('--alpha', type=float, default=1000,
                    help='sigma') 
parser.add_argument('--beta', type=float, default=1.,
                    help='sigma')                                                          
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
print('alpha',args.alpha, 'lr',args.lr, 'temperature',args.temperature)
# Data
print('==> Preparing data..')
trainloader,testloader = utils.get_data(args.dataset, args.data_path, args.batch_size, num_workers=0)
# Model
print('==> Building model..')
num_classes_dict = {
    "CIFAR10": 10,
    "CIFAR100": 100,
}
datasize=50000
model = ResNet18(num_classes=num_classes_dict[args.dataset])
if use_cuda:
    model.cuda()
    cudnn.benchmark = True
    cudnn.deterministic = True

# Sampler
if args.sampler == 'o_langevin':
    sampler = samplers_bnn.O_Langevin(model, alpha = args.alpha, beta=args.beta, wd=args.wd, datasize=datasize)   

def adjust_learning_rate(epoch, batch_idx):
    if epoch<75:
        lr = args.lr
    elif epoch >=75 and epoch <150:
        lr = args.lr*0.1
    else:
        lr = args.lr*0.01
    return lr

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    trainf = 0
    traing = 0
    correct = 0
    total = 0
    entropy_sum = 0
    epsilon = [0,0]
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        model.zero_grad()
        lr = adjust_learning_rate(epoch,batch_idx)
        outputs = model(inputs)
        floss = criterion(outputs, targets)
        gloss = floss
        sampler.step(floss,gloss,lr,epoch)
        trainf += floss.data.item()
        traing += gloss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        if batch_idx%100==0:
            print('fLoss: %.3f | lr: %s | Acc: %.3f%% (%d/%d)'
                % (trainf/(batch_idx+1),lr, 100.*correct/total, correct, total))
    return trainf/(batch_idx+1)

def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx%100==0:
                print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss/len(testloader), correct, total,
    100. * correct / total))
    return test_loss/len(testloader)

criterion = nn.CrossEntropyLoss()
mt = 0
train_ll_list = []
test_ll_list = []
for epoch in range(args.epochs):
    train_ll = train(epoch)
    test_ll = test(epoch)
    train_ll_list.append(train_ll)
    test_ll_list.append(test_ll)
