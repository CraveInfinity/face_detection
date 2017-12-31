import pandas as pd
from sklearn.metrics import accuracy_score
import os
import argparse
import logging
import cv2


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
from model import Net
import time


class NanonetDataset(Dataset):

    def __init__(self, input_dir, transform=None):
        self.transform = transform
        self.dataframe = pd.read_csv(os.path.join(input_dir, 'final.csv'))
        self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        person_1 = cv2.imread(str(self.dataframe.loc[index, 'person_1'])).astype(np.float32)/255
        person_2 = cv2.imread(str(self.dataframe.loc[index, 'person_2'])).astype(np.float32)/255
        label = self.dataframe.loc[index, 'label']

        require_dict = {'input': [person_1, person_2], 'label': label}

        if self.transform:
            require_dict = {'input': [self.transform(person_1), self.transform(person_2)], 'label': label}
            return require_dict

        return require_dict

    def __len__(self):
        return len(self.dataframe)

class PersonTransform(object):

    def __init__(self, resize):
        self.resize = resize

    def __call__(self, image, *args, **kwargs):
        image = cv2.resize(image, (self.resize, self.resize))
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)

        return image


def decay_lr(optimizer, epoch, decay_step, decay_factor):
    lr = args.lr * (decay_factor ** (epoch // decay_step))
    for param in optimizer.param_groups:
        param['lr'] = lr


def save_checkpoint(state, filename):
    torch.save(state, filename)
    print('checkpoint complete :: {}',format(filename))


def train(net, optimizer, criterion, train_loader, valid_loader):
    for samples in train_loader:
        a = samples['input'][0]
        b = samples['input'][1]
        labels = samples['label']

        optimizer.zero_grad()

        input_a, input_b = Variable(a).cuda(), Variable(b).cuda()
        labels = Variable(labels).cuda()

        outs = net([input_a, input_b])

        outs = (outs > 0.5).float()
        labels = (labels > 0.5).float()

        outs = outs.type(torch.DoubleTensor)
        labels = labels.type(torch.DoubleTensor)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()

        valid_loss = []
        valid_accuracy = []
        for valid_sample in valid_loader:
            ax = valid_sample['input'][0]
            bx = valid_sample['input'][1]
            labelsx = valid_sample['label']

            input_ax, input_bx = Variable(ax).cuda(), Variable(bx).cuda()
            labelsx = Variable(labelsx).cuda()
            optimizer.zero_grad()
            outsx = net([input_ax, input_bx])
            outsx = outsx.type(torch.FloatTensor)
            labelsx = labelsx.type(torch.FloatTensor)
            outsx = (outsx > 0.5).float()
            valid_loss.append(criterion(outsx, labelsx))
            valid_accuracy.append(accuracy_score(labelsx.data.numpy(), outsx.data.numpy()))

        print('Valid loss: {}; Valid accuracy {}'.format(sum(valid_loss)/len(valid_loader),
                                                         sum(valid_accuracy)/len(valid_loader)))

    return net, optimizer, loss




if __name__ == '__main__':

    logging.basicConfig(filename='deepnet.log', level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input_dir', type=str, action='store', dest='input_dir')
    parser.add_argument('--batch_size', type=int, action='store', dest='batch_size', default=32)
    parser.add_argument('--epochs', type=int, action='store', dest='epochs', default=20)
    parser.add_argument('--decay_step', type=int, action='store', dest='epochs', default=1)
    parser.add_argument('--decay_factor', type=float, action='store', dest='epochs', default=0.1)
    args = parser.parse_args()

    batch_size = args.batch_size
    use_gpu = torch.cuda.is_available()
    print(use_gpu)

    train_dataset = NanonetDataset(input_dir=args.input_dir,
                                   transform=PersonTransform(resize=30))

    train_loader = DataLoader(train_dataset, sampler=SubsetRandomSampler(range(1000, len(train_dataset))),
                              batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(train_dataset, sampler=SubsetRandomSampler(range(0, 1000)), batch_size=batch_size,
                              drop_last=True)

    net = Net(batch_size=batch_size)
    net.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    for epoch in range(args.epochs):
        print('Epoch {} for {} started'.format(epoch, args.epochs))
        optimizer = decay_lr(optimizer, epoch, args.decay_step, args.decay_factor)
        net, optimizer, loss = train(net, optimizer, criterion, train_loader, valid_loader)

        for valid_sample in valid_loader:
            a = valid_sample['input'][0]
            b = valid_sample['input'][1]
            labels = valid_sample['label']

            input_a, input_b = Variable(a).cuda(), Variable(b).cuda()
            labels = Variable(labels).cuda()

            outs = net([input_a, input_b])
            outs = outs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            valid_loss = criterion(outs, labels)
            outs = (outs > 0.5).float()

            valid_accuracy = labels.data.eq(outs.data.long()).sum()
        print('Valid loss: {}; Valid accuracy {}'.format(valid_loss, valid_accuracy))

        state = {'epoch': epoch,
                 'state_dict': net.state_dict(),
                 'optimizer': optimizer,
                 'loss': loss
                 }

        save_checkpoint(state, 'nanonets_model_{}.pth'.format(str(time.time())))








