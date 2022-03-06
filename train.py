from torch.autograd import Variable
from sqlalchemy import false
import argparse
import torchvision
import torch
import numpy as np
import sys
from dataloader import BatchSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from models import resnet50
from tensorboardX import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn

def train(opt):


    ########################################
    # Prepare Dataset
    ########################################

    chunk_size = opt.chunk_size
    ip_weight = opt.ip_weight

    transform_train = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Resize((256,128)),
            #transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    dataset = torchvision.datasets.CIFAR100('.\data',False, download=True, transform=transform_train)
    sampler = BatchSampler(dataset, 32, chunk_size)
    trainloader = DataLoader(dataset, batch_size=32, sampler=sampler, shuffle=False)

    ########################################
    # Initializing the model
    ########################################

    model = resnet50()

    device = torch.device('cuda:0')
    model = model.to(device)

    ########################################
    # Initializing metrics
    ########################################

    pdist = nn.PairwiseDistance()

    #writer = SummaryWriter()
    optimizer = optim.Adam([{'params': model.parameters()}], lr=opt.lr, weight_decay=opt.decay)


    for i in range(opt.epoch):
        trainloader = tqdm(trainloader)
        model.train()

        for idx, (img, label) in enumerate(trainloader):
            img = Variable(img).to(device)
            label = Variable(label).to(device)

            feat, class_feat = model(img)
            feat = feat.flatten(1)
            feat_positive = []
            feat_center = []
            feat_negative = []

            for i in range(0, feat.size(0), chunk_size):
                center = torch.sum(feat[i:i+chunk_size], dim=1) / chunk_size
                feat_center.append(center)
                for j in range(chunk_size):
                    feat_positive.append(torch.lerp(feat[i+j,:], center, ip_weight))
            








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='RegDB', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--decay',default=0.0005, type=float)
    parser.add_argument('--optim',default='Adam', type=str)
    parser.add_argument('--checkpoint',default='./checkpoint/')
    parser.add_argument('--epochs', default=70)
    parser.add_argument('--ip_weight', default=0.1)
    parser.add_argument('--log_path', default='./runs/')
    parser.add_argument('--trial',default=6,type=int)
    parser.add_argument('--chunk_size', default=4, type=int)
    parser.add_argument('--dim', default=768)
    parser.add_argument('--img_h', default=256, type=int)
    parser.add_argument('--img_w',default=128, type=int)
    parser.add_argument('--patch_size',default=16)
    parser.add_argument('--in_channel',default=3)
    parser.add_argument('--recon', default=True, type=bool)
    parser.add_argument('--batch_size',default=32, type=int)
    parser.add_argument('--margin',default=0.5)
    

    opt = parser.parse_args()



    train(opt)