from turtle import distance
from matplotlib.pyplot import axis
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
from loss import ContrastLoss
from tensorboardX import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn

from utils import AverageMeter

def train(opt):
    ########################################
    # Loss Recoder
    ########################################
    loss_cr = AverageMeter()
    loss_ce = AverageMeter()
    loss_train = AverageMeter()

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
    dataset = torchvision.datasets.CIFAR100('.\data',True, download=True, transform=transform_train)
    sampler = BatchSampler(dataset, opt.batch_size, chunk_size)
    trainloader = DataLoader(dataset, batch_size=opt.batch_size, sampler=sampler, shuffle=False)
   
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
    contras_cri = ContrastLoss()
    ce_cri = nn.CrossEntropyLoss()
    #writer = SummaryWriter()
    optimizer = optim.Adam([{'params': model.parameters()}], lr=opt.lr, weight_decay=opt.decay)
    writer = SummaryWriter('./runs')

    for e in range(opt.epochs):
        trainloader = tqdm(trainloader)
        model.train()

        for idx, (img, label) in enumerate(trainloader):
            img = Variable(img).to(device)
            label = Variable(label).to(device)

            feat, class_feat = model(img)
            feat = feat.flatten(1)
            feat_positive = None
            feat_center = None
            feat_negative = None

            for i in range(0, feat.size(0), chunk_size):
                center = (torch.sum(feat[i:i+chunk_size,:], dim=0) / chunk_size).unsqueeze(0)
                feat_center = center if feat_center is None else torch.cat((feat_center, center), dim=0)
                for j in range(chunk_size):
                    inter_pts = torch.lerp(feat[i+j,:], center, ip_weight)
                    feat_positive = inter_pts if feat_positive is None else torch.cat((feat_positive, inter_pts), dim=0)
                    
                    
            distance = torch.matmul(feat, torch.transpose(feat, 0, 1)).cpu().detach().numpy()
            dist_indices = np.argsort(distance, axis=1)
            for i in range(dist_indices.shape[0]):
                keep = True
                start = i - int(i % 4)
                end = i + (3 - int(i % 4))
                pt = -1 
                while keep:
                    index = dist_indices[i][pt]
                    if index >= start and index <= end:
                        pt -= 1
                    else:
                        neg = (feat[index, :]).unsqueeze(0)
                        
                        feat_negative = neg if feat_negative is None else torch.cat((feat_negative,neg),dim=0)
                        keep = False
            
            
            cont_loss = contras_cri(feat, feat_positive, feat_negative)
            ce_loss = ce_cri(class_feat, label)            
            total_loss = ce_loss #+ cont_loss
            
            loss_ce.update(ce_loss.item(), feat.size(0))
            loss_cr.update(cont_loss.item(), feat.size(0))
            loss_train.update(total_loss.item(), feat.size(0))
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
        writer.add_scalar('train_loss', loss_train.avg, e)
        writer.add_scalar('contrastive_loss', loss_cr.avg, e)
        writer.add_scalar('CE_loss', loss_ce.avg, e)
        
        print(
            'epoch: {}\ntrain_loss: {}\ncr_loss: {}\nid_loss: {}'.format(e, \
                loss_train.avg, loss_cr.avg, loss_ce.avg)
        )

            
                
                
                
            








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--decay',default=0.0005, type=float)
    parser.add_argument('--optim',default='Adam', type=str)
    parser.add_argument('--checkpoint',default='./checkpoint/')
    parser.add_argument('--epochs', default=100)
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
    parser.add_argument('--batch_size',default=128, type=int)
    parser.add_argument('--margin',default=0.5)
    

    opt = parser.parse_args()



    train(opt)