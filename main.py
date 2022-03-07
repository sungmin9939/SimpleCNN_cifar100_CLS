import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torchvision
import torch
import numpy as np
import sys
from dataloader import BatchSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from models import resnet50
transform_train = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Resize((256,128)),
        #transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])



dataset = torchvision.datasets.CIFAR100('.\data',True, download=True, transform=transform_train)
sampler = BatchSampler(dataset, 32, 4)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, shuffle=False)

model = resnet50()

for idx, (img, label) in enumerate(dataloader):
    print(img.shape)
    print(label)
    output, class_emb = model(img)
    print(output.shape)
    print(class_emb.shape)
    sys.exit(0)






        




    
