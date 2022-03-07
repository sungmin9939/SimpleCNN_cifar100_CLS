from turtle import forward
from sklearn.metrics import log_loss
import torch
import torch.nn as nn


class ContrastLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.l1 = nn.L1Loss()
        #self.weight = [1.0/16, 1.0/8, 1.0/4, 1.0]
        self.weight = 1.0/8
        
    def forward(self, a, p, n):
        loss = 0
    
        for i in range(len(a)):
            d_ap = self.l1(a[i], p[i])
            d_an = self.l1(a[i], n[i])
            cont = d_ap / (d_an + 1e-7)
            
            loss += self.weight * cont
        
        return loss