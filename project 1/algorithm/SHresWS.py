# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 22:59:24 2021

@author: Julien
"""
import os
u=os.path.dirname(os.getcwd())+'\\utils'
import sys
sys.path.insert(1, u)
import dlc_practical_prologue as prologue
import torch
import torch.nn as nn
import torch.optim as optim
from number2 import Rnumber

class LesserWS(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.num=Rnumber()
        
        self.lesser=nn.Sequential(
            nn.Linear(20,12),
            nn.ReLU(True),
            nn.Linear(12,8),
            nn.ReLU(True),
            nn.Linear(8,2)
            )
        
    def forward(self,x): #Nx2x14x14
        im1, im2 = torch.split(x,1,1)
        c1=self.num(im1)
        c2=self.num(im2)
        out=torch.cat((c1,c2),1).view((x.size(0),-1))
        out=self.lesser(out)
        return out

def classification(x):
    out=torch.full([x.size(0),2],0)
    for i in range(x.size(0)):
        out[i,x[i].item()]=1
    return out

def lesser(x):
    out=torch.full([x.size(0)],0)
    for i in range(x.size(0)):
        out[i]=x[i].argmax()
    return out

if __name__ == '__main__':
    n=1000
    train_input, train_target, train_classes, test_input, test_target,\
        test_classes=prologue.generate_pair_sets(n)
    
    train_target=classification(train_target).type(torch.FloatTensor)
    
    train_input/=255.0
    test_input/=255.0
    
    
    model=LesserWS()
    
    #training
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters())

    batch_size,nb_epochs=100,25
    
    batch_input=train_input.split(batch_size)
    batch_y=train_target.split(batch_size)
    
    n_batch=len(batch_input)
    
    while nb_epochs>0:
        nb_epochs-=1
        for i in range(n_batch):
            optimizer.zero_grad()
            y_pred=model(batch_input[i])
            loss=criterion(y_pred,batch_y[i])
            loss.backward()
            optimizer.step()
        print(loss.item())
    
    y_pred=model(test_input)
    
    lesser_pred=lesser(y_pred)
    
    error=(test_target!=lesser_pred).sum()/n
    
    print('error = {}'.format(error.item()))
    