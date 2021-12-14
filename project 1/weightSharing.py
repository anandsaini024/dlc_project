# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 22:59:24 2021

@author: Julien
"""

import dlc_practical_prologue as prologue
import torch
import torch.nn as nn
import torch.optim as optim

class feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            #14x14
            nn.Conv2d(1, 32, kernel_size=(3,3)),
            nn.ReLU(True),
            nn.Conv2d(32,16,kernel_size=(4,4)),
            nn.ReLU(True),
            nn.Conv2d(16,16,kernel_size=(4,4))
            )
        
        self.decoder=nn.Sequential(
            #6x6
            nn.ConvTranspose2d(16, 16, kernel_size=(4,4),stride=(2,2)),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=(2,2),stride=(2,2))
            #28x28
            )
        #feature extraction
        self.bloc=nn.Sequential(
            #Nx1x28x28
            nn.Conv2d(1,6,kernel_size=(5,5)),
            #Nx6x24x24
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2,2)),
            #Nx6x12x12
            nn.Conv2d(6,16,kernel_size=(5,5)),
            #Nx12x8x8
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2,2),)
            #Nx12x4x4
            )
    def forward(self,x):
        out=self.encoder(x)
        out=self.decoder(out)
        out=self.bloc(out)
        return out

class LesserWS(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.num=feature()
        
        self.lesser=nn.Sequential(
            nn.Linear(512,120),
            nn.ReLU(True),
            nn.Linear(120,32),
            nn.ReLU(True),
            nn.Linear(32,20),
            nn.ReLU(True),
            nn.Linear(20,1)
            )
        
    def forward(self,x): #Nx2x14x14
        im1, im2 = torch.split(x,1,1)
        c1=self.num(im1)
        c2=self.num(im2)
        out=torch.cat((c1,c2),1).view((x.size(0),-1))
        out=self.lesser(out)
        return out.view(-1)

def lesser(x):
    if x>0.5:
        return 1
    return 0

if __name__ == '__main__':
    n=1000
    train_input, train_target, train_classes, test_input, test_target,\
        test_classes=prologue.generate_pair_sets(n)
    
    train_target=train_target.type(torch.FloatTensor)
    test_target=test_target.type(torch.FloatTensor)
    
    model=LesserWS()
    
    #training
    criterion=nn.MSELoss()
    optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

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
    
    b_pred=y_pred.detach().apply_(lesser)
    
    error=(test_target!=b_pred).sum()/n
    
    print('error = {}'.format(error.item()))
    