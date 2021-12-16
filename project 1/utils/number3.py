# -*- coding: utf-8 -*-
"""

@author: Julien
"""

import dlc_practical_prologue as prologue
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Rnumber(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.Hres=nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=(2,2),stride=(2,2))
            )
        #feature extraction
        self.bloc=nn.Sequential(
            #Nx1x14x14
            nn.Conv2d(1,6,kernel_size=(5,5)),
            #Nx6x12x12
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2,2)),
            #Nx6x6x6
            nn.Conv2d(6,16,kernel_size=(5,5)),
            #Nx16x4x4
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2,2))
            #Nx32x2x2
            )
        
        #number classification
        self.classification=nn.Sequential(
            #64
            nn.Linear(256,120),
            nn.ReLU(True),
            #42
            nn.Linear(120,84),
            nn.ReLU(True),
            #24
            nn.Linear(84,10)
            #10
            )
    
    def forward(self,x): #Nx1x14x14
        # version 2
        out=self.Hres(x)
        
        out=self.bloc(out)
        out=out.view(x.size(0),-1)
        out=self.classification(out)
        return out

def classification(x):
    out=torch.full([x.size(0),10],0)
    for i in range(x.size(0)):
        out[i,x[i].item()]=1
    return out

def number(x):
    out=torch.full([x.size(0)],0)
    for i in range(x.size(0)):
        out[i]=x[i].argmax()
    return out

if __name__ == '__main__':
    n=500
    train_input, train_target, train_classes, test_input, test_target,\
        test_classes=prologue.generate_pair_sets(n)
    
    number_train=train_input.view(-1,1,14,14)/255.
    n_train=train_classes.view(-1)
    y_train=classification(n_train).type(torch.FloatTensor)
    
    number_test=test_input.view(-1,1,14,14)
    n_test=test_classes.view(-1)
    y_test=classification(n_test).type(torch.FloatTensor)
    
    model=Rnumber()
    
    #training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=0.1,momentum=0.9)

    
    batch_size,nb_epochs=100,25
    
    batch_number=number_train.split(batch_size)
    batch_y=y_train.split(batch_size)
    
    n_batch=len(batch_number)
    
    while nb_epochs>0:
        nb_epochs-=1
        for i in range(n_batch):
            optimizer.zero_grad()
            y_pred=model(batch_number[i])
            loss=criterion(y_pred,batch_y[i])
            print(loss.item())
            loss.backward()
            optimizer.step()
        
    y_pred=model(number_test/255)
    
    loss=criterion(y_pred,y_test)
    
    n_pred=number(y_pred)
    
    error=(n_pred!=n_test).sum()/(2*n)
    print("error = {}".format(error.item()))