#!/usr/bin/env python
# coding: utf-8


import os
u=os.path.dirname(os.getcwd())+'\\utils'
import sys
sys.path.insert(1, u)
import dlc_practical_prologue as prologue
import torch
import torch.nn as nn
import torch.optim as optim




class initial_model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.bloc1 = nn.Sequential(
            #n*1*14*14
            nn.Conv2d(1,6,kernel_size=(3,3)),
            #n*6*12*12    
            nn.ReLU(True),
            #n*6*12*12    
            nn.MaxPool2d(kernel_size=(2,2)),  
            #n*6*6*6
            nn.Conv2d(6,16,kernel_size=(3,3)),
            #n*16*4*4
            )
        
        self.bloc2 = nn.Sequential(
            #n*1*14*14
            nn.Conv2d(1,6,kernel_size=(3,3)),
            #n*6*12*12    
            nn.ReLU(True),
            #n*6*12*12    
            nn.MaxPool2d(kernel_size=(2,2)),  
            #n*6*6*6
            nn.Conv2d(6,16,kernel_size=(3,3)),
            #n*16*4*4
            )
        
            
        self.classification1 = nn.Sequential(
            nn.Linear(256,120),
            nn.ReLU(True),
            
            nn.Linear(120,84),
            nn.ReLU(True),
            
            nn.Linear(84,10))
            
            
        self.classification2 = nn.Sequential(
            nn.Linear(256,120),
            nn.ReLU(True),
            
            nn.Linear(120,84),
            nn.ReLU(True),
            
            nn.Linear(84,10)) 
        
        self.compare=nn.Sequential(
            #N*20
            nn.Linear(20,12),
            nn.ReLU(True),
            #N*8
            nn.Linear(12,8),
            nn.ReLU(True),
            nn.Linear(8,2)
            #N*2

            )
        
    def forward(self,x):
        img1,img2 = torch.split(x,1,1)
            
        out1=self.bloc1(img1).view(-1,256)
        out2=self.bloc2(img2).view(-1,256)
            
        out1=self.classification1(out1)
        out2=self.classification2(out2)
            
        out=torch.cat((out1,out2),1).view(x.size(0),-1)
        out = self.compare(out)
        return out
 

def classification(x):
    """
    transform a class index into a vector of classes.
    Ex : 3 becomes [0,0,0,1,0,0,0,0,0,0]
    """
    out=torch.full([x.size(0),2],0).float()
    for i in range(x.size(0)):
        out[i,x[i].item()]=1
    return out




def lesser(x):
    """
    from a two dim vector pick the index of the element with biggest value.
    Either return 1 or 0
    """
    out=torch.full([x.size(0)],0)
    for i in range(x.size(0)):
        out[i]=x[i].argmax()
    return out





if __name__ == '__main__':
    n=1000
    train_input, train_target, train_classes, test_input, test_target,\
        test_classes=prologue.generate_pair_sets(n)
    train_target=classification(train_target).type(torch.FloatTensor)
    
    train_input = train_input/255.0
    test_input = test_input/255.0

    model = initial_model()
    
    #begin training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
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

