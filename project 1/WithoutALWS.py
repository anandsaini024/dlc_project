#!/usr/bin/env python
# coding: utf-8

# In[4]:


import dlc_practical_prologue as prologue
import torch
import torch.nn as nn
import torch.optim as optim


# In[9]:


class Rnumber(nn.module):
    def __init__(self):
        super().__init__()
        
        self.bloc = nn.Sequential(
    
        nn.Conv2d(1,6,kernel_size=(3,3),stride=(1,1)),
        nn.ReLU(True),
            
        
        nn.Conv2d(6,16,kernel_size=(3,4), stride=(1,1)),
        
        nn.ReLU(True),
        nn.conv2d(6,16, kernel_size=(3, 3), stride=(1, 1))
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=(2,2),stride=(2, 2), dilation=(1, 1))
        
            
        self.classification = nn.Sequential(
         nn.Linear(256,120),
            nn.ReLU(True),
            #42
            nn.Linear(120,84),
            nn.ReLU(True),
            #24
            nn.Linear(84,10))
            
#         def forward(self,x): 
#             out=self.encoder(x)
#             out=self.decoder(out)
#             out=self.bloc(out)
#             out=out.view(x.size(0),-1)
#             out=self.classification(out)
#             return out
            
if __name__ == '__main__':
    n=500
    train_input, train_target, train_classes, test_input, test_target,\
        test_classes=prologue.generate_pair_sets(n)
    
    number_train=train_input.view(-1,1,14,14)
    n_train=train_classes.view(-1)
    y_train=classification(n_train).type(torch.FloatTensor)
    
    number_test=test_input.view(-1,1,14,14)
    n_test=test_classes.view(-1)
    y_test=classification(n_test).type(torch.FloatTensor)
    
    model=Rnumber()

