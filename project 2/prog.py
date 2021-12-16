import torch
import math


class Linear(object):
    def __init__(self,in_features,out_features):
        self.weight=0
        self.bias=0
    def forward(self):
        return
    def backward(self):
        return
    def zero_grad(self):
        return

def tanh_prime(x):
    return 1-math.tanh(x)**2

class Tahn(object):
    def forward(self,X,reset_grad):
        return X.tanh()
    
    def backward(self,index,*gradwrtoutput):
        grad=[]
        for g in gradwrtoutput:
            gradwrtoutput*self.input[index].apply_(tanh_prime)
        return grad
    
    def zero_grad(self):
        return
    
    def param(self):
        return []

class ReLU(object):
    def forward(self,X):
        return

class MSE(object):
    def forward(self):
        return ((self.y-self.y_pred)**2).mean()
    
    def __call__(self,target,pred):
        self.y=target
        self.y_pred=pred
        return self.forward()
    
    def backward(self):
        grad=(2/self.y_pred.size(0))*(self.y_pred-self.y)
        return grad
    
    def param(self):
        return []

class Sequential(object):
    def __init__(self,*chain):
        self.chain=chain
    def forward(self,X):
        return
    
    def __call__(self,X):
        return self.forward(X)

class Module(object):
    def __init__(self,*hidden_layers,in_features=2,out_features=1,ReLU=False):
        if len(hidden_layers)==0:
            hidden_layers=[25,25,25]
        if ReLU==False:
            fa=Tahn()
        else:
            fa=ReLU()
        chain=[Linear(in_features,hidden_layers[0])]
        chain+=[fa]
        for i in len(hidden_layers)-1:
            chain+=[Linear(hidden_layers[i],hidden_layers[i+1])]
            chain+=[fa]
        chain+=[Linear(hidden_layers[-1],out_features)]
        self.seq=Sequential(*chain)
    
    def forward(self,X):
        return self.seq(X)
    
    def __call__(self,X):
        return self.forward(X)
    
    def backward(self,*gradwrtoutput):
        self.seq.backward(*gradwrtoutput)

    def zero_grad(self):
        self.seq.zero_grad()

    def param(self):
        return self.seq.param(self)

class OptimSGD(object):
    def __init__(self,method,learning_rate):
        self.method=method
        self.learning_rate
    def step(self):
        parameters=self.method.param()
        for p,g in parameters:
            p-=self.learning_rate*g
    def zero_grad(self):
        self.method.zero_grad()

def inDisk(data):
    N=data.size(0)
    center=torch.tensor([0.5,0.5])
    D=(data-center).norm(2,1)
    out=torch.full([N],0.)
    out[D<=1/math.sqrt(2*math.pi)]=1.
    return out

if __name__== '__main__':
    
    ## generate data
    
    N=1000
    
    data_train=torch.empty((N,2)).uniform_(0,1)
    data_test=torch.empty((N,2)).uniform_(0,1)
    
    train_target=inDisk(data_train)
    test_target=inDisk(data_test)
    
    # no need to normalize we're in [0,1]**2
    
    ## learning
    
    model=Module()
    criterion=MSE()
    optimizer=OptimSGD(model, learning_rate=0.001)
    
    batch_size,nb_epochs=100,25
    
    data_BTrain=data_train.split(batch_size)
    BTarget=train_target.split(batch_size)
    
    n_batch=len(BTarget)
    
    while nb_epochs>0:
        nb_epochs-=1
        for i in range(n_batch):
            optimizer.zero_grad()
            b_pred=model(data_BTrain[i])
            loss=criterion(BTarget[i], b_pred)
            criterion.backward()
            optimizer.step()
        print(loss.item())
        