import torch
import math


class linear(object):
    def forward(self):
        return
    def backward(self):
        return


def tanh_prime(x):
    return 1-math.tanh(x)**2

class tahn(object):
    def forward(self,X,reset_grad):
        return X.tanh()
    
    def resetGrad(self):
        return
    
    def backward(self,index,*gradwrtoutput):
        grad=[]
        for g in gradwrtoutput:
            gradwrtoutput*self.input[index].apply_(tanh_prime)
        return grad

    def param(self):
        return []

class MSE(object):
    def forward(self,*input,y,y_pred):
        self.y=y
        self.y_pred=y_pred
        return ((y-y_pred)**2).mean()
    
    def backward(self):
        grad=(2/self.y_pred.size(0))*(self.y_pred-self.y)
        return grad
    
    def param(self):
        return []

class Module(object):
    def forward(self,*input):
        return
    
    def backward(self,*gradwrtoutput):
        self.seq.backward(*gradwrtoutput)

    def param(self):
        return self.seq.param(self)

def optim(self):
    def __init__(self,method,learning_rate):
        self.method=method
        self.learning_rate
    def step(self):
        parameters=self.method.param()
        for p,g in parameters:
            p-=self.learning_rate*g
        self.method.resetGrad()