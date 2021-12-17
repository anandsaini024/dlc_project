import torch
import math


class Linear(object):
    """
    Linear module with input/output of dim in_features/out_features
    """
    def __init__(self,in_features,out_features):
        """
        initialisation of the weight following a uniform density function
        in the range [-1/math.sqrt(in_features),1/math.sqrt(in_features)].
        """
        self.outIsOne=(out_features==1)
        self.weight=torch.empty((out_features,in_features)).uniform_\
            (-1/math.sqrt(in_features),1/math.sqrt(in_features))
        self.bias=torch.full([out_features],0.)
        
        self.weight_grad=torch.full((out_features,in_features),0.)
        self.bias_grad=torch.full([out_features],0.)
        
    def forward(self,X):
        return X@self.weight.T+self.bias
    def __call__(self,X):
        # store input for backwardation
        self.X=X
        if self.outIsOne:
            return self.forward(X).view(-1)
        return self.forward(X)
    def backward(self,*gradwrtoutput):
        """
        compute the gradients of the input and parameters from the gradient 
        of the output.
        It does this for each images so it takes as input the gradients for
        each images.
        """
        index=0
        gradwrtintput=[]
        for g in gradwrtoutput:
            self.bias_grad+=g
            self.weight_grad+=torch.outer(g,self.X[index])
            # or use torch.einsum('i,j->ij',g,x)
            gradwrtintput+=[self.weight.T@g]
            index+=1
        return gradwrtintput
    def zero_grad(self):
        """
        reset the gradients to 0.
        """
        self.weight_grad[:]=0
        self.bias_grad[:]=0
    def param(self):
        """
        return a tuple of parameters and corresponding gradients
        """
        return [(self.weight,self.weight_grad),(self.bias,self.bias_grad)]

def tanh_prime(x):
    """
    derivative of tanh(x)
    """
    return 1-x.tanh()**2

class Tahn(object):
    """
    Tahn activation function
    """
    def forward(self,X):
        return X.tanh()
    
    def __call__(self,X):
        self.input=X
        return self.forward(X)
    
    def backward(self,*gradwrtoutput):
        grad=[]
        index=0
        for g in gradwrtoutput:
            grad+=[g*tanh_prime(self.input[index])]
            index+=1
        return grad
    
    def zero_grad(self):
        return
    
    def param(self):
        return []

class ReLU(object):
    """
    ReLU activation funciton
    """
    def forward(self,X):
        out=torch.empty(X.size()).copy_(X)
        out[X<0]=0.
        return out
    
    def __call__(self,X):
        self.X=X
        return self.forward(X)
    
    def backward(self,*gradwrtoutput):
        index=0
        gradwrtintput=[]
        temp=torch.full(self.X.size(),0)
        temp[self.X>0]=1.
        for g in gradwrtoutput:
            gradwrtintput+=[g*temp[index]]
            index+=1
        return gradwrtintput
    
    def zero_grad(self):
        return
    
    def param(self):
        return []

class MSE(object):
    """
    compute MSE loss function and get gradient
    """
    def forward(self):
        return ((self.y-self.y_pred)**2).mean()
    
    def __call__(self,target,pred):
        self.y=target
        self.y_pred=pred
        return self.forward()
    
    def backward(self):
        grad=(2/self.y_pred.size(0))*(self.y_pred-self.y)
        out=[grad[i].view(-1) for i in range(grad.size(0))]
        return out
    
    def param(self):
        return []

class Sequential(object):
    """
    Create a NN that just chain the module given in initialisation
    """
    def __init__(self,*chain):
        self.chain=chain
        self.size=len(chain)
    
    def forward(self,X):
        out=X
        for i in range(self.size):
            out=self.chain[i](out)
        return out
    
    def __call__(self,X):
        return self.forward(X)
    
    def backward(self,*gradwrtoutput):
        gradwrtintput=gradwrtoutput
        for i in range(self.size):
            j=self.size-1-i
            gradwrtintput=self.chain[j].backward(*gradwrtintput)
        return gradwrtintput
    
    def zero_grad(self):
        for i in range(self.size):
            self.chain[i].zero_grad()
    
    def param(self):
        out=[]
        for i in range(self.size):
            out+=self.chain[i].param()
        return out

class Module(object):
    """
    create a NN module, where hidden_layers describe the size of its inner
    layer and in_features/out_features the size of the input/output.
    """
    def __init__(self,*hidden_layers,in_features=2,out_features=1,ReLU=False):
        if len(hidden_layers)==0:
            hidden_layers=[25,25,25]
        if ReLU==False:
            fa=Tahn()
        else:
            fa=ReLU()
        chain=[Linear(in_features,hidden_layers[0])]
        chain+=[fa]
        for i in range(len(hidden_layers)-1):
            chain+=[Linear(hidden_layers[i],hidden_layers[i+1])]
            chain+=[fa]
        chain+=[Linear(hidden_layers[-1],out_features)]
        self.seq=Sequential(*chain)
    
    def forward(self,X):
        return self.seq(X)
    
    def __call__(self,X):
        return self.forward(X)
    
    def backward(self,*gradwrtoutput):
        return self.seq.backward(*gradwrtoutput)

    def zero_grad(self):
        self.seq.zero_grad()

    def param(self):
        return self.seq.param()

class OptimSGD(object):
    """
    Optimization of a module using SGD
    """
    def __init__(self,method,learning_rate):
        self.method=method
        self.learning_rate=learning_rate
    def step(self):
        parameters=self.method.param()
        for p,g in parameters:
            p-=self.learning_rate*g
    def zero_grad(self):
        self.method.zero_grad()

def inDisk(data):
    """
    get data in [0,1]**2 and return it's classification 
    (1 if in, 0 if out of the disk)
    """
    N=data.size(0)
    center=torch.tensor([0.5,0.5])
    D=(data-center).norm(2,1)
    out=torch.full([N],0.)
    out[D<=1/math.sqrt(2*math.pi)]=1.
    return out

def prediction(data):
    """
    get the output of a NN and return the correct class
    """
    out=torch.full([data.size(0)],0)
    out[data>=0.5]=1
    return out

if __name__== '__main__':
    
    ## generate data
    
    N=1000
    
    data_train=torch.empty((N,2)).uniform_(0,1)
    data_test=torch.empty((N,2)).uniform_(0,1)
    
    train_target=inDisk(data_train)
    test_target=inDisk(data_test)
    
    s=data_train.var(0)
    
    data_train/=s
    data_test/=s
    
    
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
            
            grad=criterion.backward()
            model.backward(*grad)
            optimizer.step()
        print(loss.item())
    
    test_pred=model(data_test)
    
    test_pred_class=prediction(test_pred)
    
    error=(test_target!=test_pred_class).sum()/N
    
    print('error = {}'.format(error.item()))