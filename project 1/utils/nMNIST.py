
import dlc_practical_prologue as prologue
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets

class Rnumber(nn.Module):
    def __init__(self):
        super().__init__()
        
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
        out=self.bloc(x)
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
    mnist_train_set = datasets.MNIST(root='./data2', train = True, download = True)
    mnist_test_set = datasets.MNIST(root='./data2', train = False, download = True)

    train_input = mnist_train_set.data.view(-1, 1, 28, 28).float()/255.0
    train_target = mnist_train_set.targets
    test_input = mnist_test_set.data.view(-1, 1, 28, 28).float()/255.0
    test_target = mnist_test_set.targets
    
    y_train=classification(train_target).type(torch.FloatTensor)
    y_test=classification(test_target).type(torch.FloatTensor)
    
    model=Rnumber()
    
    #training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9)

    
    batch_size,nb_epochs=100,1
    
    batch_number=train_input.split(batch_size)
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
        
    y_pred=model(test_input)
    
    loss=criterion(y_pred,y_test)
    
    n_pred=number(y_pred)
    
    error=(n_pred!=test_target).sum()/(test_target.size(0))
    print("error = {}".format(error.item()))