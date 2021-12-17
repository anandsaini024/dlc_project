# Mini-project group T
## Project 1
The objective of this part is to test different architectures to compare two digits visible in a
two-channel image. 

It aims at showing:
1. The impact of weight sharing,

2. The use of an auxiliary loss to help the training.


It is implemented with PyTorch only code, without using other external libraries such as scikit-learn or numpy.

Implemented a deep network such that, given as input a series of 2*14*14 tensor, corresponding to pairs of 14 * 14 grayscale images, it predicts for each pair if the first digit is lesser or equal to the second.

## Project 2
The goal of this project is to design a small deep learning framework without using "torch.nn", autograd and over advance lybrary.  
We want to create a module that given a points in [0,1]^2 should determine if the points is in the disk of center (0.5,0.5) and radius 1/sqrt(2*pi).  
The network should have 3 hidden layer of 25 units and can work with 2 possible activation function (ReLU and Tanh). To train the network, we're using stochastic gradient descent.
