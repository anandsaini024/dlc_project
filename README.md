# Mini-project group T
## Project 1
## Project 2
The goal of this project is to design a small deep learning framework without using "torch.nn", autograd and over advance lybrary.  
We want to create a module that given a points in [0,1]^2 should determine if the points is in the disk of center (0.5,0.5) and radius 1/sqrt(2*pi).  
The network should have 3 hidden layer of 25 units and can work with 2 possible activation function (ReLU and Tanh). To train the network, we're using stochastic gradient descent.
