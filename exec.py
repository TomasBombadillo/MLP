from MLPlib import *
import numpy as np

def ReLU(x):
	if x <0 :
		return 0
	elif x >=0:
		return x
def dReLU(x):
	if x<0:
		return 0
	elif x>=0:
		return 1

def sigmoid(x):
	return 1/(1+np.exp(-x))
def dsigmoid(x):
	return sigmoid(x)*(1-sigmoid(x))

def swish(x):
	return x*sigmoid(x)
def dswish(x):
	return sigmoid(x)*(1+sigmoid(x)*x*np.exp(-x))

def identity(x):
	return x 
def didentity(x):
	return 1

# Array of numberof neurons per layer
neu_per_layer = np.array([2,2,1])

# Array of functions
f = [ReLU, identity]
df = [dReLU,  didentity]

# Create network with ReLU activation function
#P = perceptron(neu_per_layer, ReLU, dReLU)
P = perceptron(neu_per_layer, f, df)

# Input data
X = np.array([[0,0],[0,1],[1,0],[1,1]]) #XOR DATA

# Correct output values
Y = np.array([[0],[1],[1],[0]])

#trap init
P.W[1] = np.array([[1.,1.],[1.,1.]]) 
P.W[2] = np.array([[1.],[-2.]])
P.B[1] = np.array([[0.,-1.]])

# Shows the result to the user
print("Initial value:\n",P.Work(X))

# Training
P.Train(X,Y, alpha=1)


print("Final value:\n",P.Work(X))