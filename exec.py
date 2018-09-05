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
    "Numerically stable sigmoid function."
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        return z / (1 + z)
def dsigmoid(x):
	return sigmoid(x)*(1-sigmoid(x))

def swish(x):
	return x*sigmoid(x)
def dswish(x):
	return sigmoid(x)+x*dsigmoid(x)

def SELU(x):
	lamb=1.
	alpha=10.
	if x <=0 :
		return lamb*alpha*(np.exp(x)-1.)
	elif x >0:
		return lamb*x

def dSELU(x):
	lamb=1.
	alpha=10.
	if x <=0 :
		return lamb*alpha*(np.exp(x))
	elif x >0:
		return lamb

def identity(x):
	return x 
def didentity(x):
	return 1

# Array of numberof neurons per layer
neu_per_layer = np.array([2,2,1])

# Array of functions
f = [swish, sigmoid]
df = [dswish, dsigmoid]

# Create network with ReLU activation function
P = perceptron(neu_per_layer, f, df)

# Input data
X = np.array([[0,0],[0,1],[1,0],[1,1]]) #XOR DATA

# Correct output values
Y = np.array([[0],[1],[1],[0]])

# Shows the result to the user
print("Initial value:\n",P.Work(X))

# Training
P.Train(X,Y,learn_rate = 2.,epochs=100000)

print("Final value:\n",P.Work(X))