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

neu_per_layer = np.array([2,2,1])
# Create network with ReLU activation function
P = perceptron(neu_per_layer, ReLU, dReLU)

# ask for Input data
# X = np.array(list(map(float, input("Ingresar datos de entrada:\t").split(" ")))).reshape([1,-1])
X = np.array([[0,0],[0,1],[1,0],[1,1]]) #XOR DATA
Y = np.array([[0],[1],[1],[0]])

# Shows the result to the user
print("Initial value:\t",P.Work(X[1]))

# Training
P.Train(X[1],Y[1])

print("Final value:\t",P.Work(X[1]))