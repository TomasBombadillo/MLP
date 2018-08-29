from MLPlib import *
import numpy as np


neu_per_layer = np.array([2,2,1])
# Create network with ReLU activation function
P = perceptron(neu_per_layer, ReLU)

# ask for Input data
# X = np.array(list(map(float, input("Ingresar datos de entrada:\t").split(" ")))).reshape([1,-1])
X = np.array([[1,1],[-1,1],[1,-1],[-1,-1]]) #XOR DATA
Y = np.array([[0],[1],[1],[0]])

# Shows the result to the user
print(P.Work(X[0]))

# Prints
'''
print(P.n_layers)
for Xl in P.X:
	print(Xl)
print(P.X[1])
'''

# Training
print(P.Train(X[0],Y[0]))