#####################################################
#----------- Multi-Layer Perceptron -----------------
#####################################################
#------------Author: Tomas Londono------------------- 
############# Last edit: 22/8/18 ####################

import numpy as np

def ReLU(x):
	if x <0 :
		return 0
	elif x >=0:
		return x

def dReLU(x):
	if x<0:
		return 0
	elif x<=0:
		return 1

class perceptron(object):
	'''
	Create a simple perceptron with only one activation function, 
	it ask the user the number of layers, the number of entries and
	the number of neurons per layer.
	'''
	def __init__(self, layers, activ_function):
		self.f = activ_function

		# Number of layers
		self.n_layers = len(layers)-1

		# Create array with number of neurons per layer
		self.layer = layers

		# Array of matrices, each matrix correspond to the conections between last layer
		# and the next one, same thing with the column matrices B
		W = [np.array([])]
		B = [np.array([])]

		delta = [np.array([])]
		Delta = [np.array([])]

		for i in range(1, self.n_layers+1):
			# Create random matrices with dimensions specified
			W.append( np.array([np.random.random( (self.layer[i-1],self.layer[i]) )]) )
			B.append( np.array([np.random.random( self.layer[i] )]) )

			delta.append( np.zeros((1,self.layer[i])) )
			Delta.append( np.zeros((1,self.layer[i])) )

		self.W = W
		self.B = B

		self.delta = delta
		self.Delta = Delta

	def Work(self, X): # evaluates the entries through all network and return result
		try:
			self.X
			self.Z
		except AttributeError:
			self.X = []
			self.Z = [[]]

		self.X.append( X )
		for i in range(1,self.n_layers+1):
			# Evaluates the result of the past layer ands send it to the next one
			Z = np.array([(np.dot(X,self.W[i]) + self.B[i]).reshape(-1)])
			X = np.vectorize(self.f)( Z )

			#saves values for back-propagation
			self.Z.append( Z )
			self.X.append( X )
		return X

	def Train(self, X, Y):
		J = np.sum(self.Work(X)-Y)**2 /len(Y)
		'''
		try:
			self.delta
			self.Delta
		except AttributeError:
			for i in range(1, self.n_layers+1):
				self.delta.append( np.zeros((self.layer[i-1, self.layer[i]])) )
				self.Delta.append( np.zeros((self.layer[i-1, self.layer[i]])) )
		'''
		epsilon = 1e-5
		i=0
		while J > epsilon or i>100:
			i+=1
			for j in range(self.n_layers+1):
				l = self.n_layers-j

				if l==self.n_layers: delta = (self.X[l]-Y)/ len(X)
				Delta = delta * self.X[l].T

				self.delta.append(delta)
				self.Delta.append(Delta)

				# PRINT
				#print(l,self.X[l], self.delta[l], self.Delta[l])
				#print(l, delta,delta.shape,Delta,Delta.shape)
			if i==1: break



			

		pass

	def BackPropagation(self, df):
		pass



