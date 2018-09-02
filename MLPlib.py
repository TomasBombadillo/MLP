#####################################################
#----------- Multi-Layer Perceptron -----------------
#####################################################
#------------Author: Tomas Londono------------------- 
############# Last edit: 31/8/18 ####################

import numpy as np
import matplotlib.pyplot as plt


class perceptron(object):
	'''
	Create a simple perceptron with only one activation function, 
	it ask the user the number of layers, the number of entries and
	the number of neurons per layer.
	'''
	def __init__(self, layers, activ_function,devActiv_function):
		self.f = activ_function
		self.df = devActiv_function

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
			# Xavier initialization
			#------------------------
			# usually work, but sometimes stay in error=1 and doesn't move anywhere
			sigma = np.sqrt(2./(self.layer[i-1]+self.layer[i]))
			W.append( np.array(np.random.normal(0.5,sigma,(self.layer[i-1],self.layer[i]))) )

			# Create random matrices with dimensions specified
			#W.append( np.array(np.random.random( (self.layer[i-1],self.layer[i]) )) )
			B.append( np.zeros((1,self.layer[i])) )

			delta.append( np.zeros((1,self.layer[i])) )
			Delta.append( np.zeros((1,self.layer[i])) )

		self.W = W
		self.B = B

		self.delta = delta
		self.Delta = Delta

	def sigmoid(self,X):
		return 1/(1+np.exp(-X))

	def J(self,Y_pred,Y):
		#print("\n-----substraction-----\n",(Y_pred-Y))
		#print("--------prod--------\n",(Y_pred-Y)*(Y_pred-Y))
		#print("--------sum--------\n", np.sum((Y-Y_pred)*(Y-Y_pred)))
		return np.sum((Y-Y_pred)*(Y-Y_pred)) / len(Y)

	def binary_J(self,Y_pred,Y):
		return -1.*np.sum( Y*np.log(self.sigmoid(Y_pred)) + (1.-Y)*np.log(1.-self.sigmoid(Y_pred)) ) / len(Y)

	def Work(self, X, alpha=1): # evaluates the entries through all network and return result
		try:
			self.X
			self.Z
		except AttributeError:
			self.X = []
			self.Z = [[]]

		self.X.append( np.array(X) )
		for i in range(1,self.n_layers+1):
			# Evaluates the result of the past layer ands send it to the next one
			Z = np.array([(np.dot(X,self.W[i]) + self.B[i]).reshape(-1)]).reshape( (X.shape[0],self.B[i].shape[1]) )
			X = np.vectorize(self.f[i-1])( alpha*Z )

			#saves values for back-propagation
			self.Z.append( Z )
			self.X.append( X )

		return X

	def Train(self, X, Y, learn_rate=0.01, alpha=1, epochs=10000):
		errors = np.array([])

		Y_pred = self.Work(X,alpha)
		J = self.J(Y_pred,Y)
		print("Initial Error: \t",J*100) 
		errors = np.append(errors,J)

		epsilon = 1e-5
		i=0
		step=epochs/100.
		#while J > epsilon and i<epochs:
		while i<epochs:
			i+=1
			self.delta[-1] = (self.X[-1]-Y)/ len(Y)
			#self.delta[-1] = ( self.sigmoid(Y_pred)-Y )
			for j in range(self.n_layers):
				l = self.n_layers-j

				self.Delta[l] = self.delta[l] * np.vectorize(self.df[l-1])(alpha*self.Z[l])
				self.delta[l-1] = np.dot(self.Delta[l],self.W[l].T)

			for j in range(self.n_layers):
				l = self.n_layers-j

				gradW = np.dot(self.X[l-1].T,self.Delta[l])
				gradB = np.sum(self.Delta[l], axis=0).reshape((1,-1))

				self.W[l] -= learn_rate*gradW
				self.B[l] -= learn_rate*gradB

			if i%step==0:
				Y_pred = self.Work(X,alpha)
				J = self.J(Y_pred,Y)
				print("iter: ",i,"Value:\n",Y_pred,"Error:\t",J*100)
				errors = np.append(errors,J)

		Y_pred = self.Work(X,alpha)
		J = self.J(Y_pred,Y)
		print("Value:\n",Y_pred,"\nError:\t",J*100)

		plt.plot(np.linspace(0,len(errors)*step,len(errors)),errors, ".")
		plt.show()
		plt.close()
		return J

