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
			# Xavier initialization(works better in this case)
			sigma = np.sqrt(4./(self.layer[i-1]+self.layer[i]))
			W.append( np.array(np.random.normal(0.5,sigma,(self.layer[i-1],self.layer[i]))) )
			B.append( np.zeros((1,self.layer[i])) )

			# Create random uniform matrices with dimensions specified
			#W.append( np.random.uniform( -1,1, size=(self.layer[i-1],self.layer[i]) ) )
			#B.append( np.random.uniform( -1,1, size=(1,self.layer[i]) ) )

			# Deltas creation, they are used intraining process
			delta.append( np.zeros((1,self.layer[i])) )
			Delta.append( np.zeros((1,self.layer[i])) )

		self.W = W
		self.B = B

		self.delta = delta
		self.Delta = Delta

	def linear_J(self,Y_pred,Y):
		return np.sum((Y-Y_pred)*(Y-Y_pred)) / len(Y)

	def momentum_optimization(self, learn_rate):
		try:
			mW = self.momentumW
			mB = self.momentumB
		except AttributeError:
			mW = [[]]
			mB = [[]]
			for j in range(1,self.n_layers+1):
				mW.append( np.zeros_like(self.W[j]) )
				mB.append( np.zeros_like(self.B[j]) )

		beta = 0.9

		for j in range(self.n_layers):
			l = self.n_layers-j
			
			gradW = np.dot(self.X[l-1].T,self.Delta[l])
			gradB = np.sum(self.Delta[l], axis=0).reshape((1,-1))

			mW[l] = beta*mW[l] + learn_rate*gradW
			mB[l] = beta*mB[l] + learn_rate*gradB
			
			self.W[l] -= mW[l]
			self.B[l] -= mB[l]

		self.momentumW = mW
		self.momentumB = mB

	def gradient_descent(self, learn_rate):
		for j in range(self.n_layers):
				l = self.n_layers-j
				
				gradW = np.dot(self.X[l-1].T,self.Delta[l])
				gradB = np.sum(self.Delta[l], axis=0).reshape((1,-1))
				
				self.W[l] -= learn_rate*gradW
				self.B[l] -= learn_rate*gradB	

	def Work(self, X): # evaluates the entries through all network and return result
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
			X = np.vectorize(self.f[i-1])(Z )

			#saves values for back-propagation
			self.Z.append( Z )
			self.X.append( X )

		return X

	def Train(self, X, Y, learn_rate=0.01, epochs=10000):
		errors = np.array([])

		Y_pred = self.Work(X)
		J = self.linear_J(Y_pred,Y)
		print("Initial Error: \t",J*100) 
		errors = np.append(errors,J)

		epsilon = 1e-10
		step=epochs/100.
		umbral=10

		i=0
		while J > epsilon and i<epochs:
			self.delta[-1] = (self.X[-1]-Y)*2./ len(Y)

			for j in range(self.n_layers):
				l = self.n_layers-j
				self.Delta[l] = self.delta[l] * np.vectorize(self.df[l-1])(self.Z[l])
				self.delta[l-1] = np.dot(self.Delta[l],self.W[l].T)

			self.momentum_optimization(learn_rate)

			Y_pred = self.Work(X)
			J = self.linear_J(Y_pred,Y)

			if i==1000:
				learn_rate0 = learn_rate
			elif i>1000 and J<1e-4:
				learn_rate = learn_rate0*i/umbral
			

			if i%step==0:	
				Y_pred = self.Work(X)
				J = self.linear_J(Y_pred,Y)
				print("\nEpoch: ",i," learn_rate: ",learn_rate,"\tError:\t",J*100,"%")
			
			errors = np.append(errors,J)


			i+=1 

		plt.plot(np.linspace(0,len(errors),len(errors)),errors)
		plt.show()
		plt.close()

	
		return J

