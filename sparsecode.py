
from mlpython.learners.generic import Learner
import numpy as np
import datetime

class SparseCode(Learner):
	"""
	Sparse Code to learn feature representations.
 
	Option ``lr`` is the learning rate.
 
	Option ``hidden_size`` is the size of the sparse representation
 
	Option ``L1`` is the L1 regularization weight (weight decay).
 
	Option ``seed`` is the seed of the random number generator.
 	
	Option ``n_epochs`` number of training epochs.
 
	**Required metadata:**
 
	* ``'input_size'``: Size of the input.
	* ``'targets'``: Set of possible targets.
 
	"""
	
	def __init__(self,
				 lr=0.1,
				 size=20,
				 L1=0.1,
				 n_epochs=10,
				 seed=1234):
		self.lr=lr
		self.hidden_size=size
		self.L1=L1
		self.n_epochs=n_epochs
		self.seed=seed

		# internal variable keeping track of the number of training iterations since initialization
		self.epoch = 0 

	def initialize(self,input_size):
		"""
		This method initializes the dictionary.
		"""
		self.dict_size = input_size		
		
		self.rng = np.random.mtrand.RandomState(self.seed)   # create random number generator
		self.dictionary = self.rng.rand(self.dict_size, self.hidden_size)
		self.dictionary /= self.dictionary.sum(axis=0)		
		
		max_eig = np.linalg.eig(np.dot(self.dictionary.T, self.dictionary))[0][0]
		if (self.lr > 1/max_eig):
			print "WARNING: learning rate must be smaller than " + str(1/max_eig) + " to converge"

	def initialize_dictionary(self,dictionary):
		"""
		This method initializes the dictionary with a prebuilt dictionary.
		"""
		self.dictionary = dictionary
	
	def train(self,trainset):
		"""
		Trains the sparse code dictionary until it reaches a total number of
		training epochs of ``self.n_epochs`` since it was
		initialized.
		"""

		if self.epoch == 0:
			input_size = trainset.metadata['input_size']
			self.initialize(input_size)
		# NOTE: It's much easier to debug with a subset of the data, because training is very slow. Remove [:1000] to train on ALL data.
		# Otherwise, replace 1000 with some other number of training inputs.
		inputs = [input for input, target in trainset]
		for it in range(self.n_epochs):
			print "Epoch # " + str(it)
			self.dict_update(inputs)
		self.epoch = self.n_epochs
		
	def infer(self,input):
		"""
		Inference using the ISTA algorithm: 
		- learns a sparse representation of some dictionary
		- returns the sparse representation
		Argument ``input`` is a Numpy 1D array.
		"""
		# Note: this parameter is critical to learning a good sparse representation
		# If the threshold is too small, then the representation won't be sparse (most features won't be set to 0).
		# If the threshold is too large, then the representation won't converge at all (all features will be set to 0).
		convergence_threshold = 0.001
		
		h = np.zeros(self.hidden_size)
		hidden_zeros = np.zeros(self.hidden_size)
		
		converged = False
		while (not converged):
			old_h = np.copy(h)
			h -= self.lr * np.dot(self.dictionary.T, (np.dot(self.dictionary, h) - input))
			h = np.sign(h) * np.maximum(np.abs(h)- self.lr * self.L1, hidden_zeros)
			
			if ((np.abs(old_h - h) < convergence_threshold).all()):
				converged = True
				
		return h
	
	def dict_update(self, inputs):
		"""
		Dictionary learning algorithm which uses block-coordinate descent
		- uses gradient descent to train and update self.dictionary
		- returns nothing
		Argument ``inputs`` is a Numpy 2D array.
		"""
		A = np.zeros((self.hidden_size, self.hidden_size))
		B = np.zeros((len(inputs[0]), self.hidden_size))
		
		# First, an inference pass infers a sparse representation for each input, given the dictionary
		counter=1
		for input in inputs:
			if (counter % 100 == 0):
				print  "Inference: " + str(counter) + ' out of ' + str(len(inputs)), #'\r' +
				print "time", datetime.datetime.now()
			counter += 1
			h = self.infer(input)
			A += np.outer(h, h.T)
			B += np.outer(input, h.T)
		print "(done)"
		
		# Then, update the dictionary with block-coordinate descent
		old_dictionary = np.ones(self.dictionary.shape)
		while ((np.abs(old_dictionary - self.dictionary) > 0.00000001).all()):
			old_dictionary = np.copy(self.dictionary)
			for i in range(len(self.dictionary[0,:])):
				self.dictionary[:,i] = 1/A[i,i] * (B[:,i] - np.dot(self.dictionary, A[i]) + self.dictionary[:,i] * A[i,i])
				self.dictionary[:,i] /= self.dictionary[:,i].sum(axis=0)
		
	def show_filters(self):
		for i in range(len(self.dictionary)):
			self.dictionary[i] = (self.dictionary[i] - self.dictionary[i].min()) / (self.dictionary[i].max() - self.dictionary[i].min())
		from matplotlib.pylab import show, draw, ion
		import mlpython.misc.visualize as mlvis
		mlvis.show_filters(0.5*self.dictionary.T,
						   len(self.dictionary[0]),
						   16,
						   8,
						   10,len(self.dictionary[0])/10,2)
		show()		
		