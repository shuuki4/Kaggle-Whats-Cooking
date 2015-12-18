import numpy as np
import theano
import theano.tensor as T
import time
import math

class Skipgram(object) : 
	def __init__(self, input, vector_dim, vocab_size, hidden_size) :
		# input : index of '1' entry
		# vector_dim : vector size to use (N)
		# vocab_size : size of vocab dictionary (V)
		# hidden_size : hidden layer size for cuisine classification

		rng = np.random.RandomState(int(time.time()))
		w_bound = math.sqrt(vocab_size)
		w2_bound = math.sqrt(vector_dim)
		w3_bound = math.sqrt(hidden_size)
		# W is for original skip-gram
		self.W = theano.shared(np.asarray(rng.uniform(low=-1.0/w_bound, high=1.0/w_bound, size=(vocab_size, vector_dim)), dtype=theano.config.floatX), name='W', borrow=True)
		# W2, W3 is for cuisine : n->hid->20
		self.W2 = theano.shared(np.asarray(rng.uniform(low=-1.0/w2_bound, high=1.0/w2_bound, size=(vector_dim, hidden_size)), dtype=theano.config.floatX), name='W2', borrow=True)
		self.b2 = theano.shared(np.zeros((hidden_size,), dtype=theano.config.floatX))
		self.W3 = theano.shared(np.asarray(rng.uniform(low=-1.0/w3_bound, high=1.0/w3_bound, size=(hidden_size, 20)), dtype=theano.config.floatX), name='W3', borrow=True)
		self.b3 = theano.shared(np.zeros((20,), dtype=theano.config.floatX))

		# calculation
		hidden = self.W[input, :]
		out = T.sum(hidden * self.W, axis=1)
		#e_x1 = T.exp(out - out.max(keepdims=True))
		#self.output1 = e_x1 / e_x1.sum(keepdims=True)
		self.output1 = T.nnet.softmax(out)

		temp = T.dot(hidden, self.W2)+self.b2
		hidden2 = T.switch(temp<0, 0.01*temp, temp) # Leaky Relu for nonlinearity
		out2 = T.dot(hidden2, self.W3)+self.b3
		#e_x2 = T.exp(out2 - out2.max(keepdims=True))
		#self.output2 = e_x2 / e_x2.sum(keepdims=True)
		self.output2 = T.nnet.softmax(out2)

		# save params
		self.params = [self.W3, self.b3, self.W2, self.b2, self.W]
		self.types = [1, 1, 1, 1, 0]