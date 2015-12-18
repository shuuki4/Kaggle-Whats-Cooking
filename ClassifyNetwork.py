import numpy as np
import theano
import theano.tensor as T
import time
import math

class ClassifyNetwork(object) :
	def __init__(self, input, hidden1_size, hidden2_size, p) :
		## params
		# input = input vector (1, 1000) matrix
		# hidden1_size = hidden layer 1's size
		# hidden2_size = hidden layer 2's size
		# p = Dropconnect rate : shared variable

		input_num = 1000
		w1_bound = math.sqrt(input_num)
		w2_bound = math.sqrt(hidden1_size)
		w3_bound = math.sqrt(hidden2_size)

		# initialize weights randomly : W1, W2, W3, b1, b2, b3
		rng = np.random.RandomState(int(time.time()))

		self.W1 = theano.shared(np.asarray(rng.uniform(low=-1.0/w1_bound, high=1.0/w1_bound, size=(input_num, hidden1_size)), dtype=theano.config.floatX), name='W1', borrow=True)
		self.W2 = theano.shared(np.asarray(rng.uniform(low=-1.0/w2_bound, high=1.0/w2_bound, size=(hidden1_size, hidden2_size)), dtype=theano.config.floatX), name='W2', borrow=True)
		self.W3 = theano.shared(np.asarray(rng.uniform(low=-1.0/w3_bound, high=1.0/w3_bound, size=(hidden2_size, 20)), dtype=theano.config.floatX), name='W3', borrow=True)

		self.b1 = theano.shared(np.asarray(np.zeros(hidden1_size,), dtype=theano.config.floatX), name='b1', borrow=True)
		self.b2 = theano.shared(np.asarray(np.zeros(hidden2_size,), dtype=theano.config.floatX), name='b2', borrow=True)
		self.b3 = theano.shared(np.asarray(np.zeros(20,), dtype=theano.config.floatX), name='b3', borrow=True)

		# DropConnect
		srng = T.shared_randomstreams.RandomStreams(int(time.time()))
		p_val = p.get_value()
		select_array1 = T.cast(srng.binomial(n=1, p=1-p_val, size=(input_num, hidden1_size)), theano.config.floatX)
		select_array2 = T.cast(srng.binomial(n=1, p=1-p_val, size=(hidden1_size, hidden2_size)), theano.config.floatX)
		select_array3 = T.cast(srng.binomial(n=1, p=1-p_val, size=(hidden2_size, 20)), theano.config.floatX)
		select_vec1 = T.cast(srng.binomial(n=1, p=1-p_val, size=(hidden1_size,)), theano.config.floatX)
		select_vec2 = T.cast(srng.binomial(n=1, p=1-p_val, size=(hidden2_size,)), theano.config.floatX)
		select_vec3 = T.cast(srng.binomial(n=1, p=1-p_val, size=(20, )), theano.config.floatX)

		# feed-forward, activation = leaky ReLU
		hidden1_temp = T.dot(input, self.W1*select_array1)+self.b1*select_vec1
		self.hidden1 = T.switch(hidden1_temp<0, 0.01*hidden1_temp, hidden1_temp)
		hidden2_temp = T.dot(self.hidden1, self.W2*select_array2)+self.b2*select_vec2
		self.hidden2 = T.switch(hidden2_temp<0, 0.01*hidden2_temp, hidden2_temp)
		out_temp = T.dot(self.hidden2, self.W3*select_array3)+self.b3*select_vec3
		#self.output = T.nnet.softmax(out_temp)
		out_temp2 = T.exp(out_temp - out_temp.max(axis=1, keepdims=True))
		self.output = out_temp2 / out_temp2.sum(axis=1, keepdims=True)

		# save parameters
		self.params = [self.W3, self.b3, self.W2, self.b2, self.W1, self.b1]
		self.paramins = [hidden2_size, hidden2_size, hidden1_size, hidden1_size, input_num, input_num]



