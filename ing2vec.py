import pandas as pd
import re
import theano
import theano.tensor as T
import numpy as np
import Skipgram
import random
import pickle

def one_hot(input, length) :
	one_hot_array = np.zeros((length, ), dtype=theano.config.floatX)
	one_hot_array[input] = 1.0
	return one_hot_array

def generate_matrix(m, n, row) :
	array = np.zeros((m,n), dtype=theano.config.floatX)
	array[row,:] = np.ones((n,), dtype=theano.config.floatX) 
	return array

## Parse Data : read json & modify some ingredients' name, make ingredient vocabulary

print "Parsing Data..."

train = pd.read_json('train_data.json') # 39774 datas
test = pd.read_json('test_data.json')
ingredient_book = {}
ing_list = []
cuisine_book = {}
cuisine_list = []
ing_book_count = 0
cui_book_count = 0

for i in range(train.shape[0]) :
	ingredient_list = train.loc[i, 'ingredients']
	cuisine = train.loc[i, 'cuisine']
	if not (cuisine in cuisine_book) :
		cuisine_book[cuisine] = cui_book_count
		cuisine_list.append(cuisine)
		cui_book_count += 1
	for j in range(len(ingredient_list)) :
		ingredient_list[j]=ingredient_list[j].lower()
		# erase 'oz' phrase
		idx = ingredient_list[j].find("oz.) ")
		if idx!=-1 :
			ingredient_list[j] = ingredient_list[j][idx+5:]
		if not (ingredient_list[j] in ingredient_book) :
			ingredient_book[ingredient_list[j]] = ing_book_count
			ing_list.append(ingredient_list[j])
			ing_book_count += 1

for i in range(test.shape[0]) :
	ingredient_list = test.loc[i, 'ingredients']
	for j in range(len(ingredient_list)) :
		ingredient_list[j]=ingredient_list[j].lower()
		# erase 'oz' phrase
		idx = ingredient_list[j].find("oz.) ")
		if idx!=-1 :
			ingredient_list[j] = ingredient_list[j][idx+5:]

# ingredient book has 6696 kinds ingredients.
# cuisine book has 20 kinds of cuisines.

pic = open('books.txt', 'wb')
pickle.dump([ingredient_book, ing_list, cuisine_book, cuisine_list], pic)
pic.close()
stop

## ing2vec region

print "ing2vec.."

# layer definition
input = T.iscalar(name='input')
skipgram = Skipgram.Skipgram(input, vector_dim=200, vocab_size=len(ingredient_book), hidden_size=100)
window = 4
learning_rate_ing = 0.06
learning_rate_cui = 0.015
# cost definition
y = T.matrix(name='y') # predict ingredient
y2 = T.matrix(name='y2') # predict cuisine
ratio = 1.0 # ratio between ing / cuisine
cost = T.nnet.categorical_crossentropy(T.tile(skipgram.output1, (window-1, 1)), y).mean() + ratio*T.nnet.categorical_crossentropy(skipgram.output2, y2).mean()
# updates
updates = []
params = skipgram.params
types = skipgram.types
grad = T.grad(cost, params)
for param_i, grad_i, type_i in zip(params, grad, types) :
	if type_i == 0 : updates.append((param_i, param_i - learning_rate_ing*grad_i))
	if type_i == 1 : updates.append((param_i, param_i - learning_rate_cui*grad_i))
# functions 
f = theano.function([input, y, y2], cost, updates=updates)
test_f = theano.function([input, y, y2], [skipgram.output1, skipgram.output2], on_unused_input='ignore')

for epoch in range(5) :
	print epoch
	random_idx = np.random.permutation(train.shape[0]) # shuffle order randomly
	nowcost = 0.0
	if epoch>0 : 
		learning_rate_cui *= 0.7
		learning_rate_ing *= 0.7

	for i in range(train.shape[0]) :
		if i%100==0 :
			print "Epoch %d, Case %d, Prev Cost : %f" % (epoch, i, nowcost)
		if i==train.shape[0]/2 :
			learning_rate_cui *= 0.7
			learning_rate_ing *= 0.7
		ingredient_list = train.loc[random_idx[i], 'ingredients']
		cuisine = train.loc[random_idx[i], 'cuisine']

		for j in range(len(ingredient_list)) :
			# randomly select other three ingredients
			idx1 = -1; idx2 = -1; idx3 = -1
			if(len(ingredient_list)>=2) :	
				while (idx1==j or idx1<0) : idx1=random.randint(0, len(ingredient_list)-1)
			if(len(ingredient_list)>=3) : 
				while (idx2==j or idx2==idx1 or idx2<0) : idx2=random.randint(0, len(ingredient_list)-1)
			if(len(ingredient_list)>=4) :
				while (idx3==j or idx3==idx1 or idx3==idx2 or idx3<0) : idx3 = random.randint(0, len(ingredient_list)-1)
			
			# make y, y2 matrix for this case
			now_y = np.zeros((window-1, len(ingredient_book)), dtype=theano.config.floatX)
			if(idx1>=0) : now_y[0, ingredient_book[ingredient_list[idx1]]] = 1.0
			if(idx2>=0) : now_y[1, ingredient_book[ingredient_list[idx2]]] = 1.0
			if(idx3>=0) : now_y[2, ingredient_book[ingredient_list[idx3]]] = 1.0
			now_y2 = np.zeros((1, 20), dtype=theano.config.floatX)
			now_y2[0, cuisine_book[cuisine]] = 1.0
			# run
			nowcost = f(ingredient_book[ingredient_list[j]], now_y, now_y2)

	save_string = 'variable_save_'+str(epoch)+'.txt'
	f_write = open(save_string, 'wb')
	pickle.dump(skipgram.W.get_value(), f_write)
	f_write.close()	