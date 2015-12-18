import numpy as np
import theano
import theano.tensor as T
import pickle
import scipy.spatial.distance as distance
import pandas as pd
import ClassifyNetwork
import math

def find_closest(target, array, ingredient_list) :
	cossim_list = []
	for i in range(6696) :
		cossim_list.append((i, -distance.cosine(target.flatten(), array[i,:].flatten())+1))
	cossim_list = sorted(cossim_list, key=lambda element: element[1], reverse=True)

	#print "Closest 10 Ingredients of "+ingredient_list[center]
	print "Closest 10 Ingredients to target : "
	for i in range(10) :
		print ingredient_list[cossim_list[i][0]].encode('utf-8')+"\t%f" % cossim_list[i][1]


## fetch data
f = open('variable_save_4.txt', 'rb') 
ing2vec_array = pickle.load(f)
f.close()
f = open('books.txt', 'rb')
[ingredient_book, ingredient_list, cuisine_book, cuisine_list] = pickle.load(f)
f.close()
train = pd.read_json('train_data.json') # 39774 datas
test = pd.read_json('test_data.json')

# normalize ing2vec_array
ing2vec_array = np.add(ing2vec_array, -np.mean(ing2vec_array)) / np.std(ing2vec_array)

for i in range(train.shape[0]) :
	ingredient_list = train.loc[i, 'ingredients']
	cuisine = train.loc[i, 'cuisine']
	for j in range(len(ingredient_list)) :
		ingredient_list[j]=ingredient_list[j].lower()
		# erase 'oz' phrase
		idx = ingredient_list[j].find("oz.) ")
		if idx!=-1 :
			ingredient_list[j] = ingredient_list[j][idx+5:]

for i in range(test.shape[0]) :
	ingredient_list = test.loc[i, 'ingredients']
	for j in range(len(ingredient_list)) :
		ingredient_list[j]=ingredient_list[j].lower()
		# erase 'oz' phrase
		idx = ingredient_list[j].find("oz.) ")
		if idx!=-1 :
			ingredient_list[j] = ingredient_list[j][idx+5:]

## network & training

# use first 38000 for training, last 1774 for validation
train_num = 38000
val_num = 1774

# use randomly chosen 5 ingredients to train the network

input = T.matrix(name='input')
p = theano.shared(0.3)
classify_network = ClassifyNetwork.ClassifyNetwork(input, 500, 350, p)
learning_rate = 0.2

y = T.matrix(name='y')
cost = T.nnet.categorical_crossentropy(classify_network.output, y).sum()

params = classify_network.params
paramins = classify_network.paramins
grad = T.grad(cost, params)
updates = []
for param_i, grad_i, paramin_i in zip(params, grad, paramins) :
	updates.append((param_i, param_i - (learning_rate/math.sqrt(paramin_i))*grad_i))

f = theano.function([input, y], cost, updates=updates)
test_f = theano.function([input], classify_network.output)

for epoch in range(0) :
	random_idx = np.random.permutation(train_num) # shuffle order randomly
	
	for i in range(train_num) :
		if i%2000==0 : print str(i)
		now_recipe = train.loc[random_idx[i]]
		# use ingredients that we already have
		temp_ingredient_list = now_recipe['ingredients']
		now_ingredient_list = []
		for ing_name in temp_ingredient_list :
			if ing_name in ingredient_book : now_ingredient_list.append(ing_name)
		ingredient_num = len(now_ingredient_list)

		# make y matrix
		now_y = np.zeros((1, 20), dtype=theano.config.floatX)
		now_y[0, cuisine_book[now_recipe['cuisine']]]=1.0

		# make input vector
		input_vector = np.zeros((1, 200), dtype=theano.config.floatX)
		for ing_name in now_ingredient_list :
			ing_id = ingredient_book[ing_name]
			input_vector += ing2vec_array[ing_id,:]
		# normalize it 
		input_vector = input_vector / np.linalg.norm(input_vector, axis=1)[0]

		# run
		nowcost = f(input_vector, now_y)

		# learning rate decay
		if i%19000==0 and i>0 and epoch>=2 and epoch<6:
			learning_rate *= 0.7

		# validation data check : just once
		if i%19000==0 and (i>0 or epoch>0):
			p.set_value(0.0)
			print "Validation data check for epoch %d, iteration %d" % (epoch, i)
			error_num = 0.0
			for j in range(val_num) :
				now_recipe = train.loc[train_num+j]
				temp_ingredient_list = now_recipe['ingredients']
				now_ingredient_list = []
				for ing_name in temp_ingredient_list :
					if ing_name in ingredient_book : now_ingredient_list.append(ing_name)
				ingredient_num = len(now_ingredient_list)

				# make input vector
				input_vector = np.zeros((1, 200), dtype=theano.config.floatX)
				for ing_name in now_ingredient_list :
					ing_id = ingredient_book[ing_name]
					input_vector += ing2vec_array[ing_id,:]
				# normalize it 
				input_vector = input_vector / np.linalg.norm(input_vector, axis=1)[0]

				result = test_f(input_vector)
				if result.argmax(axis=1)[0] != cuisine_book[now_recipe['cuisine']] :
					error_num += 1.0
			error_rate = error_num / val_num
			print "Error rate : %f" % error_rate
			p.set_value(0.3)

columns=['id', 'cuisine']
final_data = pd.DataFrame(index=np.arange(test.shape[0]), columns=columns)

# final : write csv
p.set_value(0.0)
for i in range(test.shape[0]) :
	print i
	now_recipe = test.loc[i]
	# use ingredients that we already have
	temp_ingredient_list = now_recipe['ingredients']
	now_ingredient_list = []
	for ing_name in temp_ingredient_list :
		if ing_name in ingredient_book : now_ingredient_list.append(ing_name)
	ingredient_num = len(now_ingredient_list)

	# make input vector
	input_vector = np.zeros((1, 200), dtype=theano.config.floatX)
	for ing_name in now_ingredient_list :
		ing_id = ingredient_book[ing_name]
		input_vector += ing2vec_array[ing_id,:]
	# normalize it 
	input_vector = input_vector / np.linalg.norm(input_vector, axis=1)[0]
	result = test_f(input_vector).argmax(axis=1)
	final_data.loc[i, 'id'] = now_recipe['id']
	final_data.loc[i, 'cuisine'] = cuisine_list[result]

final_data.to_csv("result.csv", encoding='utf-8', index=False)