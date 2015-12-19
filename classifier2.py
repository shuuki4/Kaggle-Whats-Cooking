import numpy as np
import theano
import theano.tensor as T
import pickle
import scipy.spatial.distance as distance
from scipy.stats import multivariate_normal
import pandas as pd
import ClassifyNetwork
import math

def find_closest(target, array, cuisine_list) :
	cossim_list = []
	for i in range(20) :
		cossim_list.append((i, -distance.cosine(array[target,:], array[i,:].flatten())+1))
	cossim_list = sorted(cossim_list, key=lambda element: element[1], reverse=True)

	#print "Closest 10 Ingredients of "+ingredient_list[center]
	print "Closest Cuisines in order : "
	for i in range(20) :
		print cuisine_list[cossim_list[i][0]].encode('utf-8')+"\t%f" % cossim_list[i][1]

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

print "Start"

count_array = np.zeros((20,))
mean_matrix = np.zeros((20, 200))
cov_matrix = np.zeros((20, 200, 200))

for i in range(train.shape[0]) :
	now_vector = np.zeros((200,))
	for str in train.loc[i, 'ingredients'] :
		now_vector+=ing2vec_array[ingredient_book[str], :]
	now_vector /= len(train.loc[i, 'ingredients'])
	now_vector /= np.linalg.norm(now_vector)
	cuisine_idx = cuisine_book[train.loc[i, 'cuisine']]
	mean_matrix[cuisine_idx, :] += now_vector
	count_array[cuisine_idx] += 1.0

for i in range(20) :
	mean_matrix[i, :] /= count_array[i]

# calculate cov array
for i in range(train.shape[0]) :
	now_vector = np.zeros((200,))
	for str in train.loc[i, 'ingredients'] :
		now_vector+=ing2vec_array[ingredient_book[str], :]
	now_vector /= len(train.loc[i, 'ingredients'])
	now_vector /= np.linalg.norm(now_vector)
	cuisine_idx = cuisine_book[train.loc[i, 'cuisine']]

	cov_matrix[cuisine_idx, :, :] += np.dot((now_vector-mean_matrix[cuisine_idx, :]).reshape((200,1)), (now_vector-mean_matrix[cuisine_idx, :]).reshape((1,200)))

for i in range(20) :
	cov_matrix[i, :] /= count_array[i]
	count_array[i] /= train.shape[0]

rv = []
for i in range(20) :
	rv.append(multivariate_normal(mean=mean_matrix[i,:], cov=cov_matrix[i,:,:]))

count = 0
for i in range(500) :
	now_vector = np.zeros((200,))
	for str in train.loc[i, 'ingredients'] :
		now_vector+=ing2vec_array[ingredient_book[str], :]
	now_vector /= len(train.loc[i, 'ingredients'])
	now_vector /= np.linalg.norm(now_vector)
	cuisine_idx = cuisine_book[train.loc[i, 'cuisine']]

	result_array = np.zeros((20,))
	for j in range(20) :
		result_array[j] = rv[j].logpdf(now_vector)+math.log(count_array[j])
	#print cuisine_idx, result_array.argmax()
	if cuisine_idx != result_array.argmax() : count+=1
print count