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

def calc_gaussian(xArray, meanArray, covVec) :
	distArray = xArray - meanArray
	valVec = np.power(np.linalg.norm(distArray, axis=1), 2)
	valVec = np.exp(np.divide(valVec, covVec*(-2)))
	return valVec

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

cuisine_count = np.zeros((20,))
for i in range(train.shape[0]) :
	ingredient_list = train.loc[i, 'ingredients']
	cuisine = train.loc[i, 'cuisine']
	for j in range(len(ingredient_list)) :
		ingredient_list[j]=ingredient_list[j].lower()
		# erase 'oz' phrase
		idx = ingredient_list[j].find("oz.) ")
		if idx!=-1 :
			ingredient_list[j] = ingredient_list[j][idx+5:]
	cuisine_count[cuisine_book[cuisine]]+=1.0

for i in range(test.shape[0]) :
	ingredient_list = test.loc[i, 'ingredients']
	for j in range(len(ingredient_list)) :
		ingredient_list[j]=ingredient_list[j].lower()
		# erase 'oz' phrase
		idx = ingredient_list[j].find("oz.) ")
		if idx!=-1 :
			ingredient_list[j] = ingredient_list[j][idx+5:]

print "Start"

# RBF : use k-means to make n kernels

kernel_num = 10000

vec_array = np.zeros((train.shape[0], 200))
count_array = np.zeros((kernel_num,))
assign_array = np.zeros((train.shape[0],))
mean_matrix = np.zeros((kernel_num, 200))
cov_matrix = np.zeros((kernel_num,))

frequent_list = ['salt', 'onions', 'olive oil', 'water', 'garlic', 'sugar', 'garlic cloves', 'butter', 'ground black pepper', 'pepper', 'black pepper', 'oil', 'unsalted butter', 'red bell pepper', 'green onions']
top_list = ['kimchi', 'Gochujang base', 'irish whiskey', 'irish cream liqueur', 'guinness beer', 'baileys irish cream liqueur', 'serrano ham', 'spanish chorizo', 'spanish paprika', 'stilton cheese', 'suet', 'black treacle', 'strawberry jam', 'marmite', 'granola', 'chocolate sprinkles', 'chia seeds', 'passion fruit juice', 'pickled beets', 'farmer cheese']
# convert foods into vector

for i in range(train.shape[0]) :
	now_vector = np.zeros((200,))
	for str in train.loc[i, 'ingredients'] :
		weight = 1.0
		if str in frequent_list : weight = 0.2 # weaken the effect of frequent ingredients
		if str in top_list : weight = 3.0 
		now_vector+=weight*ing2vec_array[ingredient_book[str], :]
	now_vector /= len(train.loc[i, 'ingredients'])
	now_vector /= np.linalg.norm(now_vector)
	vec_array[i,:] = 10 * now_vector

## k-means session
print "Start K-means"

#(1) randomly set mean_matrix
random_idx = np.random.permutation(train.shape[0])
for i in range(kernel_num) :
	mean_matrix[i,:]=vec_array[random_idx[i],:]

#(2) assign & calc
for epoch in range(7) :
	print "epoch %d" % epoch
	# assign
	count_array = np.zeros((kernel_num,))
	for i in range(train.shape[0]) :
		distarray = mean_matrix - np.tile(vec_array[i,:], (kernel_num, 1))
		go_kernel = np.argmin(np.linalg.norm(distarray, axis=1))
		count_array[go_kernel] += 1.0
		assign_array[i] = go_kernel

	# calc	
	mean_matrix = np.zeros((kernel_num, 200))
	for i in range(train.shape[0]) :
		mean_matrix[assign_array[i],:] += vec_array[i,:]
	for i in range(kernel_num) :
		mean_matrix[i,:] /= count_array[i]
	print count_array

	# prevent 0
	for i in range(kernel_num) : 
		if count_array[i] == 0 :
			mean_matrix[i,:] = np.tile([100000.0], (1, 200))

# calculate cov array : restrict to diagonal
for i in range(train.shape[0]) :
	cov_matrix[assign_array[i]] += (np.linalg.norm(vec_array[i,:]-mean_matrix[assign_array[i],:])**2)*3.0
for i in range(kernel_num) :
	cov_matrix[i] /= count_array[i]
	cov_matrix[i] = math.sqrt(cov_matrix[i])

# use only >1 s

deleteList = []
for i in range(kernel_num) :
	if (count_array[i]==0 or cov_matrix[i]==0) :
		deleteList.append(i)
			
kernel_num -= len(deleteList)	
cov_matrix = np.delete(cov_matrix, deleteList)
mean_matrix = np.delete(mean_matrix, deleteList, axis=0)

# weight calc : normal equation
# weight : (kernel_num+1, 20), y : (traindata num, 20)
H = np.zeros((train.shape[0], kernel_num))
W = np.zeros((kernel_num, 20))
Y = np.zeros((train.shape[0], 20))

print "make normal equation matrix"
for i in range(train.shape[0]) :	
	if i%1000==0 : print "%d" % i
	#for j in range(kernel_num) :
	#	H[i,j] = rv[j].pdf(vec_array[i,:])
	H[i,:] = calc_gaussian(np.tile(vec_array[i,:], (kernel_num, 1)), mean_matrix, cov_matrix)
	cuisine_idx = cuisine_book[train.loc[i, 'cuisine']]
	Y[i,cuisine_idx]=1.0

print "now normal equation"
W = np.dot(np.dot(np.linalg.inv(np.dot(H.T, H)), H.T), Y)

wrong_count = np.zeros((20,))
# test (now : just from training data)
failcount = 0
for i in range(500) :
	count = np.zeros((20,))
	now_vector = vec_array[i,:]
	now_in = np.zeros((kernel_num,))
	now_in = calc_gaussian(np.tile(vec_array[i,:], (kernel_num, 1)), mean_matrix, cov_matrix)
	
	result_array = np.dot(now_in.reshape(1, kernel_num), W).reshape(20, )
	cuisine_idx = cuisine_book[train.loc[i, 'cuisine']]

	if cuisine_idx != result_array.argmax() :
		failcount+=1
		wrong_count[cuisine_idx]+=1.0

print failcount

print "Final write"
# real test & write
final_data = pd.DataFrame(index=np.arange(test.shape[0]), columns=['id', 'cuisine'])

for i in range(test.shape[0]) :
	
	now_recipe = test.loc[i]
	# use ingredients that we already have
	temp_ingredient_list = now_recipe['ingredients']
	now_ingredient_list = []
	for ing_name in temp_ingredient_list :
		if ing_name in ingredient_book : now_ingredient_list.append(ing_name)
	ingredient_num = len(now_ingredient_list)

	# make input vector
	input_vector = np.zeros((200,))
	for str in now_ingredient_list :
		weight = 1.0
		if str in frequent_list : weight = 0.2 # weaken the effect of frequent ingredients
		if str in top_list : weight = 3.0
		input_vector+=weight*ing2vec_array[ingredient_book[str], :]
		
	# normalize it 
	input_vector = input_vector / np.linalg.norm(input_vector) * 10
	now_in = calc_gaussian(np.tile(input_vector, (kernel_num, 1)), mean_matrix, cov_matrix)
	result_array = np.dot(now_in.reshape(1, kernel_num), W).reshape(20, )

	final_data.loc[i, 'id'] = now_recipe['id']
	final_data.loc[i, 'cuisine'] = cuisine_list[result_array.argmax()]

final_data.to_csv("result.csv", encoding='utf-8', index=False)
print "Write complete!"
