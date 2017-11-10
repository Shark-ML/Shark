import numpy as np
from sklearn import neighbors
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from sklearn.externals.joblib.parallel import parallel_backend

import time

dataset_X, dataset_Y = load_svmlight_file('cod-rna')
dataset_X=dataset_X.toarray()

mnist_X, mnist_Y = load_svmlight_file('mnist')
mnist_X=mnist_X.toarray()

with parallel_backend('threading'):
	model = neighbors.KNeighborsClassifier(n_neighbors=10, algorithm='kd_tree', n_jobs=-1)
	start = time.time()
	model.fit(dataset_X, dataset_Y)
	score = 1-accuracy_score(dataset_Y, model.predict(dataset_X))
	end = time.time()
	print ("KD-Tree ", end-start, score)

	model = neighbors.KNeighborsClassifier(n_neighbors=10, algorithm='brute', n_jobs=-1)
	start = time.time()
	model.fit(dataset_X, dataset_Y)
	score = 1-accuracy_score(dataset_Y, model.predict(dataset_X))
	end = time.time()
	print ("Brute-Force ", end-start, score)

	model = neighbors.KNeighborsClassifier(n_neighbors=10, algorithm='brute', n_jobs=-1)
	start = time.time()
	model.fit(mnist_X, mnist_Y)
	score = 1-accuracy_score(mnist_Y, model.predict(mnist_X))
	end = time.time()
	print ("Brute-Force-MNIST ", end-start, score)