import numpy as np
from sklearn import svm
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
import time

dataset_X_sparse, dataset_Y_sparse = load_svmlight_file('rcv1_train.binary')

for i in range(0,6):
	C= 10.0**i
	model = svm.LinearSVC(C=C, loss='hinge', tol=0.001, fit_intercept= False, max_iter=1000000)
	start = time.time()
	model.fit(dataset_X_sparse, dataset_Y_sparse)
	end = time.time()

	score = 1-accuracy_score(dataset_Y_sparse, model.predict(dataset_X_sparse))
	print (C, end-start, score)