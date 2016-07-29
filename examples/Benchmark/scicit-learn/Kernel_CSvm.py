import numpy as np
from sklearn import svm
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
import time
dataset_X, dataset_Y = load_svmlight_file('cod-rna')
dataset_X= dataset_X.toarray()
for i in range(0,1):
	C= 10.0**i
	model = svm.SVC(C=C, kernel='rbf', gamma =1.0, tol=0.001, cache_size=256)
	start = time.time()
	model.fit(dataset_X, dataset_Y)
	end = time.time()

	score = 1-accuracy_score(dataset_Y, model.predict(dataset_X))
	print (C, end-start, score, flush=True)
