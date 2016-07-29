import numpy as np
from sklearn import linear_model
from sklearn.metrics import log_loss
from sklearn.datasets import load_svmlight_file
import time


alpha = 0.1
dataset_X_dense, dataset_Y_dense = load_svmlight_file('mnist')
dataset_X_dense = dataset_X_dense.toarray()
dataset_Y_dense = [ y % 2 for y in dataset_Y_dense]

dataset_X_sparse, dataset_Y_sparse = load_svmlight_file('rcv1_train.binary')


model = linear_model.LogisticRegression(C=1/(dataset_X_dense.shape[0]*alpha), fit_intercept=True, solver='sag', tol=0.0, max_iter=200)
start = time.time()
model.fit(dataset_X_dense, dataset_Y_dense)
end = time.time()

score = log_loss(dataset_Y_dense, model.predict_proba(dataset_X_dense))
print("iter: ", model.n_iter_)
print("Cross-Entropy: %.4f" % score)
print('Time: \n', end-start)

model = linear_model.LogisticRegression(C=1/(dataset_X_sparse.shape[0]*alpha), fit_intercept=True, solver='sag', tol=0, max_iter=2000)
start = time.time()
model.fit(dataset_X_sparse, dataset_Y_sparse)
end = time.time()

score = log_loss(dataset_Y_sparse, model.predict_proba(dataset_X_sparse))
print("iter: ", model.n_iter_)
print("Cross-Entropy: %.4f" % score)
print('Time: \n', end-start)