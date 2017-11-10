import numpy as np
from sklearn import linear_model
from sklearn.metrics import log_loss
from sklearn.datasets import load_svmlight_file
import time


dataset_X, dataset_Y = load_svmlight_file('mnist')
dataset_X = dataset_X.toarray()

alpha = 1.0
C=1/(dataset_X.shape[0]*alpha)

model = linear_model.LogisticRegression(C=C, fit_intercept=True, multi_class='multinomial', solver='lbfgs', max_iter=200, n_jobs=8)

start = time.time()
model.fit(dataset_X, dataset_Y)
end = time.time()

score = log_loss(dataset_Y, model.predict_proba(dataset_X))

print("Cross-Entropy: %.4f" % score)
print('Time: \n', end-start)