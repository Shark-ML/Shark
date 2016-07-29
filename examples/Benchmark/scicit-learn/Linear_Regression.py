import numpy as np
from sklearn import linear_model
import time

dataset = np.loadtxt('blogData_train.csv', dtype = np.dtype('d'), delimiter=',')
dataset_X,dataset_Y = np.split(dataset,[280],1);


regr = linear_model.LinearRegression(copy_X=True, fit_intercept=True, normalize=False)

start = time.time()
regr.fit(dataset_X, dataset_Y)
end = time.time()

print("Residual sum of squares: %.2f" % np.mean((regr.predict(dataset_X) - dataset_Y) ** 2))
print('Time: \n', end-start)