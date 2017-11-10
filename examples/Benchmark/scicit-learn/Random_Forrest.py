import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import time

X, Y = load_svmlight_file('covtype')
dataset_X, X_test, dataset_Y, y_test = train_test_split(X, Y, train_size=400000, random_state=42)
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features="sqrt")
start = time.time()
model.fit(dataset_X, dataset_Y)
end = time.time()

score_train = accuracy_score(dataset_Y, model.predict(dataset_X))
score_test = accuracy_score(y_test, model.predict(X_test))
print (end-start, score_train, score_test)

dataset_X, dataset_Y = load_svmlight_file('cod-rna')
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, max_features = None)
start = time.time()
model.fit(dataset_X, dataset_Y)
end = time.time()

score_train = mean_squared_error(dataset_Y, model.predict(dataset_X))
print (end-start, score_train)
