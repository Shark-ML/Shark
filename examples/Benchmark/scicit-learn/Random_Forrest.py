import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
import time

dataset_X, dataset_Y = load_svmlight_file('cod-rna')
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features="auto")
start = time.time()
model.fit(dataset_X, dataset_Y)
end = time.time()

score = 1-accuracy_score(dataset_Y, model.predict(dataset_X))
print (end-start, score)