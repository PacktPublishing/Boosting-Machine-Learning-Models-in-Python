"""
Searching for optimal parameters.
"""
from section1_video5_data import get_data

from sklearn import model_selection
from xgboost import XGBClassifier
import numpy as np

seed=123

# Load prepared data
X, Y = get_data('../data/video1_diabetes.csv')
# Build our single model
c = XGBClassifier(random_state=seed, n_estimators=60, max_depth=3)

learning_rate_range=[ 0.0001, 0.001, 0.01, 0.1, 0.0002, 0.002, 0.02, 0.2, 0.3]
n_trees = range(10, 500, 50)
max_depth = range(3, 5)
print(learning_rate_range)
params_to_search = dict(learning_rate=learning_rate_range)
grid_search = model_selection.GridSearchCV(c, params_to_search, scoring="neg_log_loss", n_jobs=-1, cv=10, iid=False)
grid_result = grid_search.fit(X, Y)
print("Found best params: %s" % (grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for m, p in zip(means, params):
	print("%f: %r" % (m, p))

# Check accuracy of a classfier once again
c = XGBClassifier(random_state=seed, n_estimators=60, max_depth=3, **grid_result.best_params_)
results = c.fit(X, Y)
# using 10-fold Cross Validation.
results_kfold_model = model_selection.cross_val_score(c, X, Y, cv=10)
print("XGBoost accuracy:\t{:2.2f}%".format(results_kfold_model.mean()*100))
