"""
Searching for optimal parameters.
"""
from section1_video5_data import get_data

from sklearn import model_selection
from xgboost import XGBClassifier

seed=123

# Load prepared data
X, Y = get_data('../data/video1_diabetes.csv')
# Build our single model
c = XGBClassifier(random_state=seed)

#n_trees = range(500, 1000, 50)
#max_depth = range(1, 3) # 72.44% - {'max_depth': 1, 'n_estimators': 500}
#max_depth = range(3, 5)  # 68.70% - {'max_depth': 3, 'n_estimators': 500}

n_trees = range(10, 500, 50)
max_depth = range(3, 5) # - 74.10% {'max_depth': 1, 'n_estimators': 260}
#max_depth = range(1, 3) # - 72.24% {'max_depth': 3, 'n_estimators': 60}

params_to_search = dict(n_estimators=n_trees, max_depth=max_depth)
grid_search = model_selection.GridSearchCV(c, params_to_search, scoring="neg_log_loss", n_jobs=-1, cv=10, iid=False)
grid_result = grid_search.fit(X, Y)
print("Found best params: %s" % (grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for m, p in zip(means, params):
	print("%f: %r" % (m, p))

# Check accuracy of a classfier once again
c = XGBClassifier(random_state=seed, **grid_result.best_params_)
results = c.fit(X, Y)
# using 10-fold Cross Validation.
results_kfold_model = model_selection.cross_val_score(c, X, Y, cv=10)
print("XGBoost accuracy:\t{:2.2f}%".format(results_kfold_model.mean()*100))
