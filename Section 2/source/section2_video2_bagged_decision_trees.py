"""
Compare the accuracy of a single Decision Three
to a Bagged Decision Trees.
"""
from section2_video1_data import get_data

from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

seed=123

# Load prepared data
X, Y = get_data()
# Load our single model
dtc = DecisionTreeClassifier(random_state=seed)
# Create our bagged algorithm with 100 decision trees
model = BaggingClassifier(base_estimator=dtc, n_estimators=100, random_state=seed)

# Fit a single model
results_dtc = dtc.fit(X, Y)
# Fit a bagged model (100 Decision Trees)
results_model = model.fit(X, Y)

# Validate the peformance of both using 10-fold Cross Validation.
results_kfold_dtc = model_selection.cross_val_score(dtc, X, Y, cv=10)
results_kfold_model = model_selection.cross_val_score(model, X, Y, cv=10)

print('\t\tSingle model accuracy\t\tBagged model accuracy')
print("10-fold CV\t{:2.2f}%\t\t\t\t{:2.2f}%".format(results_kfold_dtc.mean()*100, results_kfold_model.mean()*100))
