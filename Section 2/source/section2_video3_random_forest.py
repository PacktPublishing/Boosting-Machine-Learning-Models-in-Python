"""
Compare a single Decision Tree to
Random Forest with 100 trees.
"""
from section2_video1_data import get_data

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

seed=123

# Load prepared data
X, Y = get_data()
# Load our single model
dtc = DecisionTreeClassifier(random_state=seed)
# Create 100 decision trees in a forest
# Note: no need to provide a single model
#       it already used DecisionTreeClassifier
# When you don't provide max_features the default value
# is sqrt(number_of_input_values) which a good starting point
# for classification.
# In our case we've got a slight boost in performance with max_features='log2'.
model = RandomForestClassifier(n_estimators=100, max_features='log2', random_state=seed)

# Fit a single model
results_dtc = dtc.fit(X, Y)
# Fit random forest (100 Decision Trees)
results_model = model.fit(X, Y)

# Validate the peformance of both using 10-fold Cross Validation.
results_kfold_dtc = model_selection.cross_val_score(dtc, X, Y, cv=10)
results_kfold_model = model_selection.cross_val_score(model, X, Y, cv=10)

print('\t\tSingle model accuracy\t\tRandom Forest model accuracy')
print("10-fold CV\t{:2.2f}%\t\t\t\t{:2.2f}%".format(results_kfold_dtc.mean()*100, results_kfold_model.mean()*100))
