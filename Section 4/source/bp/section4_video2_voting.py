"""
Voting classifier example, by default it's set up for
majority/hard voting mode.
"""
from section1_video5_data import get_data

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

seed=123

# Load prepared data
X, Y = get_data('../data/video1_diabetes.csv')
# Build our single model
c1 = RandomForestClassifier(n_estimators=10, random_state=seed)
c2 = ExtraTreesClassifier(n_estimators=10, random_state=seed)
# Build an ensemble model
c3 = AdaBoostClassifier(n_estimators=10, random_state=seed)

model = VotingClassifier([('c1', c1), ('c2', c2), ('c3', c3)])
# Fit a single model
results_c=[]
models_c=[c1, c2, c3]
for c in models_c:
    results = c.fit(X, Y)
    results_kfold = model_selection.cross_val_score(c, X, Y, cv=10)
    results_c.append(results_kfold)

# Fit an ensemble model
results_model = model.fit(X, Y)

# Validate the peformance of a ensemble model
# using 10-fold Cross Validation.
results_kfold_model = model_selection.cross_val_score(model, X, Y, cv=10)

for i, rc in enumerate(results_c):
    print('{:s}\t{:2.2f}%'.format(models_c[i].__class__.__name__, rc.mean()*100))
print("{:s}\t{:2.2f}%".format(model.__class__.__name__, results_kfold_model.mean()*100))
