"""
Measuring features importance and feature selection.

Note:
Include %matplotlib notebook in the first line of
your jupyter notebook to see the charts.
"""
from section1_video5_data import get_data

from sklearn import model_selection
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier, plot_importance
from matplotlib import pyplot

seed=123

# Load prepared data
X, Y = get_data('../data/video1_diabetes.csv')
# Build our single model
c = XGBClassifier(random_state=seed, n_estimators=60, max_depth=3)

c.fit(X, Y)
results_kfold_model = model_selection.cross_val_score(c, X, Y, cv=10)
print("XGBoost accuracy:\t{}:{:2.2f}%".format("all features", results_kfold_model.mean()*100))

plot_importance(c, importance_type='gain')
pyplot.show()

# Sort features by least important first
fi=c.feature_importances_
fi=list(zip(range(len(fi)), list(fi)))
fi.sort(key=lambda x: x[1])

print('Starting from including the least important features first')
for i, f in fi:
	selected_model = SelectFromModel(c, threshold=f, prefit=True)
	selected_X = selected_model.transform(X)
	selected_c = XGBClassifier(random_state=seed, n_estimators=60, max_depth=3)
	selected_c.fit(selected_X, Y)
	# using 10-fold Cross Validation.
	results_kfold_model = model_selection.cross_val_score(selected_c, selected_X, Y, cv=10)
	print("f{} {:2.2f}:{:2.2f}%".format(i, f, results_kfold_model.mean()*100))
