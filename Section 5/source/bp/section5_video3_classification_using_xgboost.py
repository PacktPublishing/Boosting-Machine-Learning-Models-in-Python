"""
Simple classification with XGBoost.
"""
from section1_video5_data import get_data

from sklearn import model_selection
from xgboost import XGBClassifier
import numpy as np

seed=123

# Load prepared data
X, Y = get_data('../data/video1_diabetes.csv')
# Build our single model
c = XGBClassifier(random_state=seed)

results = c.fit(X, Y)

# using 10-fold Cross Validation.
results_kfold_model = model_selection.cross_val_score(c, X, Y, cv=10)

print("XGBoost accuracy:\t{:2.2f}%".format(results_kfold_model.mean()*100))

# Use trained model on a new dataset
# 'Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'
new_data=np.array([[0,148,72,0,0,33.6,0,50]])
print("New dataset:{}".format(new_data))
# Predict the most probably class that the new_data belongs to
p=c.predict(new_data)
print("Has diabieties?:\t{}".format("yes" if p[0] == 1 else "no"))

# Predict probability for new_data to belong to each class
pred=c.predict_proba(new_data)
print("Probability of having diabities:\t{:2.2f}%".format(pred[0][1]*100))
print("Probability of NOT having diabities:\t{:2.2f}%".format(pred[0][0]*100))
