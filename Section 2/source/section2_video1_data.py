"""
Load, cut and return the data.

You can learn more about the dataset and download it here:
https://www.kaggle.com/uciml/pima-indians-diabetes-database
"""
import pandas
from pprint import pprint

def get_data():
    # Load data from .csv file
    dataframe = pandas.read_csv("data/video1_diabetes.csv")
    # Cut it into the same number of rows
    # for each type of outcome
    xc0=dataframe[dataframe['Outcome']==0].iloc[0:268,:]
    xc1=dataframe[dataframe['Outcome']==1].iloc[0:268,:]
    # Merge it back together
    dataframe=pandas.concat([xc0,xc1])
    # Choose the dataset's columns as input values
    X = dataframe.loc[:, ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
    # the just one column as a output values
    Y = dataframe.loc[:, 'Outcome']
    return X.to_numpy(), Y.to_numpy()

if __name__ == '__main__':
    X, Y=get_data()
    print('X (input values - medical predictors)')
    pprint(X[0:5])
    print('Y (output values: 0 - has no diabietes, 1 - has diabetes)')
    pprint(Y[0:5])
