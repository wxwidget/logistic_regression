from sklearn import datasets
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

def load_data():
    x_test = pd.read_csv('../data/test.csv',header=None)
    y = pd.read_csv('../data/trainLabels.csv',header=None)
    x = pd.read_csv('../data/train.csv',header=None)
    return x_test, x, y

if __name__ == '__main__':
    x_test, x, y = load_data()
    logit = sm.Logit(y,x)
    result = logit.fit()
    #logit.predict(x_test)
    #print result.summary()
    #print result.conf_int()
    #print result.params
    #logistic = lambda s: 1.0 / (1.0 + np.exp(-s))
    p = result.predict(x)
    t = y[0].as_matrix()
    print classification_report(t, [i>0.5 and 1 or 0 for i in p])
    pass
