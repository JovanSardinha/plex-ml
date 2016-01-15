from sklearn import datasets
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy import stats

lr = linear_model.LinearRegression()
boston = datasets.load_boston()

print boston.data.shape