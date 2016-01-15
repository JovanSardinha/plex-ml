from sklearn_pandas import *
import pandas as pd
import numpy as np
from sklearn import *

'''
data = pd.DataFrame({'pet':['cat', 'dog', 'dog', 'fish', 'cat', 'dog', 'cat', 'fish'],
                     'children': [4., 6, 3, 3, 2, 3, 5, 4],
                     'salary':[90, 24, 44, 27, 32, 59, 36, 27]})

mapper = DataFrameMapper([('pet', sklearn.preprocessing.LabelBinarizer()),(['children'],
                                                                           sklearn.preprocessing.StandardScaler())])
np.round(mapper.fit_transform(data.copy()), 2)
sample = pd.DataFrame({'pet': ['cat'], 'children': [5.]})
temp = np.round(mapper.transform(sample), 2)
temp2 =   np.round(mapper.fit_transform(sample), 2)

pipe = sklearn.pipeline.Pipeline([ ('featurize', mapper), ('lm', sklearn.linear_model.LinearRegression())])
jovan = np.round(cross_val_score(pipe, data.copy(), data.salary, 'r2'), 2)
'''
'''
clf = linear_model.Ridge (alpha = .5)
clf.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1])

from sklearn.datasets import load_svmlight_file
X_train, y_train = load_svmlight_file("/Users/JovanSardinha/Downloads/a1a.t")
'''

import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import pickle
from sklearn.externals import joblib
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=0)


clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
prediction = clf.predict(X_test)
confusion = metrics.confusion_matrix(y_test, prediction)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(iris.target_names))
    plt.xticks(tick_marks, iris.target_names, rotation=45)
    plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


