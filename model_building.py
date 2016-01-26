# PURPOSE: This scripts is used for model building and traning.
# GOAL: To produce a model and output it in the ./model folder.

# Libraries used
from pymongo import MongoClient
import pandas as pd
import datetime
import time
import math
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
import sklearn.metrics as skm
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.dates as md
from sklearn.learning_curve import learning_curve
from sklearn.externals import joblib

# Used to time the total script run time.
# Notes: Average (historic) run times of the script = ~2 seconds
time_start = time.time()
###############################

# Step 1: Reading data from mongo
#mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db) # For future refrence incase we add username and password to DB
client = MongoClient('40.122.215.160')
db = client['archive[2016-01-13@13:08ET]_test']

# Reading data from collections
collection_acceleration = db['acceleration']
#collection_gyroscope = db['gyroscope']
#collection_linearAcceleration = db['linearAcceleration']
#collection_magnetic = db['magnetic']
#collection_orientation = db['orientation']
#collection_rotation = db['rotation']

# Importing collection data into pandas data frame for easy manipulation
df_acceleration = pd.DataFrame(list(collection_acceleration.find()))
#df_gyroscope = pd.DataFrame(list(collection_gyroscope.find()))
#df_linearAcceleration = pd.DataFrame(list(collection_linearAcceleration.find()))
#df_magnetic =` pd.DataFrame(list(collection_magnetic.find()))
#df_orientation = pd.DataFrame(list(collection_orientation.find()))
#df_rotation = pd.DataFrame(list(collection_rotation.find()))

# Step 2: Data pre-processing
time_converter = lambda x: datetime.datetime.fromtimestamp(float(x)/1000)

df_acceleration['timestamp'] = df_acceleration['timestamp'].map(time_converter)
#df_gyroscope['timestamp'] = df_gyroscope['timestamp'].map(time_converter)
#df_linearAcceleration['timestamp'] = df_linearAcceleration['timestamp'].map(time_converter)
#df_magnetic['timestamp'] = df_magnetic['timestamp'].map(time_converter)
#df_orientation['timestamp'] = df_orientation['timestamp'].map(time_converter)
#df_rotation['timestamp'] = df_rotation['timestamp'].map(time_converter)

def acceleration_magnitude(x,y,z):
    return math.sqrt(x**2 + y**2 + z**2)

df_acceleration['acceleration_magnitude'] = np.vectorize(acceleration_magnitude)(df_acceleration['x'],df_acceleration['y'],df_acceleration['z'])


# Step 3: Data Splitting
# Notes: Using the data split as advised by Andrew Ng. ->  20% Test, 20% CV 60% Train
df_total = df_acceleration[['isDriving', 'x', 'y', 'z', 'acceleration_magnitude']].copy()
df_total['isDriving'] = df_total['isDriving'].map({'true': 1, 'false': 0}).astype(int)
data_total = df_total.values
data_train, data_test = cross_validation.train_test_split(data_total, test_size=0.20, random_state=13)

# Step 4: Model Building
model = SVC()
model = model.fit(data_train[:,1:], data_train[:,0])

# Step 5: Predicting
output = model.predict(data_test[:,1:])

# Step 6: Saving the model
joblib.dump(model, './model/plex_model.pkl')

# Step 7: Post-Analysis
# Parameters:
# Base Parameters
accuracy = skm.accuracy_score(data_test[:,0], output)
precision = skm.precision_score(data_test[:,0], output)
recall = skm.recall_score(data_test[:,0], output)

# Parameters - Scores
average_precision = skm.average_precision_score(data_test[:,0], output)
f1 = skm.f1_score(data_test[:,0], output)
roc_auc = skm.roc_auc_score(data_test[:,0], output)

# Parameters - Losses
log_loss = skm.log_loss(data_test[:,0], output)



# Training Curves

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 50)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

cv = cross_validation.ShuffleSplit(data_train.shape[0], n_iter=6,test_size=0.2, random_state=0)
train_sizes, train_scores, valid_scores = learning_curve(estimator=model, X=data_train[:,1:], y=data_train[:,0], cv=cv)

plot_learning_curve(model, "Plex.ai Test", X=data_train[:,1:], y=data_train[:,0], cv=cv )

##################################
# End of script: The following is used to calculate total run times
time_end = time.time()
time_total = time_end - time_start
print("--- Run time is: %s seconds ---" % (time_total))