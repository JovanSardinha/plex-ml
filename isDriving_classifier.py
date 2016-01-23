from pymongo import MongoClient
import pandas as pd
import datetime as dt
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

time_start = time.time()
###############################

# Working example with pymongo
#mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
client = MongoClient('40.122.215.160')
db = client['test']

collection_acceleration = db['linearAcceleration']

df_acceleration = pd.DataFrame(list(collection_acceleration.find()))

# Data Processing
time_converter = lambda x: dt.datetime.fromtimestamp(float(x)/1000)

df_acceleration['timestamp'] = df_acceleration['timestamp'].map(time_converter)


def acceleration_magnitude(x,y,z):
    return math.sqrt(x**2 + y**2 + z**2)

df_acceleration['acceleration_magnitude'] = np.vectorize(acceleration_magnitude)(df_acceleration['x'],df_acceleration['y'],df_acceleration['z'])

df_total = df_acceleration[['isDriving', 'x', 'y', 'z', 'acceleration_magnitude']].copy()
df_total['isDriving'] = df_total['isDriving'].map({'true': 1, 'false': 0}).astype(int)
data_total = df_total.values

# Loading the model
model = joblib.load('./model/plex_model.pkl')

# Predicting
output = model.predict(data_total[:,1:])

# Combining with data fame
df_acceleration['predicted'] = output
##################################
time_end = time.time()
time_total = time_end - time_start
print("--- Run time is: %s seconds ---" % (time_total))