
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



client = MongoClient('40.122.215.160')
db = client['test']
collection_acceleration = db['linearAcceleration']
df_acceleration = pd.DataFrame(list(collection_acceleration.find()))


# Step 2: Data pre-processing
time_converter = lambda x: datetime.datetime.utcfromtimestamp(float(x)/1000) if float(x) > 1000000000000 \
    else datetime.datetime.utcfromtimestamp(float(x))

df_acceleration['timestamp'] = df_acceleration['timestamp'].map(time_converter)

terek = df_acceleration[df_acceleration['userId'] == 'jan_26_2016_tj']

import json
from pprint import pprint
json_string = open('2016-01-26-21-15-27.json').read()
json_string = '{"one" : "1", "two" : "2", "three" : "3"}'

json_data = open('2016-01-26-21-15-27.json')
stringOfJsonData = json.dumps(json_data)
data = json.load(json_data)


data = pd.DataFrame(list(json.load(json_data)))


df_acceleration = pd.DataFrame(list(collection_acceleration.find()))

print ("done")

