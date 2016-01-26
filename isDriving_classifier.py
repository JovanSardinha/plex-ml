# PURPOSE: This scripts takes all the data from the mongo DB and put a predicted driving value.
# GOAL: Add two new values to each document in mongo, namely: [isDriving_predicted] and [lastModified] timestamp.

# Libraries used.
from pymongo import MongoClient
import pandas as pd
import time
import math
import numpy as np
from sklearn.externals import joblib
import datetime

# Used to time the total script run time.
# Notes: Average (historic) run times of the script = ~2 seconds
time_start = time.time()
###############################

# Step 1: Reading from Mongo DB
# Notes: mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
client = MongoClient('40.122.215.160')
db = client['archive[2016-01-25@14:28ET]_test']
collection_acceleration = db['linearAcceleration']

# Putting data into a pandas data frame for easy manipulation
df_acceleration = pd.DataFrame(list(collection_acceleration.find()))

# Step 2: Data pre-processing
time_converter = lambda x: datetime.datetime.utcfromtimestamp(float(x)/1000) if float(x) > 1000000000000 \
    else datetime.datetime.utcfromtimestamp(float(x))

df_acceleration['timestamp'] = df_acceleration['timestamp'].map(time_converter)
df_acceleration['lastModified'] = df_acceleration['lastModified'].map(time_converter)

# Function to calculate force magnitude
def acceleration_magnitude(x,y,z):
    return math.sqrt(x**2 + y**2 + z**2)

df_acceleration['acceleration_magnitude'] = np.vectorize(acceleration_magnitude)(df_acceleration['x'],
                                                                                 df_acceleration['y'],
                                                                                 df_acceleration['z'])

df_total = df_acceleration[['isDriving', 'x', 'y', 'z', 'acceleration_magnitude']].copy()
df_total['isDriving'] = df_total['isDriving'].map({'true': 1, 'false': 0}).astype(int)
data_total = df_total.values

# Step 3: Loading the model from disk
model = joblib.load('./model/plex_model.pkl')

# Step 4: Predicting
output = model.predict(data_total[:,1:])

# Step 5: Combining with data fame and augmenting data frame with additional values
df_acceleration['isDriving_predicted'] = output
df_acceleration['isDriving_predicted'] = df_acceleration['isDriving_predicted'].map({0: 'false', 1: 'true'})
df_acceleration['lastModified'] = time.time() # Note: this time stamp is in Epoch time to match other time stamps

# Step 6: Updating mongo to reflect changes
i = 0
for i in range(0, df_acceleration.shape[0]):
    result = collection_acceleration.update_one({
        '_id': df_acceleration['_id'][i]},
            {
                '$set': {'isDriving_predicted': df_acceleration['isDriving_predicted'][i],
                         'lastModified':df_acceleration['lastModified'][i]}
            }, upsert = False)

##################################
# End of script: The following is used to calculate total run times
time_end = time.time()
time_total = time_end - time_start
print("--- Run time is: %s seconds ---" % (time_total))