from pymongo import MongoClient
import pandas as pd
import datetime as dt
import time

time_start = time.time()
###############################


# Working example with pymongo
#mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
client = MongoClient('40.122.215.160')
db = client['test']

collection_gyroscope = db['gyroscope']
collection_linearAcceleration = db['linearAcceleration']
collection_location = db['location']
collection_magnetic = db['magnetic']
collection_orientation = db['orientation']
collection_rotation = db['rotation']


df_gyroscope = pd.DataFrame(list(collection_gyroscope.find()))
df_linearAcceleration = pd.DataFrame(list(collection_linearAcceleration.find()))
df_location = pd.DataFrame(list(collection_location.find()))
df_magnetic = pd.DataFrame(list(collection_magnetic.find()))
df_orientation = pd.DataFrame(list(collection_orientation.find()))
df_rotation = pd.DataFrame(list(collection_rotation.find()))


time_converter = lambda x: dt.datetime.fromtimestamp(float(x)/1000)

df_gyroscope['timestamp'] = df_gyroscope['timestamp'].map(time_converter)
df_linearAcceleration['timestamp'] = df_linearAcceleration['timestamp'].map(time_converter)
df_location['timestamp'] = df_location['timestamp'].map(time_converter)
df_magnetic['timestamp'] = df_magnetic['timestamp'].map(time_converter)
df_orientation['timestamp'] = df_orientation['timestamp'].map(time_converter)
df_rotation['timestamp'] = df_rotation['timestamp'].map(time_converter)

##################################
time_end = time.time()
time_total = time_end - time_start
print("--- Run time is: %s seconds ---" % (time_total))


# df_acceleration[df_acceleration['userId'] =='tester-jovan'] # COMMAND