#!/usr/bin/python
from pymongo import MongoClient
import pandas as pd
import datetime
import time
import math
import numpy as np
import pymssql
import pdb

# Acceleration due to gravity (m/s^2)
g = 9.80665

# (m/s^2)
hard_acceleration_threshold = 3.33333

# Jan 1st, 1970
epoch = datetime.datetime.utcfromtimestamp(0)

time_start = time.time()
###############################

# Working example with pymongo
#mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
client = MongoClient('40.122.215.160')
db = client['archive[2016-01-25@14:28ET]_test']

collection_acceleration = db.linearAcceleration

df_acceleration = pd.DataFrame(list(collection_acceleration.find()))

# Acceleration's from iOS are reported in multiples of g
df_acceleration.loc[df_acceleration.deviceType == 'iOS', "x"] *= g
df_acceleration.loc[df_acceleration.deviceType == 'iOS', "y"] *= g
df_acceleration.loc[df_acceleration.deviceType == 'iOS', "z"] *= g

def acceleration_magnitude(x,y,z):
    return math.sqrt(x**2 + y**2 + z**2)

def toHourTimestamp(ts):
    tsf = float(ts)
    if tsf > 1000000000000:
        dt = datetime.datetime.utcfromtimestamp(tsf/1000)
    else:
        dt = datetime.datetime.utcfromtimestamp(tsf)
    dt = dt.replace(minute=0, second=0, microsecond=0)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def key(userId, timestamp):
    return userId + toHourTimestamp(timestamp)

sqlValueStr = lambda v: "('%s','%s',%s,%s)" % (v["userId"].replace("'","''").encode('ascii', 'ignore'), v["blockStartTime"], v["hardAccelerationCount"], v["hardBrakingCount"])

df_acceleration.acceleration = np.vectorize(acceleration_magnitude)(df_acceleration.x,df_acceleration.y,df_acceleration.z)

userIds = pd.unique(df_acceleration.userId)

inserts = {}

for userId in userIds:
    rows = df_acceleration[df_acceleration.userId == userId].sort_values('timestamp')
    for index, row in rows.iterrows():
        k = key(userId, row.timestamp)
        if k not in inserts:
            inserts[k] = {"userId": userId, "blockStartTime": toHourTimestamp(row.timestamp), "hardAccelerationCount": 0, "hardBrakingCount": 0}
        inserts[k]["hardAccelerationCount"] += 1

queryStr = "INSERT INTO user_profile (userId, blockStartTime, hardAccelerationCount, hardBrakingCount) OUTPUT Inserted.userId VALUES "
queryStr += ','.join(map(sqlValueStr, inserts.values()))

# print queryStr
try:
    cnxn = pymssql.connect("plexdb.database.windows.net","plex-admin@plexdb","OxygnClub123","plexdb")
    cursor = cnxn.cursor()

    cursor.execute(queryStr)
    cnxn.commit()
    print str(len(inserts)) + " rows inserted into user_profile."
except pymssql.IntegrityError as e:
    print "Violation of PRIMARY KEY condition. Cannot insert duplicate data into user_profile table."

##################################
time_end = time.time()
time_total = time_end - time_start
print("--- Run time is: %s seconds ---" % (time_total))
