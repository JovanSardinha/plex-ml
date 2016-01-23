#!/usr/bin/python

import pymssql
import datetime

now = datetime.datetime.now()

cnxn = pymssql.connect("plexdb.database.windows.net","plex-admin@plexdb","OxygnClub123","plexdb")
cursor = cnxn.cursor()

queryStr = "INSERT INTO user_profile (userId, blockStartTime, hardAccelerationCount, hardBrakingCount) OUTPUT INSERTED.userId VALUES ('%s','%s', %s, %s)" % ('jovan', now.strftime('%Y-%m-%d %H:%M:%S'), 1, 2)
# print queryStr

cursor.execute(queryStr)
# rows = cursor.fetchall()
# for row in rows:
# 	print str(row[0])

cnxn.commit()
