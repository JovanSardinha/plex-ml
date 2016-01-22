#!/usr/bin/python

import pymssql
cnxn = pymssql.connect("plexdb.database.windows.net","plex-admin@plexdb","OxygnClub123","plexdb")
cursor = cnxn.cursor()
cursor.execute("""
select * from information_schema.tables
""")

rows = cursor.fetchall()
for row in rows:
	print row
