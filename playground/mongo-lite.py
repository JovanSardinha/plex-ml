from mongolite import *
import datetime

connection = Connection('40.122.215.160')

@connection.register
class SnapShot(Document):
    __database__ = 'test'
    __collection__ = 'snapShot'
    skeleton = {
         'id':unicode,
         'class':unicode,
         'accelerometer':unicode
    }
    optional = {
         'links': [unicode],
    }
    default_values = {}

snapShot = connection.SnapShot()


blogpost = connection.test.example.find_one()