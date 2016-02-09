# Libraries used
from pymongo import MongoClient
import pandas as pd
from datetime import datetime
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

##################################
# End of script: The following is used to calculate total run times
time_end = time.time()
time_total = time_end - time_start
print("--- Run time is: %s seconds ---" % (time_total))