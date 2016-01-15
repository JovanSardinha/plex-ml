import csv
import datetime
import numpy as np
import pandas as pd
from ggplot import *
import matplotlib.pyplot as plt
from sklearn_pandas import DataFrameMapper, cross_val_score


link = "/Users/JovanSardinha/Dropbox/Data/dataset_web/participant_1.csv"

reader = None
with open(link, 'rb') as f:
    reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONE)
    lst = list(reader)

df = pd.DataFrame(lst[1:], columns=lst[0])


changer = lambda x: x[:-4] + "." + x[9:]
converter = lambda x: pd.to_datetime(x, format="%H:%M:%S.%f")

df['Time_Biotrace'] = df['Time_Biotrace'].apply(changer)
df['Time_Videorating'] = df['Time_Videorating'].apply(changer)
df['Time_Light'] = df['Time_Light'].apply(changer)
df['Time_Accel'] = df['Time_Accel'].apply(changer)
df['Time_GPS'] = df['Time_GPS'].apply(changer)

df['Time_Biotrace'] = df['Time_Biotrace'].apply(converter)
df['Time_Videorating'] = df['Time_Videorating'].apply(converter)
df['Time_Light'] = df['Time_Light'].apply(converter)
df['Time_Accel'] = df['Time_Accel'].apply(converter)
df['Time_GPS'] = df['Time_GPS'].apply(converter)

df = df.convert_objects(convert_numeric=True)
#nf = df.as_matrix()

#plt.plot(df['Time_Videorating'], df['AccelX'], 'ro')
#plt.show()
