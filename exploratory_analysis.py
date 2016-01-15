from pymongo import MongoClient
import pandas as pd
import datetime as dt
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md

time_start = time.time()
###############################


# Working example with pymongo
#mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
client = MongoClient('40.122.215.160')
db = client['archive[2016-01-13@13:08ET]_test']

collection_acceleration = db['acceleration']
collection_gyroscope = db['gyroscope']
collection_linearAcceleration = db['linearAcceleration']
collection_magnetic = db['magnetic']
collection_orientation = db['orientation']
collection_rotation = db['rotation']


df_acceleration = pd.DataFrame(list(collection_acceleration.find()))
#df_gyroscope = pd.DataFrame(list(collection_gyroscope.find()))
#df_linearAcceleration = pd.DataFrame(list(collection_linearAcceleration.find()))
#df_magnetic = pd.DataFrame(list(collection_magnetic.find()))
#df_orientation = pd.DataFrame(list(collection_orientation.find()))
#df_rotation = pd.DataFrame(list(collection_rotation.find()))


time_converter = lambda x: dt.datetime.fromtimestamp(float(x)/1000)

df_acceleration['timestamp'] = df_acceleration['timestamp'].map(time_converter)
#df_gyroscope['timestamp'] = df_gyroscope['timestamp'].map(time_converter)
#df_linearAcceleration['timestamp'] = df_linearAcceleration['timestamp'].map(time_converter)
#df_magnetic['timestamp'] = df_magnetic['timestamp'].map(time_converter)
#df_orientation['timestamp'] = df_orientation['timestamp'].map(time_converter)
#df_rotation['timestamp'] = df_rotation['timestamp'].map(time_converter)

def acceleration_magnitude(x,y,z):
    return math.sqrt(x**2 + y**2 + z**2)

df_acceleration['acceleration_magnitude'] = np.vectorize(acceleration_magnitude)(df_acceleration['x'],df_acceleration['y'],df_acceleration['z'])

#df_acceleration['fft'] = np.fft.fft(df_acceleration['acceleration_magnitude'])

tdf = df_acceleration['acceleration_magnitude'].copy()
data = tdf
fft_size = 250
overlap_fac = 0.5




# playing with STFT (https://kevinsprojects.wordpress.com/2014/12/13/short-time-fourier-transform-using-python-and-numpy/)
hop_size = np.int32(np.floor(fft_size * (1-overlap_fac)))
pad_end_size = fft_size          # the last segment can overlap the end of the data array by no more than one window size
total_segments = np.int32(np.ceil(len(data) / np.float32(hop_size)))

window = np.hanning(fft_size)  # our half cosine window
inner_pad = np.zeros(fft_size) # the zeros which will be used to double each segment size

proc = np.concatenate((data, np.zeros(pad_end_size)))              # the data to process
result = np.empty((total_segments, fft_size), dtype=np.float32)    # space to hold the result

for i in xrange(total_segments):                      # for each segment
    current_hop = hop_size * i                        # figure out the current segment offset
    segment = proc[current_hop:current_hop+fft_size]  # get the current segment
    windowed = segment * window                       # multiply by the half cosine function
    padded = np.append(windowed, inner_pad)           # add 0s to double the length of the data
    spectrum = np.fft.fft(padded) / fft_size          # take the Fourier Transform and scale by the number of samples
    autopower = np.abs(spectrum * np.conj(spectrum))  # find the autopower spectrum
    result[i, :] = autopower[:fft_size]               # append to the results array

result = 20*np.log10(result)          # scale to db
result = np.clip(result, -40, 200)    # clip values

img = plt.imshow(result, origin='lower', cmap='jet', interpolation='nearest', aspect='auto')
plt.show()



##################################
time_end = time.time()
time_total = time_end - time_start
print("--- Run time is: %s seconds ---" % (time_total))




#Graveyard
'''
# Plotting
import matplotlib.pyplot as plt
import matplotlib.dates as md

#ax=plt.gca()
#xfmt = md.DateFormatter('%H:%M:%S')
#ax.xaxis.set_major_formatter(xfmt)
plt.plot(fft_tdf, 'g^')
plt.show()


signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
count = np.array([1,2,3,4,5,6,7,8])
df = pd.DataFrame(signal, count)
df['temp'] =
fourier = np.fft.fft(signal)
n = signal.size
timestep = 0.1
freq = np.fft.fftfreq(n, d=timestep)
'''