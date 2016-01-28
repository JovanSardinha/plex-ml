#!/usr/bin/python
from optparse import OptionParser
import pandas as pd
import numpy as np
import json

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename", help="data file",
                  action="store", type="string")

(options, args) = parser.parse_args()
arr = []
with open(options.filename) as f:
    for line in f:
        try:
            lineJson = json.loads(line)
            arr.append({'name': lineJson['name'], 'value': lineJson['value'], 'timestamp': lineJson['timestamp']})
        except ValueError:
            print "Error processing line: " + line
            continue

df = pd.DataFrame(arr)
print df
