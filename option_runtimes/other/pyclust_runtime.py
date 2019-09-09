'''
This script creates the pyclust data for Figure 4 in the pyclust paper
'''

import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from graspy.cluster.autogmm  import AutoGMMCluster

base = 40
factor = 2
num_sets = 18
output_file = 'pyclust_times_ward_new.csv'

ns = base*np.power(factor,np.arange(num_sets))
ts = np.zeros(ns.shape)
results = pd.DataFrame(columns=['N','Time'])

ks = [i for i in range(2,6)]
affinities = 'euclidean'
linkages = 'ward'
covariance_types='full'

for i,n in enumerate(ns):
    file = ".\data\\" + str(n) + ".csv"
    x = np.genfromtxt(file, delimiter=',',skip_header=0)
    x = x[:,np.arange(1,x.shape[1])]
    c_true = np.genfromtxt(file, delimiter=',', usecols = (0),skip_header=0)

    start_time = time.time()
    pyc = PyclustCluster(min_components=ks[0],max_components=ks[len(ks)-1],
        affinity=affinities,linkage=linkages,covariance_type=covariance_types)
    pyc.fit(x,c_true)
    entry = {'N':n, 'Time':time.time() - start_time}
    results = results.append(entry, ignore_index=True)
    
    print(entry)
results.to_csv(path_or_buf =output_file)





