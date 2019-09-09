'''
This script creates the graspyclust data for Figure 4 in the pyclust paper
'''

import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("..")
from brute_cluster_graspyclust import brute_graspy_cluster

base = 40
factor = 2
num_sets = 13
output_file = 'graspyclust_times_full.csv'

ns = base*np.power(factor,np.arange(num_sets)) 
ts = np.zeros(ns.shape)
results = pd.DataFrame(columns=['N','Time'])

#*************Options***********
ks = [i for i in range(2,6)]
covariance_types=['full']
Ns=[50]
savefigs = None
graph_types= [] #['true', 'all_bics', 'best_ari', 'best_bic', 'ari_vs_bic']

for i,n in enumerate(ns):
    file = ".\data\\" + str(n) + ".csv"
    x = np.genfromtxt(file, delimiter=',',skip_header=0)
    x = x[:,np.arange(1,x.shape[1])]
    c_true = np.genfromtxt(file, delimiter=',', usecols = (0),skip_header=0)

    start_time = time.time()
    c_hat,_,_,_,_ = brute_graspy_cluster(Ns, x, covariance_types, ks, 
                            c_true, savefigs, graph_types)
    entry = {'N':n, 'Time':time.time() - start_time}
    results = results.append(entry, ignore_index=True)
    
    print(entry)
results.to_csv(path_or_buf =output_file)





