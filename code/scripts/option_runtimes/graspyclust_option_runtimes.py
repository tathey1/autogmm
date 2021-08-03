import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("/code/scripts")
from brute_cluster_graspyclust import brute_graspy_cluster

path = '/code/scripts/option_runtimes/data/'

covariance_types = ['full','tied','diag','spherical']
Ns=[50]
savefigs = None
graph_types= []
ks = [i for i in range(2,6)]

base = 40
factor = 2
num_sets = 14 #indicates the maximally sized dataset
output_file = '/results/graspyclust_option_times.csv'

ns = base*np.power(factor,np.arange(num_sets))
ts = np.zeros(ns.shape)
results = pd.DataFrame(columns=['N','Covariance_Type','Time'])



for covariance_type in covariance_types:
    for i,n in enumerate(ns):
        file = path + str(n) + ".csv"
        x = np.genfromtxt(file, delimiter=',',skip_header=0)
        x = x[:,np.arange(1,x.shape[1])]
        c_true = np.genfromtxt(file, delimiter=',', usecols = (0),skip_header=0)

        start_time = time.time()
        c_hat,_,_,_,_ = brute_graspy_cluster(Ns, x, covariance_type, ks, 
                            c_true, savefigs, graph_types)
        entry = {'N':n,'Covariance_Type':covariance_type, 'Time':time.time() - start_time};
        results = results.append(entry, ignore_index=True)
        
        # print(entry)
results.to_csv(path_or_buf =output_file)