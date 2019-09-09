import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from graspy.cluster.autogmm import AutoGMMCluster

affinities = ['none','manhattan','euclidean','cosine']
linkages = ['ward','complete','average','single']
covariance_types = ['full','tied','diag','spherical']
ks = [i for i in range(2,6)]

base = 40
factor = 2
num_sets = 14 #indicates the maximally sized dataset (16 in the paper)
output_file = 'autogmm_option_times.csv'

ns = base*np.power(factor,np.arange(num_sets))
ts = np.zeros(ns.shape)
results = pd.DataFrame(columns=['N','Affinity','Linkage','Covariance_Type','Time'])


for affinity in affinities:
    for linkage in linkages:
        if linkage == 'ward' and affinity != 'euclidean':
            continue
        if affinity == 'none' and linkage != 'complete':
            continue
        for covariance_type in covariance_types:
            for i,n in enumerate(ns):
                file = ".\data\\" + str(n) + ".csv"
                x = np.genfromtxt(file, delimiter=',',skip_header=0)
                x = x[:,np.arange(1,x.shape[1])]
                c_true = np.genfromtxt(file, delimiter=',', usecols = (0),skip_header=0)

                start_time = time.time()
                pyc = AutoGMMCluster(min_components=ks[0],max_components=ks[len(ks)-1],
                    affinity=affinity,linkage=linkage,covariance_type=covariance_type)
                pyc.fit(x,c_true)
                entry = {'N':n,'Affinity':affinity,'Linkage':linkage,'Covariance_Type':covariance_type, 'Time':time.time() - start_time};
                results = results.append(entry, ignore_index=True)
                
                print(entry)
results.to_csv(path_or_buf =output_file)