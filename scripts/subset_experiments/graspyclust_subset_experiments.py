#%%
#Synthetic
from graspologic.cluster import GaussianCluster
import numpy as np
import sys
sys.path.append("..")
from brute_cluster_graspyclust import brute_graspy_cluster
import time
import pandas as pd
import matplotlib.pyplot as plt
import csv

#None - no figures will be saved, string - files will be saved with that name
savefigs = None
graph_types= [] #['true', 'all_bics', 'best_ari', 'best_bic', 'ari_vs_bic']
num_runs=10
Ns = [50]

ks = [i for i in range(1,21)]
covariance_types=['full','tied','diag','spherical']

x = np.genfromtxt('../../data/synthetic.csv', delimiter=',',skip_header=0)
x = x[:,np.arange(1,x.shape[1])]
c_true = np.genfromtxt('../../data/synthetic.csv', delimiter=',', usecols = (0),skip_header=0)
idxs_full = pd.read_csv('idxs_synthetic.csv')
output_file = 'graspyclust_synthetic.csv'

results = pd.DataFrame(columns=['ARI','Time'])
x_full = x
c_true_full = c_true
n_full = x.shape[0]

for i in range(num_runs):
    print('Run number: ' + str(i))
    idxs = idxs_full.iloc[:,i].values - 1
    x = x_full[idxs,]
    c_true = c_true_full[idxs,]

    start_time = time.time()
    c_hat,_,_,best_ari_bic,_ = brute_graspy_cluster(Ns, x, covariance_types, ks, 
                            c_true, savefigs, graph_types)
    results = results.append({'Time':time.time() - start_time,'ARI':best_ari_bic}, ignore_index=True)
results.to_csv(path_or_buf =output_file)


#%%
#Wisconsin Breast Cancer Diagnostic Data

#None - no figures will be saved, string - files will be saved with that name
savefigs = None
graph_types= [] #['true', 'all_bics', 'best_ari', 'best_bic', 'ari_vs_bic']
num_runs = 10

#Wisconsin Diagnostic Data
ks = [i for i in range(1,21)]
covariance_types=['full','tied','diag','spherical']

#read mean texture, extreme area, and extreme smoothness
x = np.genfromtxt('../../data/wdbc.data',delimiter=',', usecols = (3,25,26),skip_header=0)
with open('../../data/wdbc.data') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    c_true = []
    for row in reader:
        c_true.append(row[1])
        
c_true = np.asarray([int(c == 'M') for c in c_true])
idxs_full = pd.read_csv('idxs_bc.csv')
output_file = 'graspyclust_bc.csv'

results = pd.DataFrame(columns=['ARI','Time'])
x_full = x
c_true_full = c_true
n_full = x.shape[0]

for i in range(num_runs):
    print('Run number: ' + str(i))
    idxs = idxs_full.iloc[:,i].values - 1
    x = x_full[idxs,]
    c_true = c_true_full[idxs,]

    start_time = time.time()
    c_hat,_,_,best_ari_bic,_ = brute_graspy_cluster(Ns, x, covariance_types, ks, 
                            c_true, savefigs, graph_types)
    results = results.append({'Time':time.time() - start_time,'ARI':best_ari_bic}, ignore_index=True)
results.to_csv(path_or_buf =output_file)

#%%
#Drosophila

#None - no figures will be saved, string - files will be saved with that name
savefigs = None
graph_types= [] #['true', 'all_bics', 'best_ari', 'best_bic', 'ari_vs_bic']
num_runs = 10

#Wisconsin Diagnostic Data
ks = [i for i in range(1,21)]
covariance_types=['full','tied','diag','spherical']
x = np.genfromtxt('../../data/embedded_right.csv',delimiter=',',skip_header=1)
c_true = np.genfromtxt('../../data/classes.csv',skip_header=1)
idxs_full = pd.read_csv('idxs_drosophila.csv')
output_file = 'graspyclust_drosophila.csv'

results = pd.DataFrame(columns=['ARI','Time'])
x_full = x
c_true_full = c_true
n_full = x.shape[0]

for i in range(num_runs):
    print('Run number: ' + str(i))
    idxs = idxs_full.iloc[:,i].values - 1
    x = x_full[idxs,]
    c_true = c_true_full[idxs,]

    start_time = time.time()
    c_hat,_,_,best_ari_bic,_ = brute_graspy_cluster(Ns, x, covariance_types, ks, 
                            c_true, savefigs, graph_types)
    results = results.append({'Time':time.time() - start_time,'ARI':best_ari_bic}, ignore_index=True)
results.to_csv(path_or_buf =output_file)
