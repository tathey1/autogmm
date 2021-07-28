#This script runs the ARI and time experiments for autogmm
#%%
#Synthetic
import numpy as np
import sys
sys.path.append("..")
import csv
import time
from sklearn.datasets import load_iris
from sklearn import datasets
import pandas as pd
from graspologic.cluster.autogmm import AutoGMMCluster

path = '/code/scripts/subset_experiments/'
num_runs = 10

ks = [i for i in range(1,21)]
affinities = 'all'
linkages = 'all'
covariance_types='all'

x = np.genfromtxt('../../../data/synthetic.csv', delimiter=',',skip_header=0)
x = x[:,np.arange(1,x.shape[1])]
c_true = np.genfromtxt('../../../data/synthetic.csv', delimiter=',', usecols = (0),skip_header=0)
idxs_full = pd.read_csv(path + 'idxs_synthetic.csv')
output_file = '/results/autogmm_synthetic.csv'

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
    pyc = AutoGMMCluster(min_components=ks[0],max_components=ks[len(ks)-1],
        affinity=affinities,linkage=linkages,covariance_type=covariance_types,random_state=0)
    pyc.fit(x,c_true)
    best_ari_bic = pyc.ari_
    results = results.append({'Time':time.time() - start_time,'ARI':best_ari_bic}, ignore_index=True)
results.to_csv(path_or_buf =output_file)

#%%
#Wisconsin Breast Cancer Diagnostic Data

num_runs = 10

ks = [i for i in range(1,21)]
affinities = 'all'
linkages = 'all'
covariance_types='all'

#read mean texture, extreme area, and extreme smoothness
x = np.genfromtxt('../../../data/wdbc.data',delimiter=',', usecols = (3,25,26),skip_header=0)
with open('../../../data/wdbc.data') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    c_true = []
    for row in reader:
        c_true.append(row[1])
        
c_true = np.asarray([int(c == 'M') for c in c_true])
idxs_full = pd.read_csv(path + 'idxs_bc.csv')
output_file = '/results/autogmm_bc.csv'

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
    pyc = AutoGMMCluster(min_components=ks[0],max_components=ks[len(ks)-1],
        affinity=affinities,linkage=linkages,covariance_type=covariance_types,random_state=0)
    pyc.fit(x,c_true)
    best_ari_bic = pyc.ari_
    results = results.append({'Time':time.time() - start_time,'ARI':best_ari_bic}, ignore_index=True)
results.to_csv(path_or_buf =output_file)

#%%
#Drosophila

num_runs = 10

ks = [i for i in range(1,21)]
affinities = 'all'
linkages = 'all'
covariance_types='all'

x = np.genfromtxt('../../../data/embedded_right.csv',delimiter=',',skip_header=1)
c_true = np.genfromtxt('../../../data/classes.csv',skip_header=1)
idxs_full = pd.read_csv(path + 'idxs_drosophila.csv')
output_file = '/results/autogmm_drosophila.csv'

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
    pyc = AutoGMMCluster(min_components=ks[0],max_components=ks[len(ks)-1],
        affinity=affinities,linkage=linkages,covariance_type=covariance_types,random_state=0)
    pyc.fit(x,c_true)
    best_ari_bic = pyc.ari_
    results = results.append({'Time':time.time() - start_time,'ARI':best_ari_bic}, ignore_index=True)
results.to_csv(path_or_buf =output_file)