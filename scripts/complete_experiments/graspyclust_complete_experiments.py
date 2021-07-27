'''
This script reproduces the graspyclust clusterings in Figures 1-3 of the paper (graspyclust_clustering_dataset*.png)
Additionally, it outputs the clustering option that was selected, reproducing information in Table 2 of the paper.
'''

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
import matplotlib.pyplot as plt
import matplotlib.colors as colors


#%%
#Change this field for different datasets**************************
dataset = 2 #0-synthetic, 1-BC, 2-drosophila
#*********************************************************************

print('Running graspyclust on dataset #' + str(dataset))
if dataset==0:
    ks = [i for i in range(1,21)]
    affinities = 'all'
    linkages = 'all'
    covariance_types='all'
    
    x = np.genfromtxt('../../data/synthetic.csv', delimiter=',',skip_header=0)
    x = x[:,np.arange(1,x.shape[1])]
    c_true = np.genfromtxt('../../data/synthetic.csv', delimiter=',', usecols = (0),skip_header=0)
elif dataset==1:
    #Wisconsin Diagnostic Data
    ks = [i for i in range(1,21)]
    affinities = 'all'
    linkages = 'all'
    covariance_types='all'
    
    #read mean texture, extreme area, and extreme smoothness
    x = np.genfromtxt('../../data/wdbc.data',delimiter=',', usecols = (3,25,26),skip_header=0)
    with open('../../data/wdbc.data') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        c_true = []
        for row in reader:
            c_true.append(row[1])   
    c_true = np.asarray([int(c == 'M') for c in c_true])
elif dataset == 2:
    #Drosophila
    ks = [i for i in range(1,21)]
    affinities = 'all'
    linkages = 'all'
    covariance_types='all'
    x = np.genfromtxt('../../data/embedded_right.csv',delimiter=',',skip_header=1)
    c_true = np.genfromtxt('../../data/classes.csv',skip_header=1)


def make_cluster_plots(x,c_hat_graspy):
    c_list = ['red', 'green', 'blue','orange','purple','yellow','gray']

    plt.figure(figsize=(8,8))
    max_c = int(np.max(c_hat_graspy))
    plt.scatter(x[:,0],x[:,1],c=c_hat_graspy,cmap=colors.ListedColormap(c_list[0:max_c+1]))
    plt.xlabel('First Dimension',fontsize=24)
    plt.ylabel('Second Dimension',fontsize=24)
    plt.title('GraspyClust Clustering',fontsize=24,fontweight='bold')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    fname = './graspyclust_clustering_dataset' + str(dataset) + '.png'
    plt.savefig(fname)

c_hat_graspy,best_cov_bic, best_k_bic, best_ari_bic,best_bic = brute_graspy_cluster(Ns=[50], x=x,
    covariance_types=['full','tied','diag','spherical'], ks=ks,c_true=c_true)

print('Info for Table:')
print('Best model: ' + best_cov_bic)
print('Best k: ' + str(best_k_bic))
print('Best BIC: ' + str(best_bic))
print('Best ARI: ' + str(best_ari_bic))
print('Making Clustering Plots...')
make_cluster_plots(x,c_hat_graspy)
