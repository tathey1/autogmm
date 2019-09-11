'''
This script shows the results of autogmm on the three datasets
First, it shows the bicplot, which is included in the paper as Figure 3 for the Drosophila dataset
Then, it outputs information about the chosen model, which is in Table 2 in the paper
Lastly, it shows the true clustering and autogmm clustering which are in the Appendix of the paper
'''

#%%
import numpy as np
import sys
sys.path.append("..")
import csv
import time
from sklearn.datasets import load_iris
from sklearn import datasets
import pandas as pd
from graspy.cluster.autogmm import AutoGMMCluster
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
random.seed(0)



#Change this field for different datasets**************************
dataset = 2 #0-synthetic, 1-BC, 2-drosophila
#*********************************************************************
print('Running AutoGMM on dataset #' + str(dataset))

if dataset==0:
    ks = [i for i in range(1,21)]
    affinities = 'all'
    linkages = 'all'
    covariance_types='all'
    
    x = np.genfromtxt('../data/synthetic.csv', delimiter=',',skip_header=0)
    x = x[:,np.arange(1,x.shape[1])]
    c_true = np.genfromtxt('../data/synthetic.csv', delimiter=',', usecols = (0),skip_header=0)
elif dataset==1:
    #Wisconsin Diagnostic Data
    ks = [i for i in range(1,21)]
    affinities = 'all'
    linkages = 'all'
    covariance_types='all'
    
    #read mean texture, extreme area, and extreme smoothness
    x = np.genfromtxt('../data/wdbc.data',delimiter=',', usecols = (3,25,26),skip_header=0)
    with open('../data/wdbc.data') as csvfile:
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
    x = np.genfromtxt('../data/embedded_right.csv',delimiter=',',skip_header=1)
    c_true = np.genfromtxt('../data/classes.csv',skip_header=1)



def make_bic_plots(results,best_cov,best_k_bic,best_bic):
    #plot of all BICS*******************************
    titles = ['Full','Tied','Diagonal','Spherical']
    bics = np.zeros((44,20))
    cov_types = ['full','tied','diag','spherical']
    for i,cov_type in enumerate(cov_types):
        bics[i*11+0,:] = -results.loc[(results['covariance_type'] == cov_type) &
            (results['affinity'] == 'none')]['bic'].values
        bics[i*11+1,:] = -results.loc[(results['covariance_type'] == cov_type) &
            (results['affinity'] == 'euclidean') & (results['linkage'] == 'ward')]['bic'].values
        bics[i*11+2,:] = -results.loc[(results['covariance_type'] == cov_type) &
            (results['affinity'] == 'euclidean') & (results['linkage'] == 'complete')]['bic'].values
        bics[i*11+3,:] = -results.loc[(results['covariance_type'] == cov_type) &
            (results['affinity'] == 'euclidean') & (results['linkage'] == 'average')]['bic'].values
        bics[i*11+4,:] = -results.loc[(results['covariance_type'] == cov_type) &
            (results['affinity'] == 'euclidean') & (results['linkage'] == 'single')]['bic'].values
        bics[i*11+5,:] = -results.loc[(results['covariance_type'] == cov_type) &
            (results['affinity'] == 'manhattan') & (results['linkage'] == 'complete')]['bic'].values
        bics[i*11+6,:] = -results.loc[(results['covariance_type'] == cov_type) &
            (results['affinity'] == 'manhattan') & (results['linkage'] == 'average')]['bic'].values
        bics[i*11+7,:] = -results.loc[(results['covariance_type'] == cov_type) &
            (results['affinity'] == 'manhattan') & (results['linkage'] == 'single')]['bic'].values
        bics[i*11+8,:] = -results.loc[(results['covariance_type'] == cov_type) &
            (results['affinity'] == 'cosine') & (results['linkage'] == 'complete')]['bic'].values
        bics[i*11+9,:] = -results.loc[(results['covariance_type'] == cov_type) &
            (results['affinity'] == 'cosine') & (results['linkage'] == 'average')]['bic'].values
        bics[i*11+10,:] = -results.loc[(results['covariance_type'] == cov_type) &
            (results['affinity'] == 'cosine') & (results['linkage'] == 'single')]['bic'].values


    labels = {0:'none',1:'l2/ward',2:'l2/complete',3:'l2/average',4:'l2/single',
            5:'l1/complete',6:'l1/average',7:'l1/single',8:'cos/complete',
            9:'cos/average',10:'cos/single'}
        
    fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2,sharey='row',sharex='col',figsize=(7,6))
    for row in np.arange(bics.shape[0]):
        if all(bics[row,:]==-np.inf):
            continue
        if row<=10:
            ax0.plot(np.arange(1,len(ks)+1),bics[row,:])
        elif row<=21:
            ax1.plot(np.arange(1,len(ks)+1),bics[row,:],label=labels[row%11])
        elif row<=32:
            ax2.plot(np.arange(1,len(ks)+1),bics[row,:])
        elif row<=43:
            ax3.plot(np.arange(1,len(ks)+1),bics[row,:])
    
    #plot line indicating chosen model
    if best_cov == 'full':
        ylims = ax0.get_ylim()
        ax0.plot([best_k_bic,best_k_bic],[ylims[0],best_bic],color='black',linestyle='dashed',linewidth=2)
    elif best_cov == 'tied':
        ylims = ax1.get_ylim()
        ax1.plot([best_k_bic,best_k_bic],[ylims[0],best_bic],color='black',linestyle='dashed',linewidth=2)
    elif best_cov == 'diag':
        ylims = ax2.get_ylim()
        ax2.plot([best_k_bic,best_k_bic],[ylims[0],best_bic],color='black',linestyle='dashed',linewidth=2)
    elif best_cov == 'spherical':
        ylims = ax3.get_ylim()
        ax3.plot([best_k_bic,best_k_bic],[ylims[0],best_bic],color='black',linestyle='dashed',linewidth=2)
    
    #fig.suptitle('a) autogmm',fontsize=20,fontweight='bold')
    fig.text(0.5, 0.04, 'Number of Components', ha='center',fontsize=18,fontweight='bold')
    fig.text(0.01, 0.5, 'BIC', va='center', rotation='vertical',fontsize=18,fontweight='bold')
    
    ax0.set_title(titles[0],fontsize=22,fontweight='bold')
    ax0.locator_params(axis='y',tight=True,nbins=4)
    ax0.set_yticklabels(ax0.get_yticks(),fontsize=18)

    ax1.set_title(titles[1],fontsize=22,fontweight='bold')
    legend = ax1.legend(loc='best',title='Agglomeration\nMethod',fontsize=12)
    plt.setp(legend.get_title(),fontsize=14)

    ax2.set_title(titles[2],fontsize=22,fontweight='bold')
    ax2.set_xticks(range(0,21,4))
    ax2.set_xticklabels(ax2.get_xticks(),fontsize=18)
    ax2.locator_params(axis='y',tight=True,nbins=4)
    ax2.set_yticklabels(ax2.get_yticks(),fontsize=18)
    

    ax3.set_title(titles[3],fontsize=22,fontweight='bold')
    ax3.set_xticks(range(0,21,4))
    ax3.set_xticklabels(ax3.get_xticks(),fontsize=18)
    fname = './autogmm_bicplot_dataset'+ str(dataset) + '.png'
    plt.savefig(fname)

def make_cluster_plots(x,c_true,c_hat_autogmm):
    c_list = ['red', 'green', 'blue','orange','purple','yellow','gray']

    plt.figure(figsize=(8,8))
    max_c = int(np.max(c_true))
    plt.scatter(x[:,0],x[:,1],c=c_true,cmap=colors.ListedColormap(c_list[0:max_c+1]))
    plt.xlabel('First Dimension',fontsize=24)
    plt.ylabel('Second Dimension',fontsize=24)
    plt.title('True Clustering',fontsize=24,fontweight='bold')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    fname = './true_clustering_dataset'+ str(dataset) + '.png'
    plt.savefig(fname)
    
    plt.figure(figsize=(8,8))
    max_c = int(np.max(c_hat_autogmm))
    plt.scatter(x[:,0],x[:,1],c=c_hat_autogmm,cmap=colors.ListedColormap(c_list[0:max_c+1]))
    plt.xlabel('First Dimension',fontsize=24)
    plt.ylabel('Second Dimension',fontsize=24)
    plt.title('AutoGMM Clustering',fontsize=24,fontweight='bold')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    fname = './autogmm_clustering_dataset'+ str(dataset) + '.png'
    plt.savefig(fname)


pyc = AutoGMMCluster(min_components=ks[0],max_components=ks[len(ks)-1],
    affinity=affinities,linkage=linkages,covariance_type=covariance_types,
    random_state=0)
c_hat_autogmm,ari = pyc.fit_predict(x,c_true)
#np.savetxt('autogmm.csv',labels, delimiter=',')

combo = [pyc.affinity_,pyc.linkage_,pyc.covariance_type_]
k = pyc.n_components_
reg = pyc.reg_covar_
bic = -pyc.bic_
results = pyc.results_

print('Info for table:')
print('Best model: ' + str(combo))
print('Best reg: ' + str(reg))
print('Best k: ' + str(k))
print('Best BIC: ' + str(bic))
print('Best ARI: ' + str(ari))

#%%
print('Making BIC Plots...')
make_bic_plots(results,combo[2],k,bic)

print('Making Clustering Plots...')
make_cluster_plots(x,c_true,c_hat_autogmm)

#%%