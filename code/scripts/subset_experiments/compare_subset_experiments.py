
'''
This script plots the ARI and Runtime values obtained from graspyclust_experiments.py, autogmm_experiments.py, and mclust_experiments.r
It saves the figures as subset_abc.png and subset_def.png
'''
#%%
import numpy as np
from scipy.stats import mode
from scipy.stats import wilcoxon
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

#%%
print('Reading data...')
path = '/code/scripts/subset_experiments/paper_results/'
#read the data
mclust_s = pd.read_csv(path + "mclust_synthetic.csv")
mclust_s = mclust_s.loc[:,['ARI','Time']]
mclust_s['Dataset'] = mclust_s.shape[0]*['Synthetic']
mclust_s['Algorithm'] = mclust_s.shape[0]*['mclust']

mclust_bc = pd.read_csv(path + "mclust_bc.csv")
mclust_bc = mclust_bc.loc[:,['ARI','Time']]
mclust_bc['Dataset'] = mclust_bc.shape[0]*['Breast Cancer']
mclust_bc['Algorithm'] = mclust_bc.shape[0]*['mclust']

mclust_dro = pd.read_csv(path + "mclust_drosophila.csv")
mclust_dro = mclust_dro.loc[:,['ARI','Time']]
mclust_dro['Dataset'] = mclust_dro.shape[0]*['Drosophila']
mclust_dro['Algorithm'] = mclust_dro.shape[0]*['mclust']

autogmm_s = pd.read_csv(path + "autogmm_synthetic.csv")
autogmm_s = autogmm_s.loc[:,['ARI','Time']]
autogmm_s['Dataset'] = autogmm_s.shape[0]*['Synthetic']
autogmm_s['Algorithm'] = autogmm_s.shape[0]*['AutoGMM']

autogmm_bc = pd.read_csv(path + "autogmm_bc.csv")
autogmm_bc = autogmm_bc.loc[:,['ARI','Time']]
autogmm_bc['Dataset'] = autogmm_bc.shape[0]*['Breast Cancer']
autogmm_bc['Algorithm'] = autogmm_bc.shape[0]*['AutoGMM']

autogmm_dro = pd.read_csv(path + "autogmm_drosophila.csv")
autogmm_dro = autogmm_dro.loc[:,['ARI','Time']]
autogmm_dro['Dataset'] = autogmm_dro.shape[0]*['Drosophila']
autogmm_dro['Algorithm'] = autogmm_dro.shape[0]*['AutoGMM']

graspyclust_s = pd.read_csv(path + "graspyclust_synthetic.csv")
graspyclust_s = graspyclust_s.loc[:,['ARI','Time']]
graspyclust_s['Dataset'] = graspyclust_s.shape[0]*['Synthetic']
graspyclust_s['Algorithm'] = graspyclust_s.shape[0]*['graspyclust']

graspyclust_bc = pd.read_csv(path + "graspyclust_bc.csv")
graspyclust_bc = graspyclust_bc.loc[:,['ARI','Time']]
graspyclust_bc['Dataset'] = graspyclust_bc.shape[0]*['Breast Cancer']
graspyclust_bc['Algorithm'] = graspyclust_bc.shape[0]*['graspyclust']

graspyclust_dro = pd.read_csv(path + "graspyclust_drosophila.csv")
graspyclust_dro = graspyclust_dro.loc[:,['ARI','Time']]
graspyclust_dro['Dataset'] = graspyclust_dro.shape[0]*['Drosophila']
graspyclust_dro['Algorithm'] = graspyclust_dro.shape[0]*['graspyclust']

data = pd.concat([autogmm_s,mclust_s,graspyclust_s,
    autogmm_bc,mclust_bc,graspyclust_bc,
    autogmm_dro,mclust_dro,graspyclust_dro],axis=0)

sns.set(style='whitegrid')

print('Significant Differences:')

#ARI plot
g_ari = sns.catplot(x='Algorithm',y='ARI',col='Dataset',data=data,kind='swarm',
    height=4.5, aspect=0.7)

plt.sca(g_ari.axes[0][0])
plt.title('a) Synthetic',fontweight='bold',fontsize=14)
plt.xlabel('')
plt.ylabel('ARI',fontweight='bold',fontsize=14)
plt.xticks(fontsize=13)

#synthetic - AutoGMM vs mclust
_,p = wilcoxon(autogmm_s['ARI'].values,mclust_s['ARI'].values)
if p < 0.05:
    print('Synthetic ARI - mclust vs AutoGMM')
    warnings.warn('This result is not in the paper')

#synthetic, mclust vs graspyclust
_,p = wilcoxon(mclust_s['ARI'].values,graspyclust_s['ARI'].values)
if p < 0.05:
    print('Synthetic ARI - mclust vs graspyclust')
    print(p)
    txt = "*p=%.3f"%p
    x1, x2 = 1,2
    y, h, col = 1.1,0.05,'k'
    plt.plot([x1, x1, x2, x2], [y-0.5*h, y, y, y-h-0.5*h], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h-0.5*h, txt, ha='center', va='bottom', color=col)
    plt.ylim((0,1.2))

#synthetic, autogmm vs graspyclust
_,p = wilcoxon(autogmm_s['ARI'].values,graspyclust_s['ARI'].values)
if p < 0.05:
    print('Synthetic ARI - AutoGMM vs graspyclust')
    warnings.warn('This result is not in the paper')

plt.sca(g_ari.axes[0][1])
plt.title('b) Breast Cancer',fontweight='bold',fontsize=14)
plt.xlabel('Algorithm',fontweight='bold',fontsize=14)
plt.xticks(fontsize=13)

#bc - autogmm vs mclust
_,p = wilcoxon(autogmm_bc['ARI'].values,mclust_bc['ARI'].values)
if p < 0.05:
    print('Breast Cancer ARI - mclust vs AutoGMM')
    warnings.warn('This result is not in the paper')

#bc - mclust vs graspyclust
_,p = wilcoxon(mclust_bc['ARI'].values,graspyclust_bc['ARI'].values)
if p < 0.05:
    print('Breast Cancer ARI - mclust vs graspyclust')
    warnings.warn('This result is not in the paper')

#bc - autogmm vs graspyclust
_,p = wilcoxon(autogmm_bc['ARI'].values,graspyclust_bc['ARI'].values)
if p < 0.05:
    print('Breast Cancer ARI - graspyclust vs AutoGMM')
    warnings.warn('This result is not in the paper')

plt.sca(g_ari.axes[0][2])
plt.title('c) Drosophila',fontweight='bold',fontsize=14)
plt.xlabel('')
plt.xticks(fontsize=13)

#drosophila - autogmm vs mclust
_,p = wilcoxon(autogmm_dro['ARI'].values,mclust_dro['ARI'].values)
if p < 0.05:
    print('Drosophila ARI - mclust vs AutoGMM')
    warnings.warn('This result is not in the paper')

#drosophila, mclust vs graspyclust
_,p = wilcoxon(mclust_dro['ARI'].values,graspyclust_dro['ARI'].values)
if p < 0.05:
    print('Drosophila ARI - mclust vs graspyclust')
    warnings.warn('This result is not in the paper')

#drosophila - autogmm vs graspyclust
_,p = wilcoxon(autogmm_dro['ARI'].values,graspyclust_dro['ARI'].values)
if p < 0.05:
    print('Drosophila ARI - graspyclust vs AutoGMM')
    warnings.warn('This result is not in the paper')

plt.yticks([0.2*i for i in range(1,6)],fontsize=14)
#plt.figtext(0.99, 0.01, '* p<0.05 for Wilcoxon signed-rank test', horizontalalignment='right')
plt.subplots_adjust(top=0.85)
plt.suptitle('Clustering ARIs on Different Datasets',fontweight='bold',fontsize=18)

print('Saving ARI figure')
plt.savefig('/results/subset_abc.png')

#Time plot
g_time = sns.catplot(x='Algorithm',y='Time',col='Dataset',data=data,kind='swarm',
    height=4.5, aspect=0.7)

plt.sca(g_time.axes[0][0])
plt.ylabel('Time (s)')
plt.title('d) Synthetic',fontweight='bold',fontsize=14)
plt.xlabel('')
plt.ylabel('Time (s)',fontweight='bold',fontsize=14)
plt.xticks(fontsize=13)

h_txt = 550
top = 600
bottom_high = 500
bottom_low = 300
col='k'


#synthetic - autogmm vs mclust
_,p = wilcoxon(mclust_s['Time'].values,autogmm_s['Time'].values)
if p < 0.05:
    print('Synthetic time - AutoGMM vs mclust')
    print(p)
    txt = "*p=%.3f"%p
    txt="*"
    x1, x2 = 0,1
    plt.plot([x1, x1, x2, x2], [bottom_high, top, top, bottom_low], lw=1.5, c=col)
    plt.text((x1+x2)*.5, h_txt, txt, ha='center', va='bottom', color=col)

#synthetic - graspyclust vs mclust
_,p = wilcoxon(mclust_s['Time'].values,graspyclust_s['Time'].values)
if p < 0.05:
    print('Synthetic time - mclust vs graspyclust')
    print(p)
    txt = "*p=%.3f"%p
    txt="*"
    x1, x2 = 1,2
    plt.plot([x1, x1, x2, x2], [bottom_low, top, top, bottom_high], lw=1.5, c=col)
    plt.text((x1+x2)*.5, h_txt, txt, ha='center', va='bottom', color=col)

#synthetic - graspyclust vs AutoGMM
_,p = wilcoxon(autogmm_s['Time'].values,graspyclust_s['Time'].values)
if p < 0.05:
    print('Synthetic time - AutoGMM vs graspyclust')
    print(p)
    txt = "*p=%.3f"%p
    x1, x2 = 0,2
    plt.plot([x1, x1, x2, x2], [bottom_high*3, top*3, top*3, bottom_low*3], lw=1.5, c=col)
    plt.text((x1+x2)*.5, h_txt*4, txt, ha='center', va='bottom', color=col)

plt.sca(g_time.axes[0][1])
plt.title('e) Breast Cancer',fontweight='bold',fontsize=14)
plt.xlabel('Algorithm',fontweight='bold',fontsize=14)
plt.xticks(fontsize=13)

#bc - autogmm vs mclust
_,p = wilcoxon(mclust_bc['Time'].values,autogmm_bc['Time'].values)
if p < 0.05:
    print('Breast cancer time - AutoGMM vs mclust')
    print(p)
    txt = "*p=%.3f"%p
    txt="*"
    x1, x2 = 0,1
    plt.plot([x1, x1, x2, x2], [bottom_high, top, top, bottom_low], lw=1.5, c=col)
    plt.text((x1+x2)*.5, h_txt, txt, ha='center', va='bottom', color=col)

#bc - graspyclust vs mclust
_,p = wilcoxon(mclust_bc['Time'].values,graspyclust_bc['Time'].values)
if p < 0.05:
    print('Breast cancer time - graspyclust vs mclust')
    print(p)
    txt = "*p=%.3f"%p
    txt="*"
    x1, x2 = 1,2
    plt.plot([x1, x1, x2, x2], [bottom_low, top, top, bottom_high], lw=1.5, c=col)
    plt.text((x1+x2)*.5, h_txt, txt, ha='center', va='bottom', color=col)

#bc - autogmm vs graspyclust
_,p = wilcoxon(autogmm_bc['Time'].values,graspyclust_bc['Time'].values)
if p < 0.05:
    print('Breast cancer time - graspyclust vs AutoGMM')
    print(p)
    txt = "*p=%.3f"%p
    txt="*"
    x1, x2 = 0,2
    plt.plot([x1, x1, x2, x2], [bottom_high*3, top*3, top*3, bottom_low*3], lw=1.5, c=col)
    plt.text((x1+x2)*.5, h_txt*3, txt, ha='center', va='bottom', color=col)

plt.sca(g_time.axes[0][2])
plt.title('f) Drosophila',fontweight='bold',fontsize=14)
plt.xlabel('')
plt.xticks(fontsize=13)

#drosophila - autogmm vs mclust
_,p = wilcoxon(mclust_dro['Time'].values,autogmm_dro['Time'].values)
if p < 0.05:
    print('Drosophila time - AutoGMM vs mclust')
    print(p)
    txt = "*p=%.3f"%p
    txt="*"
    x1, x2 = 0,1
    plt.plot([x1, x1, x2, x2], [bottom_high, top, top, bottom_low], lw=1.5, c=col)
    plt.text((x1+x2)*.5, h_txt, txt, ha='center', va='bottom', color=col)

#drosophila - graspyclust vs mclust
_,p = wilcoxon(mclust_dro['Time'].values,graspyclust_dro['Time'].values)
if p < 0.05:
    print('Drosophila time - graspyclust vs mclust')
    print(p)
    txt = "*p=%.3f"%p
    txt="*"
    x1, x2 = 1,2
    plt.plot([x1, x1, x2, x2], [bottom_low,top, top, bottom_high], lw=1.5, c=col)
    plt.text((x1+x2)*.5, h_txt, txt, ha='center', va='bottom', color=col)

#drosophila - autogmm vs graspyclust
_,p = wilcoxon(autogmm_dro['Time'].values,graspyclust_dro['Time'].values)
if p < 0.05:
    print('Drosophila time - graspyclust vs AutoGMM')
    print(p)
    txt = "*p=%.3f"%p
    txt="*"
    x1, x2 = 0,2
    plt.plot([x1, x1, x2, x2], [bottom_high*3, top*3, top*3, bottom_low*3], lw=1.5, c=col)
    plt.text((x1+x2)*.5, h_txt*3, txt, ha='center', va='bottom', color=col)

plt.ylim(0.1,5000)
#plt.yticks([50*i for i in range(0,8)])
#plt.figtext(0.99, 0.01, '* p<0.05 for Wilcoxon signed-rank test', horizontalalignment='right')
plt.subplots_adjust(top=0.85)
plt.yscale('log')
plt.suptitle('Clustering Runtimes on Different Datasets',fontweight='bold',fontsize=18)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

print('Saving Runtime figure')
plt.savefig('/results/subset_def.png')
#%%
