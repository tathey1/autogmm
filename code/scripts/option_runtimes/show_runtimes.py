'''
This script reproduces figure 6 in the paper (fig6.png)
'''

#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

path = '/code/scripts/option_runtimes/paper/'

print('Reading AutoGMM results...')
#Now plot the different autogmm options
df = pd.read_csv(path + "autogmm_option_times.csv")
df = df.loc[:,['N','Affinity','Linkage','Covariance_Type','Time']]


first = True
f = plt.figure(figsize=(8,6))
kwargs = {}
kwargs['linewidth'] = 0.5
kwargs['alpha'] = 1
kwargs['linestyle'] = ':'

kwargs_reg = {}
kwargs_reg['linewidth'] = 3


kwargs['color'] = 'lightcoral'


#Plot regression
plot_start = 1000
start = 20480
end = 2e6
step = 1000
df_end = df[df['N'] >= start]
model = LinearRegression().fit(np.log(df_end.N.values.reshape([-1,1])), np.log(df_end.Time.values))
x = np.arange(plot_start,end,step)
y = np.exp(model.intercept_)*np.power(x,model.coef_)
pyslope = model.coef_[0]
plt.plot(x,y,color='red',label='AutoGMM slope: %1.2f' %pyslope,**kwargs_reg)

#Plot lines
for affinity in df.Affinity.unique():
    for linkage in df.Linkage.unique():
        if linkage == 'ward' and affinity != 'euclidean':
            continue
        if affinity == 'none' and linkage != 'complete':
            continue
        for covariance_type in df.Covariance_Type.unique():
            N = df.loc[(df['Affinity'] == affinity) & (df['Linkage'] == linkage) &
                (df['Covariance_Type'] == covariance_type)]['N']
            T = df.loc[(df['Affinity'] == affinity) & (df['Linkage'] == linkage) &
                (df['Covariance_Type'] == covariance_type)]['Time']
            lbl = affinity + '/' + linkage + '/' + covariance_type

            plt.plot(N,T,**kwargs)


print('Reading mclust results...')
df = pd.read_csv(path + "mclust_option_times.csv")
df = df.loc[:,['N','Model','Time']]

#Plot regression
start = 81920
df_end = df[df['N'] >= start]
model = LinearRegression().fit(np.log(df_end.N.values.reshape([-1,1])), np.log(df_end.Time.values))
x = np.arange(plot_start,end,step)
y = np.exp(model.intercept_)*np.power(x,model.coef_)
mslope = model.coef_[0]
plt.plot(x,y,color='blue',label='mclust slope: %1.2f'%mslope,**kwargs_reg)

#Plot lines
kwargs['color'] = 'cornflowerblue'

for model in df.Model.unique():
    N = df.loc[df['Model'] == model]['N']
    T = df.loc[df['Model'] == model]['Time']
    lbl = model

    plt.plot(N,T,**kwargs)

#graspyclust*****************************************

print('Reading graspyclust results...')
df = pd.read_csv(path + "graspyclust_option_times.csv")
df = df.loc[:,['N','Covariance_Type','Time']]

#Plot regression
start = 5120
df_end = df[df['N'] >= start]
model = LinearRegression().fit(np.log(df_end.N.values.reshape([-1,1])), np.log(df_end.Time.values))
x = np.arange(plot_start,end,step)
y = np.exp(model.intercept_)*np.power(x,model.coef_)
graspyslope = model.coef_[0]
plt.plot(x,y,color='green',label = 'GraSPyclust slope: %1.2f'%graspyslope,**kwargs_reg)

#Plot lines
kwargs['color'] = 'chartreuse'

for covariance_type in df.Covariance_Type.unique():
    N = df.loc[df['Covariance_Type'] == covariance_type]['N']
    T = df.loc[df['Covariance_Type'] == covariance_type]['Time']
    lbl = covariance_type

    plt.plot(N,T,**kwargs)


plt.xscale('log')
plt.yscale('log')
plt.xlabel('# of samples (n)',fontsize=24,fontweight='bold')
plt.ylabel('Time (s)',fontsize=24,fontweight='bold')
plt.title('Runtimes of All Clustering Options \n on Different Sized Datasets',
fontsize=24,fontweight='bold')
plt.xticks([1e2, 1e4, 1e6],fontsize=20)
plt.yticks([1e-3, 1, 1e3],fontsize=20)
plt.legend(prop={'size': 20})
fname = '/results/option_runtimes.png'
plt.savefig(fname)
#%%

