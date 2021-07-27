'''
This saves figures showing the results of various clustering algorithms,
including AutoGMM, agglomerative clustering, k-means, naive GM (using single
or mutiple inits), on the double-cigar dataset.
It outputs the two subplots of Figure 7 in the paper.
'''

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from graspologic.cluster import AutoGMMCluster
from joblib import Parallel, delayed
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

sns.set_context('talk')

#%%
def AGM(X, y, n_clusters_range, n_init):
    agmm = AutoGMMCluster(
        max_components=n_clusters_range[-1], min_components=n_clusters_range[0], kmeans_n_init=n_init, init_params='kmeans'
    )
    pred = agmm.fit_predict(X, y)
    return agmm.model_, agmm.ari_, pred

def KM(X, y, n_clusters_range, n_init):
    ari_km = -1
    for n_clus in np.unique(n_clusters_range):
        kmeans = KMeans(n_clusters=n_clus, n_init=n_init)
        pred_km = kmeans.fit_predict(X)
        ari_ = adjusted_rand_score(pred_km, y)
        if ari_ > ari_km:
            ari_km = ari_
            best_model = kmeans
    return best_model, ari_km, best_model.predict(X)

def Agg(X, y, n_clusters_range):
    af = ['euclidean', 'manhattan', 'cosine']
    li = ['ward', 'complete', 'single', 'average']
    ari_ag = -1
    for af_ in af:
        for li_ in li:
            if li_ == 'ward' and af_ != 'euclidean':
                continue
            else:
                for n_clus in np.unique(n_clusters_range):
                    agg = AgglomerativeClustering(
                        n_clusters=n_clus, affinity=af_, linkage=li_
                    )
                    pred_ag = agg.fit_predict(X)
                    ari_ = adjusted_rand_score(pred_ag, y)
                    if ari_ > ari_ag:
                        ari_ag = ari_
                        best_model = agg
                        best_pred = pred_ag
    return best_model, ari_ag, best_pred

def naive_GMM(X, y, n_components_range, n_init, init_params='kmeans'):
    lowest_bic = np.infty
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in np.unique(n_components_range):
            gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type,
                                  n_init=n_init, init_params=init_params)
            gmm.fit(X)
            bic_ = gmm.bic(X)
            if bic_ < lowest_bic:
                lowest_bic = bic_
                best_gmm = gmm

    pred = best_gmm.predict(X)
    return best_gmm, adjusted_rand_score(pred, y), pred

def exp(X, y, n_clusters_range, n_init=10):
    agm, ari_agm, pred_agm = AGM(X, y, n_clusters_range, n_init)
    gm, ari_gm, pred_gm = naive_GMM(X, y, n_clusters_range, 1)
    gm_m, ari_gm_multi, pred_gm_multi = naive_GMM(X, y, n_clusters_range, n_init)
    km, ari_km, pred_km = KM(X, y, n_clusters_range, n_init)
    ag, ari_ag, pred_ag = Agg(X, y, n_clusters_range)
    
    aris = [ari_agm, ari_gm, ari_gm_multi, ari_km, ari_ag]
    preds = [pred_agm, pred_gm, pred_gm_multi, pred_km, pred_ag]
    models = [agm, gm, gm_m, km, ag]
    return aris, preds, models


#%%
# plot predictions on single cigar dataset by various clustering algorithms
np.random.seed(32)
cov_1 = 1
cov_2 = 200
n = 100
mu_ = 3
mu1 = [-mu_, 0]
mu2 = [mu_, 0]

cov1 = np.array([[cov_1, 0], [0, cov_2]])
cov2 = np.array([[cov_1, 0], [0, cov_2]])
X1 = np.random.multivariate_normal(mu1, cov1, n)
X2 = np.random.multivariate_normal(mu2, cov2, n)
X = np.concatenate((X1, X2))
y = np.repeat([0, 1], int(len(X)/2))
aris, preds, _ = exp(X, y, range(2,6))

algs = ["AutoGMM", "Naive GM", "Naive GM (multi init)", "K-Means", "Agglomerative"]
fig = plt.figure(figsize=(16, 4.5))
fig.subplots_adjust(top=0.8)
c_list = ['red', 'green', 'blue','orange','purple','yellow','gray']
for i in range(len(algs)):
    ax = plt.subplot(1, 5, i+1)
    max_c = np.max(preds[i])
    plt.scatter(X[:,0], X[:,1], c=preds[i], s=10,
                cmap=colors.ListedColormap(c_list[:max_c+1]))
    plt.title(algs[i])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    if i == 2:
        plt.xlabel('First Dimension', fontsize=22)
    elif i == 0:
        plt.ylabel('Second Dimension', fontsize=22)
    if i !=0:
        ax.set(yticklabels=[])
    plt.text(.97, .92, ('%.2f' % aris[i]).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
fig.suptitle('Synthetic Double-Cigar Dataset', y=0.93, fontsize=22, fontweight='bold')
plt.savefig('cigar_pred.png', transparancy=False, facecolor='white', bbox_inches = "tight", dpi=300)

#%%
# make boxplot of ARIs for cigar datasets
np.random.seed(1)
rep = 100
cov_1 = 1
cov_2 = 200
n = 100
mu_ = 3
mu1 = [-mu_, 0]
mu2 = [mu_, 0]

def _run():
    cov1 = np.array([[cov_1, 0], [0, cov_2]])
    cov2 = np.array([[cov_1, 0], [0, cov_2]])
    X1 = np.random.multivariate_normal(mu1, cov1, n)
    X2 = np.random.multivariate_normal(mu2, cov2, n)
    X = np.concatenate((X1, X2))
    y = np.repeat([0, 1], int(len(X)/2))
    aris, _, _ = exp(X, y, range(2,6), 10)
    return aris

aris_all_ellip = Parallel(n_jobs=35, verbose=1)(
    delayed(_run)() for _ in range(rep)
)

df = pd.DataFrame(aris_all_ellip)
df.columns = ['AutoGMM', 'Naive GM', 'Naive GM\n(multi init)', 'K-Means', 'Agglomerative']
df = pd.melt(df, value_vars=df.columns, value_name='ARI', var_name='Algorithm')
fig, ax = plt.subplots(1, figsize=(12, 6))
sns.swarmplot(data=df, x='Algorithm', y='ARI', ax=ax, size=3)
sns.boxplot(data=df, x='Algorithm', y='ARI', ax=ax, boxprops=dict(alpha=.25),
            whis=np.inf, notch=True, width=0.1)
ax.set_title('ARIs for Different Clustering Algorithms', fontsize=20, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('ARI', fontsize=20)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
plt.savefig('cigar_aris_boxplot.png', transparancy=False, facecolor='white', bbox_inches = "tight", dpi=300)
plt.show()
