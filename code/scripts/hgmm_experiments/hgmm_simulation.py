"""
This saves figures showing the clustering results of HGMM on simulated dataset.
It outputs the 4 subplots of Figure 8 in the paper.
"""

#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from sklearn.metrics import adjusted_rand_score
from graspologic.cluster import DivisiveCluster

#%%
def calculate_means(loc_scale, center, length, level):
    means = [center - length, center + length]
    for lvl in range(level - 1):
        length = length * loc_scale
        for i in range(len(means)):
            means[i] = [means[i] - length, means[i] + length]
            means = list(np.array(means, dtype=object).ravel())
    return means


def GM_data(cov_scale, n_sample, loc_scale=0.3):

    length = 10
    n_level = 3
    n_GM = 2

    means = calculate_means(loc_scale, center=0, length=length, level=n_level)

    n_class = len(means)
    x = list(np.zeros((n_class, 1)))
    for i in range(n_class):
        x[i] = [np.random.normal(means[i], cov_scale, size=(n_sample, 1))]
    x = np.array(x).ravel()

    n_total_sample = n_sample * (n_GM ** n_level)
    y = np.zeros((n_total_sample, n_level))
    for lvl in range(n_level):
        n_repeat = n_total_sample // (n_GM ** (lvl + 1))
        y[:, lvl] = np.repeat(range(n_GM ** (lvl + 1)), n_repeat)

    return x, y


#%%
# plot the histogram of the simulated data

np.random.seed(1)
cov_scales = [0.5]
x = {}
for i in range(len(cov_scales)):
    x[i] = np.empty((1, 1))

n_trials = 100
n_samples = 100
for i in range(len(cov_scales)):
    for j in range(n_trials):
        x[i] = np.append(
            x[i], GM_data(cov_scales[i], n_samples)[0].reshape((-1, 1)), axis=0
        )

x_all = {}
for i in range(len(cov_scales)):
    x_all[i] = x[i][1:]

y = np.repeat(np.arange(1, 9), n_samples)
y = np.tile(y, n_trials)

fig, ax = plt.subplots(1, figsize=(10, 5))
binwidth = 0.2

i = 0
x = x_all[i]
df = pd.DataFrame(np.hstack((x, y.reshape((-1, 1)))))
df[1] = df[1].astype(int)
df.columns = ["x", "label"]
hists = sns.histplot(data=df, x="x", hue="label", binwidth=0.2, palette="deep", ax=ax)
ax.set_yticklabels((ax.get_yticks() / 100).astype(int), fontsize=20)
ax.set_xticklabels(ax.get_xticks().astype(int), fontsize=20)
ax.set_xlabel("First Dimension", fontsize=22)
ax.set_ylabel("Count", fontsize=22)
ax.set_title(
    "a) Distribution of Synthetic Dataset", fontsize=24, fontweight="bold", pad=22
)
legend = hists.get_legend()
ax.legend(
    legend.legendHandles,
    (1, 2, 3, 4, 5, 6, 7, 8),
    title="label",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    borderaxespad=0.0,
)
plt.setp(ax.get_legend().get_texts(), fontsize="22")  # for legend text
plt.setp(ax.get_legend().get_title(), fontsize="22")  # for legend title
plt.tight_layout()
plt.savefig("/results/data_hist.png", bbox_inches="tight", dpi=300)

#%%
def heatmap_label(pred_sorted, title, figname):
    p = pred_sorted.copy()
    n_level = p.shape[1]
    ari = np.zeros((n_level, 2))
    for l in range(n_level):
        if l <= 2:
            ari[l, 0] = adjusted_rand_score(y[:, l], p[:, l])
            ari[l, 1] = 2 ** (l + 1)
        else:
            ari[l, 0] = adjusted_rand_score(y[:, -1], p[:, l])
            ari[l, 1] = 8

    fig, axs = plt.subplots(
        n_level, 1, figsize=(20, n_level + 1.5), sharex=True, sharey=True
    )
    n_clusters = [max(p[:, i] + 1) for i in range(n_level)]
    for i in range(n_level):
        ax = axs[i]
        sns.heatmap(
            p[:, i].reshape((1, -1)) + 1,
            annot=False,
            cbar=False,
            xticklabels=100,
            yticklabels="",
            square=False,
            cmap="RdBu_r",
            center=0,
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticks().astype(int), fontsize=18)
        if i < p.shape[1] - 1:
            ax.set(xticklabels="")
        else:
            ax.set_xlabel("Node Index", fontsize=22)
        if i == 0:
            ax.set_title(title, fontsize=25, fontweight="bold", pad=20)
            ax.text(825, -0.5, "# clusters", fontsize=22, va="center", ha="center")
            ax.text(880, -0.5, "ARI", fontsize=22, va="center", ha="center")

        ax.set_ylabel(i + 1, fontsize=22)
        ax.text(825, 0.5, str(n_clusters[i]), fontsize=22, va="center", ha="center")
        ax.text(880, 0.5, np.round(ari[i, 0], 3), fontsize=22, va="center", ha="center")

    fig.text(-0.01, 0.5, "Depth", fontsize=22, rotation=90, va="center", ha="center")
    plt.tight_layout()
    plt.savefig(figname, bbox_inches="tight", dpi=300)


def relabel(pred):
    # reorder the labels so that the clusters in each array
    # recieve increasing labels
    for i in range(pred.shape[1]):
        temp = pred[:, i].copy()
        _, index = np.unique(temp, return_index=True)
        # return unique labels in the order of their appearance
        uni_labels = temp[np.sort(index)]
        for label, ul in enumerate(uni_labels):
            inds = temp == ul
            temp[inds] = -label - 1
        pred[:, i] = -(temp + 1)

    return pred


#%%
# make the heatmap of clustering results

np.random.seed(6)
rc = DivisiveCluster(max_level=6, max_components=2)
x, y = GM_data(0.5, 100)
x = x.reshape((-1, 1))
pred_scale5 = rc.fit_predict(x, fcluster=True)
heatmap_label(
    relabel(pred_scale5), "b) Clustering Assignments", "/results/sample_dendrogram.png"
)

#%%
def dim1_3lvl_2GM(cov_scale, n_sample, loc_scale=0.3):

    length = 10
    n_level = 3
    n_GM = 2

    means = calculate_means(loc_scale, center=0, length=length, level=n_level)

    n_class = len(means)
    x = list(np.zeros((n_class, 1)))
    for i in range(n_class):
        x[i] = [np.random.normal(means[i], cov_scale, size=(n_sample, 1))]
    x = np.array(x).ravel()

    n_total_sample = n_sample * (n_GM ** n_level)
    y = np.zeros((n_total_sample, n_level))
    for lvl in range(n_level):
        n_repeat = n_total_sample // (n_GM ** (lvl + 1))
        y[:, lvl] = np.repeat(range(n_GM ** (lvl + 1)), n_repeat)

    return x, y


#%%
# make the scatterplot and histogram to show the ARIs of flat clusterings
# and the numbers of clusters of leaf clusterings, respectively.

np.random.seed(1)
scale = 0.5
iteration = 50


def _run():
    x, _ = dim1_3lvl_2GM(scale, 100, loc_scale=0.3)
    x = x.reshape((-1, 1))
    rc = DivisiveCluster(max_level=6, max_components=2)
    pred = rc.fit_predict(x, fcluster=True)

    return pred


pred_all = Parallel(n_jobs=25, verbose=1)(delayed(_run)() for _ in range(iteration))


_, y = GM_data(0.5, 100)

n = 50
ARIs = {}
for i in range(n):
    pred = pred_all[i]
    ARI = np.zeros((y.shape[1], pred.shape[1]))
    for j in range(y.shape[1]):
        for k in range(pred.shape[1]):
            ARI[j, k] = adjusted_rand_score(y[:, j], pred[:, k])
    if ARI.shape[1] > 6:
        ARI = ARI[:, :6]
    ARIs[i] = ARI

scales = [0.5]
b = [ARIs[i] for i in range(n)]
df = pd.DataFrame([])
for i in range(n):
    b1 = [b[i][0, 0], b[i][1, 1]]
    for j in range(len(b[i][2, 2:])):
        b1.append(b[i][2, 2:][j])
    b1 = pd.DataFrame(np.array(b1).reshape(1, -1))
    df = pd.concat([df, b1])

df.columns = [n + 1 for n in range(df.shape[1])]
df.index = range(len(df))

fig, axs = plt.subplots(1, 2, figsize=(20, 8))

ax = axs[0]
jitter = 0.1
df_x_jitter = pd.DataFrame(
    np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns
)
df_x_jitter += np.arange(len(df.columns))

for col in df:
    ax.plot(df_x_jitter[col], df[col], "o", alpha=0.40, zorder=1)
ax.set_xticks(range(len(df.columns)))
ax.set_xticklabels(df.columns)
ax.set_xlim(-0.5, len(df.columns) - 0.5)

for col in range(1, 6):
    for idx in df.index:
        ax.plot(
            df_x_jitter.loc[idx, [col, col + 1]],
            df.loc[idx, [col, col + 1]],
            color="grey",
            linewidth=0.5,
            linestyle="--",
            zorder=-1,
        )

ax.set(ylim=(0.79, 1.02), xlim=(-0.5, 5.5))
ax.set_xlabel("Depth", fontsize=22)
ax.set_ylabel("ARI", fontsize=22)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(np.round(ax.get_yticks(), 3), fontsize=20)
ax.set_title(
    "c) Flat Clustering ARIs across Depths", fontsize=24, fontweight="bold", pad=20
)

ax = axs[1]
k_hat = [len(np.unique(pred_all[i][:, -1])) for i in range(len(pred_all))]
sns.histplot(k_hat, ax=ax, binwidth=1)
ax.set_xlabel("Number of Clusters", fontsize=22)
ax.set_ylabel("Frequency", fontsize=22)
ax.set_title(
    "d) Distribution of the Number of Clusters\n in the Leaf Clustering",
    fontsize=24,
    fontweight="bold",
    pad=20,
)
ax.set_xticks(ax.get_xticks() + 0.5)
ax.set_xticklabels(ax.get_xticks().astype(int), fontsize=22)
ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=22)
ymin, ymax = plt.ylim()
ax.set_ylim(ymax=ymax)
plt.plot([8.5, 8.5], [ymin, ymax], linewidth=4, color="r", linestyle="--")

plt.tight_layout()
plt.savefig("/results/clustering_performance.png", bbox_inches="tight", dpi=300)
