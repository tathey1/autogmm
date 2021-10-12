"""
This outputs the double-dendrogram showing the clustering predictions of HGMM
on the Drosophila dataset, which is Figure 9 in the paper.
"""

#%%
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib as mpl

import pyreadr
import colorcet as cc

from graspologic.cluster import DivisiveCluster

rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
    "axes.edgecolor": "lightgrey",
    "ytick.color": "grey",
    "xtick.color": "grey",
    "axes.labelcolor": "dimgrey",
    "text.color": "dimgrey",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
}
for key, val in rc_dict.items():
    mpl.rcParams[key] = val
context = sns.plotting_context(context="talk", font_scale=3, rc=rc_dict)
sns.set_context(context)

#%%
# load data
path = "../../../data"

# R hemi
c_true_R = pd.read_csv(path + "/classes_R.csv")
c_true_R = c_true_R.to_numpy().reshape((len(c_true_R)))
Xhat_R = pd.read_csv(path + "/Xhat_R.csv")
Xhat_R = Xhat_R.iloc[:, 1:].to_numpy()
mb_R_claw = pyreadr.read_r(path + "/claw.rda")


# L hemi
c_true_L = pd.read_csv(
    path + "/classes_L.csv", index_col=0
)  # these are strings of types
c_true_L = c_true_L.replace({"KC": 1, "MBIN": 2, "MBON": 3, "PN": 4})
c_true_L = c_true_L.to_numpy().reshape((len(c_true_L)))
Xhat_L = pd.read_csv(path + "/Xhat_L.csv")
Xhat_L = Xhat_L.iloc[:, 1:].to_numpy()

#%%
# cluster the two hemispheres separately

np.random.seed(8888)
rc = DivisiveCluster(max_components=6, max_level=2)
labels_R_mb2017 = rc.fit_predict(Xhat_R, fcluster=True)

np.random.seed(8888)
rc = DivisiveCluster(max_components=6, max_level=2)
labels_L_mb2017 = rc.fit_predict(Xhat_L, fcluster=True)

#%%
# using predicted labels to generate meta data


def df_label(labels):
    # level 0
    l0 = np.zeros((len(labels), 1))
    l0 = l0.astype(int).astype(str)

    # level 1
    l1 = np.empty((1, 1))
    for i in range(len(labels)):
        new_label = str(0) + "-" + str(labels[i, 0])
        l1 = np.append(l1, new_label)
    l1 = l1[1:]

    # level 2
    n_clus = np.max(labels[:, 0]) + 1  # n_clus at level 1
    l2 = labels[:, 1].copy().astype(str)
    for i in range(n_clus):
        inds = labels[:, 0] == i
        indx = np.where(labels[:, 0] == i)[0]
        single_clus = labels[:, 1][indx]
        _, single_clus_label = np.unique(single_clus, return_inverse=True)
        n_unique = np.sum(inds)
        for j in range(n_unique):
            new_label = str(0) + "-" + str(i) + "-" + str(single_clus_label[j])
            l2[indx[j]] = new_label

    l = np.column_stack((l0, l1, l2))

    labels = pd.DataFrame(l.copy())
    labels.columns = ["lvl0_labels", "lvl1_labels", "lvl2_labels"]

    return labels


# R hemi
class_label = "class0"
meta_R = pd.DataFrame(np.zeros(len(c_true_R)))
meta_R[class_label] = ""
major_classes = ["KC", "MBIN", "MBON", "PN"]
for i in range(len(major_classes)):
    inds = np.where(c_true_R == i + 1)[0]
    meta_R[class_label].iloc[inds] = major_classes[i]
del meta_R[0]

len_KC = np.sum(c_true_R == 1)

mb_r_sub = pd.read_csv(path + "/out_vdf_v_R.csv")
mb_r_sub = mb_r_sub.loc[:, ~mb_r_sub.columns.str.contains("^Unnamed")]
mb_r_sub["inds"] = range(len(mb_r_sub))

mb_r_sub["Skeleton.ID"] = mb_r_sub["x"].iloc[:len_KC].str.split("#").str[1]
mb_r_sub["Skeleton.ID"].iloc[27] = "29"  # b/c this neuron has 2 IDs: #0 & #29
mb_r_sub["Skeleton.ID"].iloc[:len_KC] = (
    mb_r_sub["Skeleton.ID"].iloc[:len_KC].astype(int)
)
mb_r_sub["type"] = ""

mb_r_claw = pyreadr.read_r(path + "/claw.rda")

for i in range(len_KC):
    inds = (
        mb_r_claw["claw"]
        .index[mb_r_claw["claw"]["Skeleton.ID"] == mb_r_sub["Skeleton.ID"].iloc[i]]
        .tolist()
    )
    mb_r_sub["type"].iloc[i] = mb_r_claw["claw"]["type"].iloc[inds[0]]

meta_R = pd.concat([meta_R, mb_r_sub["type"]], axis=1)
meta_R["type"].iloc[len_KC:] = meta_R["class0"].iloc[len_KC:]
meta_R["inds"] = range(len(meta_R))

labels_R_mb2017 = df_label(labels_R_mb2017)

for i in range(3):
    meta_R[f"lvl{i}_labels"] = labels_R_mb2017[f"lvl{i}_labels"].values

lowest_level = 2
level_names = [f"lvl{i}_labels" for i in range(lowest_level + 1)]

sort_class = level_names + ["type"]
total_sort_by = []
for sc in sort_class:
    total_sort_by.append(sc)
meta_R = meta_R.sort_values(total_sort_by)


# L hemi
class_label = "class0"
meta_L = pd.DataFrame(np.zeros(len(c_true_L)))
meta_L[class_label] = ""
major_classes = ["KC", "MBIN", "MBON", "PN"]
for i in range(len(major_classes)):
    inds = np.where(c_true_L == i + 1)[0]
    meta_L[class_label].iloc[inds] = major_classes[i]
del meta_L[0]

len_KC = np.sum(c_true_L == 1)

mb_L_sub = pd.read_csv(path + "/out_vdf_v_L.csv")
mb_L_sub = mb_L_sub.loc[:, ~mb_L_sub.columns.str.contains("^Unnamed")]
mb_L_sub["inds"] = range(len(mb_L_sub))

mb_L_sub["Skeleton.ID"] = mb_L_sub["x"].iloc[:len_KC].str.split("#").str[1]
mb_L_sub["Skeleton.ID"].iloc[:len_KC] = (
    mb_L_sub["Skeleton.ID"].iloc[:len_KC].astype(int)
)
mb_L_sub["type"] = ""

for i in range(len_KC):
    inds = (
        mb_r_claw["claw"]
        .index[mb_r_claw["claw"]["Skeleton.ID"] == mb_L_sub["Skeleton.ID"].iloc[i]]
        .tolist()
    )
    mb_L_sub["type"].iloc[i] = mb_r_claw["claw"]["type"].iloc[inds[0]]

meta_L = pd.concat([meta_L, mb_L_sub["type"]], axis=1)
meta_L["type"].iloc[len_KC:] = meta_L["class0"].iloc[len_KC:]
meta_L["inds"] = range(len(meta_L))

labels_L_mb2017 = df_label(labels_L_mb2017)

for i in range(3):
    meta_L[f"lvl{i}_labels"] = labels_L_mb2017[f"lvl{i}_labels"].values

lowest_level = 2
level_names = [f"lvl{i}_labels" for i in range(lowest_level + 1)]

sort_class = level_names + ["type"]
total_sort_by = []
for sc in sort_class:
    total_sort_by.append(sc)
meta_L = meta_L.sort_values(total_sort_by)

#%%
# plot dendrograms

colors_MBON = np.array(cc.glasbey_bw_minc_20)[[7]]
colors_MBIN = np.array(cc.glasbey_bw_minc_20)[[4]]
colors_KC = np.array(cc.glasbey_bw_minc_20)[[82]]
colors_PN = np.array(cc.glasbey_bw_minc_20)[[11]]

colors = [colors_MBON, colors_KC, colors_MBIN, colors_PN]

# retrieve colors from cc.glasbey in the order of colors;
# diff desat so classes in colors[i] have sightly diff colors

pal = sns.color_palette(palette=colors[0], n_colors=len(colors[0]))

colors_length = [1, 3, 1, 1]  # MBON(1) + KC(3) + MBIN(1) + PN(1)
for j in range(len(colors)):
    desats = np.array(
        [
            1,
            0.8,
            0.6,
        ]
    )
    desats = desats[: colors_length[j]]
    for i in range(len(desats)):
        pal_new = sns.color_palette(
            palette=colors[j], n_colors=len(colors[j]), desat=desats[i]
        )
        pal = np.concatenate([pal, pal_new], axis=0)
pal = pal[1:]


# R hemi
class_label = "type"
sizes = meta_R.groupby([class_label], sort=False).size()
uni_class = sizes.index.unique()

# reorder pal to match the ordering of classes in uni_class
pal_new_R = np.zeros((pal.shape[0], pal.shape[1]))
seq = np.array([1, 5, 0, 2, 4, 3])
for i in range(len(seq)):
    pal_new_R[i, :] = pal[seq[i], :]

counts = sizes.values
count_map_R = dict(zip(uni_class, counts))

# L hemi
class_label = "type"
sizes = meta_L.groupby([class_label], sort=False).size()
uni_class = sizes.index.unique()

# reorder pal to match the ordering of classes in uni_class
pal_new_L = np.zeros((pal.shape[0], pal.shape[1]))
seq = np.array([0, 1, 4, 2, 3, 5])
for i in range(len(seq)):
    pal_new_L[i, :] = pal[seq[i], :]

#%%
def get_last_mids(label, last_mid_map, max_comp):
    last_mids = []
    if label + "-" in last_mid_map:
        last_mids.append(last_mid_map[label + "-"])

    for i in range(max_comp + 1):
        if label + f"-{i}" in last_mid_map:
            last_mids.append(last_mid_map[label + f"-{i}"])

    if label in last_mid_map:
        last_mids.append(last_mid_map[label])
    if len(last_mids) == 0:
        print(label + " has no anchor in mid-map")
    return last_mids


def calc_bar_params(sizes, label, mid, color_map, palette=None):
    heights = sizes.loc[label]
    n_in_bar = heights.sum()
    offset = mid - n_in_bar / 2
    starts = heights.cumsum() - heights + offset
    colors = [color_map.get(heights.index[n]) for n in range(len(heights))]
    return heights, starts, colors


def draw_bar_dendrogram(
    meta,
    ax,
    first_mid_map,
    class_label,
    color_map,
    max_comp,
    mat_gap,
    labels,
    lowest_level=7,
    width=0.5,
    draw_labels=False,
):
    last_mid_map = first_mid_map
    line_kws = dict(linewidth=1, color="k")
    for level in np.arange(lowest_level + 1)[::-1]:
        x = level
        sizes = meta.groupby([f"lvl{level}_labels", class_label], sort=False).size()
        uni_labels = sizes.index.unique(0)  # these need to be in the right order

        mids = []
        for ind, ul in enumerate(uni_labels):
            last_mids = get_last_mids(ul, last_mid_map, max_comp)
            grand_mid = np.mean(last_mids)

            heights, starts, colors = calc_bar_params(sizes, ul, grand_mid, color_map)

            minimum = starts[0]
            maximum = starts[-1] + heights[-1]
            if level == 0:
                maximum_for_labels = maximum
                heights_for_labels = heights
            mid = (minimum + maximum) / 2
            mids.append(mid)

            # draw the bars
            for i in range(len(heights)):
                ax.bar(
                    x=x,
                    height=heights[i],
                    width=width,
                    bottom=starts[i],
                    color=colors[i],
                )
                if (labels is not None) and (level == 0):
                    ax.text(
                        x=-0.7,
                        y=heights[i] / 2 + starts[i],
                        s=labels[i],
                        rotation=90,
                        va="center",
                    )

            # draw a horizontal line from the middle of this bar
            if level != 0:  # dont plot dash on the last
                ax.plot([x - 0.5 * width, x - width], [mid, mid], **line_kws)

            # line connecting to children clusters
            if level != lowest_level:  # don't plot first dash
                ax.plot(
                    [x + 0.5 * width, x + width], [grand_mid, grand_mid], **line_kws
                )

            # draw a vertical line connecting the two child clusters
            if len(last_mids) > 1:
                # ax.plot([x + width, x + width], last_mids, **line_kws)
                ax.plot(np.repeat([x + width], len(last_mids)), last_mids, **line_kws)

        last_mid_map = dict(zip(uni_labels, mids))
    return maximum_for_labels, heights_for_labels


def get_mid_map(full_meta, class_label, leaf_key=None, bilat=False, gap=10):
    if not bilat:
        meta = full_meta[full_meta["hemisphere"] == "L"].copy()
    else:
        meta = full_meta.copy()

    sizes = meta.groupby([leaf_key, class_label], sort=False).size()

    uni_labels = sizes.index.unique(0)

    mids = []
    offset = 0
    for ul in uni_labels:
        heights = sizes.loc[ul]
        starts = heights.cumsum() - heights + offset
        offset += heights.sum() + gap
        minimum = starts[0]
        maximum = starts[-1] + heights[-1]
        mid = (minimum + maximum) / 2
        mids.append(mid)

    left_mid_map = dict(zip(uni_labels, mids))
    if bilat:
        first_mid_map = {}
        for k in left_mid_map.keys():
            left_mid = left_mid_map[k]
            first_mid_map[k + "-"] = left_mid
        return first_mid_map

    # right
    meta = full_meta[full_meta["hemisphere"] == "R"].copy()

    sizes = meta.groupby([leaf_key, class_label], sort=False).size()

    # uni_labels = np.unique(labels)
    uni_labels = sizes.index.unique(0)

    mids = []
    offset = 0
    for ul in uni_labels:
        heights = sizes.loc[ul]
        starts = heights.cumsum() - heights + offset
        offset += heights.sum() + gap
        minimum = starts[0]
        maximum = starts[-1] + heights[-1]
        mid = (minimum + maximum) / 2
        mids.append(mid)

    right_mid_map = dict(zip(uni_labels, mids))

    keys = list(set(list(left_mid_map.keys()) + list(right_mid_map.keys())))
    first_mid_map = {}
    for k in keys:
        left_mid = left_mid_map[k]
        right_mid = right_mid_map[k]
        first_mid_map[k + "-"] = max(left_mid, right_mid)
    return first_mid_map


def palplot(
    xmin,
    xmax,
    ymin,
    ymax,
    pal,
    count_map,
    cmap="viridis",
    figsize=(1, 10),
    ax=None,
    start=0,
    stop=None,
):

    k = len(pal)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    pal_single = pal[0].reshape((1, 1, 3))
    pal_single = np.repeat(pal_single, list(count_map.values())[0], axis=0)
    pals = pal_single
    for i in range(1, len(pal)):
        pal_single = pal[i].reshape((1, 1, 3))
        pal_single = np.repeat(pal_single, list(count_map.values())[i], axis=0)
        pals = np.concatenate((pals, pal_single), axis=0)
    pals = pals.reshape(len(pals), 1, 3)[::-1]
    ax.imshow(pals, extent=(xmin, xmax, ymin, ymax))


#%%
width_ratios = [0.5, 0.5]
lowest_level = 2
fig, axs = plt.subplots(
    1, 2, figsize=(15, 50), gridspec_kw=dict(width_ratios=width_ratios)
)

# L hemi
sizes = meta_L.groupby(["type"], sort=False).size()
uni_class = sizes.index.unique()
counts = sizes.values
color_map = dict(zip(uni_class, pal_new_L))
leaf_key = f"lvl{lowest_level}_labels"
n_leaf = meta_L[leaf_key].nunique()
n_cells = len(meta_L)
first_mid_map = get_mid_map(meta_L, class_label, leaf_key=leaf_key, bilat=True)
gap = 10
width = 0.5
ax = axs[0]
ax.set_ylim((-gap, (n_cells + gap * n_leaf)))
ax.set_xlim((-0.5, lowest_level + 0.5))

maximum_for_labels, heights_for_labels = draw_bar_dendrogram(
    meta_L,
    ax,
    first_mid_map,
    class_label,
    color_map,
    mat_gap=0.4,
    labels=["MBON", "KC(y)", "MBIN", "KC(m)", "KC(s)", "PN"],
    lowest_level=lowest_level,
    draw_labels=False,
    max_comp=6,
)

ax.set_yticks([])
ax.set_xticks(np.arange(lowest_level + 1))
ax.tick_params(axis="both", which="both", length=0)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.set_xlabel("Depth")
ax.set_xticklabels(np.arange(lowest_level + 1), rotation=90)
ax.bar(
    x=0, height=10, bottom=0, width=width, color="k"
)  # add a scale bar in the bottom left
ax.text(x=0.35, y=5, s="10 neurons", rotation=90, va="center", fontsize=35)
_, ymax = ax.get_ylim()
ax.text(
    x=0,
    y=ymax,
    s="Left",
    rotation=90,
    fontsize=60,
    fontweight="bold",
    ha="center",
    va="top",
)

# R hemi
sizes = meta_R.groupby(["type"], sort=False).size()
uni_class = sizes.index.unique()
counts = sizes.values
color_map = dict(zip(uni_class, pal_new_R))
leaf_key = f"lvl{lowest_level}_labels"
n_leaf = meta_R[leaf_key].nunique()
n_cells = len(meta_R)
first_mid_map = get_mid_map(meta_R, class_label, leaf_key=leaf_key, bilat=True)
gap = 10
width = 0.5
ax = axs[1]
ax.set_ylim((-gap, (n_cells + gap * n_leaf)))
ax.set_xlim((lowest_level + 0.5, -0.5))  # reversed

maximum_for_labels, heights_for_labels = draw_bar_dendrogram(
    meta_R,
    ax,
    first_mid_map,
    class_label,
    color_map,
    mat_gap=0.6,
    labels=None,
    lowest_level=lowest_level,
    draw_labels=False,
    max_comp=6,
)

ax.set_yticks([])
ax.set_xticks(np.arange(lowest_level + 1))
ax.tick_params(axis="both", which="both", length=0)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.set_xlabel("Depth")
ax.set_xticklabels(np.arange(lowest_level + 1), rotation=90)
_, ymax = ax.get_ylim()
ax.text(
    x=0,
    y=ymax,
    s="Right",
    rotation=90,
    fontsize=60,
    fontweight="bold",
    ha="center",
    va="top",
)

plt.subplots_adjust(hspace=0)
plt.tight_layout()
plt.savefig("/results/maggot_dendrograms", bbox_inches="tight", dpi=300)
