from graspologic.cluster import GaussianCluster
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import csv
import random
from sklearn.metrics import adjusted_rand_score


def brute_graspy_cluster(
    Ns, x, covariance_types, ks, c_true, savefigs=None, graphList=None
):
    if graphList != None and "all_bics" in graphList:
        _, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
            2, 2, sharey="row", sharex="col", figsize=(10, 10)
        )
    titles = ["full", "tied", "diag", "spherical"]
    best_bic = -np.inf
    for N in Ns:
        bics = np.zeros([len(ks), len(covariance_types), N])
        aris = np.zeros([len(ks), len(covariance_types), N])
        for i in np.arange(N):
            graspy_gmm = GaussianCluster(
                min_components=ks[0],
                max_components=ks[len(ks) - 1],
                covariance_type=covariance_types,
                random_state=i,
            )
            c_hat = graspy_gmm.fit_predict(x, y=c_true)
            ari = adjusted_rand_score(c_hat, c_true)
            bic_values = -graspy_gmm.bic_.values
            ari_values = graspy_gmm.ari_.values
            bics[:, :, i] = bic_values
            aris[:, :, i] = ari_values
            bic = bic_values.max()

            if bic > best_bic:
                idx = np.argmax(bic_values)
                idxs = np.unravel_index(idx, bic_values.shape)
                best_ari_bic = ari
                best_bic = bic
                best_k_bic = ks[idxs[0]]
                best_cov_bic = titles[3 - idxs[1]]
                best_c_hat_bic = c_hat

        max_bics = np.amax(bics, axis=2)
        title = "N=" + str(N)
        if graphList != None and "all_bics" in graphList:
            ax0.plot(np.arange(1, len(ks) + 1), max_bics[:, 3])
            ax1.plot(np.arange(1, len(ks) + 1), max_bics[:, 2], label=title)
            ax2.plot(np.arange(1, len(ks) + 1), max_bics[:, 1])
            ax3.plot(np.arange(1, len(ks) + 1), max_bics[:, 0])

    if graphList != None and "best_bic" in graphList:
        # Plot with best BIC*********************************
        if c_true is None:
            best_ari_bic_str = "NA"
        else:
            best_ari_bic_str = "%1.3f" % best_ari_bic

        fig_bestbic = plt.figure(figsize=(8, 8))
        ax_bestbic = fig_bestbic.add_subplot(1, 1, 1)
        # ptcolors = [colors[i] for i in best_c_hat_bic]
        ax_bestbic.scatter(x[:, 0], x[:, 1], c=best_c_hat_bic)
        # mncolors = [colors[i] for i in np.arange(best_k_bic)]
        mncolors = [i for i in np.arange(best_k_bic)]
        ax_bestbic.set_title(
            "py(agg-gmm) BIC %3.0f from " % best_bic
            + str(best_cov_bic)
            + " k="
            + str(best_k_bic)
            + " ari="
            + best_ari_bic_str
        )  # + "iter=" + str(best_iter_bic))
        ax_bestbic.set_xlabel("First feature")
        ax_bestbic.set_ylabel("Second feature")
        if savefigs is not None:
            plt.savefig(savefigs + "_python_bestbic.jpg")

    if graphList != None and "all_bics" in graphList:
        # plot of all BICS*******************************
        titles = ["full", "tied", "diag", "spherical"]
        # ax0.set_title(titles[0],fontsize=20,fontweight='bold')
        # ax0.set_ylabel('BIC',fontsize=20)
        ax0.locator_params(axis="y", tight=True, nbins=4)
        ax0.set_yticklabels(ax0.get_yticks(), fontsize=14)

        # ax1.set_title(titles[1],fontsize=20,fontweight='bold')
        legend = ax1.legend(loc="best", title="Number of\nRuns", fontsize=12)
        plt.setp(legend.get_title(), fontsize=14)

        # ax2.set_title(titles[2],fontsize=20,fontweight='bold')
        # ax2.set_xlabel('Number of components',fontsize=20)
        ax2.set_xticks(np.arange(0, 21, 4))
        ax2.set_xticklabels(ax2.get_xticks(), fontsize=14)
        # ax2.set_ylabel('BIC',fontsize=20)
        ax2.locator_params(axis="y", tight=True, nbins=4)
        ax2.set_yticklabels(ax2.get_yticks(), fontsize=14)

        # ax3.set_title(titles[3],fontsize=20,fontweight='bold')
        # ax3.set_xlabel('Number of components',fontsize=20)
        ax3.set_xticks(np.arange(0, 21, 4))
        ax3.set_xticklabels(ax3.get_xticks(), fontsize=14)

        if savefigs is not None:
            plt.savefig(
                ".\\figures\\25_6_19_paperv2\\" + savefigs + "_graspy_bicplot2.jpg"
            )
    plt.show()

    return best_c_hat_bic, best_cov_bic, best_k_bic, best_ari_bic, best_bic
