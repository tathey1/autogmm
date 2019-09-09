# pyclust
This repo reproduces the results in "Pyclust: Automatic Gaussian Mixture Modeling in Python"


Directory:
subset_experiments

These scripts/files reproduce figures 1 and 2

The csv files that start with "idxs_" are the random subsets of the datasets that were used in the experiments.

All other csv files and images can be reproduced with the scripts

First, run pyclust_subset_experiments.py, mclust_subset_experiments.r, and graspyclust_subset_experiments.py. \
These will generate csv files with the results of each method.

Then, run compare_experiments.py to plot the results.



complete_experiments

There is a script for each clustering algorithm. \
Each script reproduces the respective results for Table 2, the Appendix, \
and in the case of the Drosophila data, Figure 3


figure_4

These scripts reproduce figure 4

The data directory contains the synthetic data used for this experiment (created by make_data.py)

The experiments can be reproduced by running pyclust_runtime.py, graspyclust_runtime.py, and mclust_runtime.r. \
These scripts save the runtime data then show_runtimes.py plots them.
