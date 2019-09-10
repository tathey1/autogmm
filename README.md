# AutoGMM
This repo reproduces the results in "AutoGMM: Automatic Gaussian Mixture Modeling in Python" (https://arxiv.org/abs/1909.02688)

# Installation
The algorithm is located in the branch of https://github.com/neurodata/graspy. So, first install graspy through github, then navigate to the autogmm branch:

```
git clone https://github.com/neurodata/graspy
cd graspy
git checkout autogmm
python3 setup.py install
```

or

```
git clone https://github.com/neurodata/graspy
cd graspy
git checkout autogmm
pip install -e .
```

The algorithm is located in graspy/graspy/cluster/autogmm.py. After autogmm is installed, you can run the scripts below.

# Directories
## complete_experiments
These files reproduce Table 2, Figures 1-3, and Figure 5. They run the clustering algorithms on the complete datasets. Instructions within.

## subset_experiments
These files reproduce Figure 4. They run the clustering algorithms on the subsets of the data. Instructions within.

## option_runtimes

These files reproduce Figure 6. Instructions within.

**brute_cluser_graspyclust.py** - implementation of graspyclust \
**make_gmix.py** - script that was used to make data/synthetic.csv \
**./data/** - contains the datasets that was used in the paper
