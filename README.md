# AutoGMM

## `AutoGMM` is a module for automatic and hierarchical Gaussian mixture modeling in `graspologic`.


# Documentation
The official documentation with usage is at https://graspologic.readthedocs.io/en/latest/

Please visit the [tutorial section](https://microsoft.github.io/graspologic/tutorials/clustering/autogmm.html) in the official website for more in depth usage.

# System Requirements
## Hardware requirements
`graspologic` package requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
`graspologic` is tested on the following OSes:
- Linux x64
- macOS x64
- Windows 10 x64

And across the following versions of Python:
- 3.6 (x64)
- 3.7 (x64)
- 3.8 (x64)


# Installation Guide
## Install from pip
```
pip install graspologic
```

## Install from Github
```
git clone https://github.com/microsoft/graspologic
cd graspologic
python3 -m venv venv
source venv/bin/activate
python3 setup.py install
```

The algorithms are located in [microsoft/graspologic/graspologic/cluster/](https://github.com/microsoft/graspologic/tree/dev/graspologic/cluster).

To run the R scripts, you will need to install R and the mclust library (we use version 5.4.2 in the paper). We recommend the RStudio IDE https://www.rstudio.com/.
Users may need to "Set Working Directory" to "Source File Location," for the scripts to find find relative paths correctly.

# Directories
## scripts
### complete_experiments
These files reproduce Table 2, Figures 1-3, and Figure 5. They run the clustering algorithms on the complete datasets. Instructions within.

### subset_experiments
These files reproduce Figure 4. They run the clustering algorithms on the subsets of the data. Instructions within.

### option_runtimes
These files reproduce Figure 6. Instructions within.

### compare_clusterings
These files reproduce Figure 7. They run and compare various clustering algorithms on the double-cigar dataset. Instructions within.

### hgmm_experiments
These files reproduce Figures 8-9. They run the hierarchical clustering algorithm on simulated and real datasets. Instructions within.

**brute_cluser_graspyclust.py** - implementation of graspyclust \
**make_gmix.py** - script that was used to make data/synthetic.csv 

## data
contains the datasets that were used in the paper


# Contributing
We welcome contributions from anyone. Please see our [contribution guidelines](https://github.com/microsoft/graspologic/blob/dev/CONTRIBUTING.md) before making a pull request. Our
[issues](https://github.com/microsoft/graspologic/issues) page is full of places we could use help!
If you have an idea for an improvement not listed there, please
[make an issue](https://github.com/microsoft/graspologic/issues/new) first so you can discuss with the developers.

# License
This project is covered under the MIT License.

# Issues
We appreciate detailed bug reports and feature requests (though we appreciate pull requests even more!). Please visit our [issues](https://github.com/microsoft/graspologic/issues) page if you have questions or ideas.

# Citing `AutoGMM`
If you find `AutoGMM` useful in your work, please cite the algorithm via the [AutoGMM paper](https://arxiv.org/abs/1909.02688)

> Athey, T. L., Pedigo, B. D., Liu, T., & Vogelstein, J. T. (2019). AutoGMM: Automatic and Hierarchical Gaussian Mixture Modeling in Python. arXiv preprint arXiv:1909.02688.