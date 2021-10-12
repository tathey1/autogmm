# Datasets

## synthetic mixture
**synthetic.csv**
made from make_gmix.py

## breast cancer data
**wdbc.data/.names**
from https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

## Drosophila data
- **embedded_right.csv, classes.csv** 
from https://github.com/youngser/mbstructure. Specifically, the embeddings are from the Xhat variable and the classes from the vdf$type variable in https://github.com/youngser/mbstructure/blob/master/demo/sec3.R.

- **classes_R/_L.csv, Xhat_R/_L.csv, out_vdf_v_R/_L.csv, claw.rda**
Embeddings and specific labels (or subclasses) for both hemispheres used for generating the double-dendrogram (Figure 9 in the paper) are generated from the script in **mb_data.R** or retrieved directly from the [mbstructure repo](https://github.com/youngser/mbstructure/blob/master/data/claw.rda)