To recreate Figure 6:

1. Run *_option_runtimes.py/r to run the algorithms on the data
2. Run show_runtimes.py

**make_data.py** - file used to make synthetic mixture data in the **./data** directory. \
**./data/** - synthetic mixture data used in the paper.
Note that due to Github filesize constraints, we did not upload the n=655360 dataset or the n=1310720 dataset.
Thus, if you want to make a figure that more closely represents that in the paper, then you should used make_data.py to make datasets of those sizes, then adjust the "num_sets" variable in each script if necessary. \
**./paper/** - clustering results published in the paper.
