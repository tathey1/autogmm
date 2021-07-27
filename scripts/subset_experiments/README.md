To recreate Figure 4:

1. Run the 3 scripts *_subset_experiments.py/r
2. Run compare_subset_experiments.py

**idxs_*.csv** - 10 random subsets of the datasets. Each subset contains 80% as many elements as the full dataset. These are the random subsets that were used in the paper.

**./paper_results/** - Contains the results from Step 1 that were published. If one wishes to recreate exactly Figure 4, copy these csvs into the subset_experiments directory then perform Step 2.

Note: For the R script, users may need to "Set Working Directory" to "Source File Location," for the scripts to find find relative paths correctly.