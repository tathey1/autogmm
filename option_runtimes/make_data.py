import sys
sys.path.append("..")
from make_gmix import make_gmix_data
import numpy as np
import random
random.seed(0)

base = 10
factor = 2
num_sets = 22

ns = base*np.power(factor,np.arange(num_sets))


for n in ns:
    file = ".\data\\" + str(n) + ".csv"
    _ = make_gmix_data(n,file)