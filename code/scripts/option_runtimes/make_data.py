import sys
sys.path.append("/code/scripts")
from make_gmix import make_gmix_data
import numpy as np
import random
random.seed(0)
# path = '/code/scripts/option_runtimes/data/'
path = '/results/'

# run the following code to generate simulation data
# python scripts/option_runtimes/make_data.py -o /results/

base = 40
factor = 2
exponent = 14 #in the paper, we had the exponent range from 1 to 15

n = base*np.power(factor,exponent)


file = path + str(n) + ".csv"
_ = make_gmix_data(n,file)