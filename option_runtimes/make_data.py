import sys
sys.path.append("..")
from make_gmix import make_gmix_data
import numpy as np
import random
random.seed(0)

base = 40
factor = 2
exponent = 14 #in the paper, we had the exponent range from 1 to 15

n = base*np.power(factor,exponent)


file = ".\data\\" + str(n) + ".csv"
_ = make_gmix_data(n,file)