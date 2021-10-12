#%%
import numpy as np


def make_gmix_data(n, file=None):
    d = 3
    w = [0.33, 0.33, 0.33]

    k = len(w)
    z = np.zeros([d])
    means = [np.copy(z) for i in range(k)]
    means[1][0] = 5
    means[2][1] = 5
    cov = [np.identity(d) for i in w]
    x = np.zeros([n, d + 1])

    thresholds = np.cumsum(w)
    for i in np.arange(n):
        u = np.random.uniform(high=np.sum(w), size=1)
        component = np.argmax((u < thresholds))
        x[i, 0] = component
        x[i, 1:] = np.random.multivariate_normal(means[component], cov[component])

    if file != None:
        np.savetxt(file, x, delimiter=",")
    return x


#%%
