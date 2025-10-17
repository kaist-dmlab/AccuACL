import numpy as np
from .strategy import Strategy
import pdb

class RandomSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, cl_strategy, args, fname, **kwargs):
        super(RandomSampling, self).__init__(X, Y, idxs_lb, cl_strategy, args, fname, **kwargs)

    def query(self, n, **kwargs):
        inds = np.where(self.idxs_lb==0)[0]
        return inds[np.random.permutation(len(inds))][:n]
