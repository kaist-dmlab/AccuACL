import numpy as np
from .strategy import Strategy
import pdb
class LeastConfidence(Strategy):
    def __init__(self, X, Y, idxs_lb, cl_strategy, args, fname, **kwargs):
        super(LeastConfidence, self).__init__(X, Y, idxs_lb, cl_strategy, args, fname, **kwargs)

    def query(self, n, **kwargs):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], np.asarray(self.Y)[idxs_unlabeled])
        U = probs.max(1)[0]
        return idxs_unlabeled[U.sort()[1][:n]]
