import numpy as np
import torch
from .strategy import Strategy

class EntropySampling(Strategy):
	def __init__(self, X, Y, idxs_lb, cl_strategy, args, fname, **kwargs):
		super().__init__(X, Y, idxs_lb, cl_strategy, args, fname, **kwargs)

	def query(self, n, debug=False, **kwargs):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)
		if debug is True:
			return U.sort()[1][:n]
		return idxs_unlabeled[U.sort()[1][:n]]
