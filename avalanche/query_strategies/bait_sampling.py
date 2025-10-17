import numpy as np
from torch.utils.data import DataLoader
from .strategy import Strategy
import pickle
import gc
from scipy.spatial.distance import cosine
import sys
import gc
from scipy.linalg import det
from scipy.linalg import pinv as inv
from copy import copy as copy
from copy import deepcopy as deepcopy
import torch
from torch import nn
import torchfile
from torch.autograd import Variable
from .util import DataHandler 

# import resnet
# import vgg
import torch.optim as optim
import pdb
from torch.nn import functional as F
import argparse
import torch.nn as nn
from collections import OrderedDict
from scipy import stats
import numpy as np
import scipy.sparse as sp
from itertools import product
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.utils.extmath import row_norms, squared_norm, stable_cumsum
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import _num_samples
from sklearn.utils import check_array
from sklearn.utils import gen_batches
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


def select(X, K, fisher, iterates, device, lamb=1, nLabeled=0):

    numEmbs = len(X)
    indsAll = []
    dim = X.shape[-1]
    rank = X.shape[-2]

    currentInv = torch.inverse(lamb * torch.eye(dim).to(device) + iterates.to(device) * nLabeled / (nLabeled + K))
    X = X * np.sqrt(K / (nLabeled + K))
    fisher = fisher.to(device)

    # forward selection, over-sample by 2x
    print('forward selection...', flush=True)
    over_sample = 2
    for i in tqdm(range(int(over_sample *  K))):

        # check trace with low-rank updates (woodbury identity)
        xt_ = X.to(device) 
        innerInv = torch.inverse(torch.eye(rank).to(device) + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
        traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)

        # clear out gpu memory
        xt = xt_.cpu()
        del xt, innerInv
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        # get the smallest unselected item
        traceEst = traceEst.detach().cpu().numpy()
        for j in np.argsort(traceEst)[::-1]:
            if j not in indsAll:
                ind = j
                break

        indsAll.append(ind)
        # print(i, ind, traceEst[ind], flush=True)
       
        # commit to a low-rank update
        xt_ = X[ind].unsqueeze(0).to(device)
        innerInv = torch.inverse(torch.eye(rank).to(device) + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

    # backward pruning
    print('backward pruning...', flush=True)
    for i in tqdm(range(len(indsAll) - K)):

        # select index for removal
        xt_ = X[indsAll].to(device)
        innerInv = torch.inverse(-1 * torch.eye(rank).to(device) + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)
        delInd = torch.argmin(-1 * traceEst).item()
        # print(len(indsAll) - i, indsAll[delInd], -1 * traceEst[delInd].item(), flush=True)


        # low-rank update (woodbury identity)
        xt_ = X[indsAll[delInd]].unsqueeze(0).to(device)
        innerInv = torch.inverse(-1 * torch.eye(rank).to(device) + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

        del indsAll[delInd]

    del xt_, innerInv, currentInv
    torch.cuda.empty_cache()
    gc.collect()
    return indsAll

class BaitSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, cl_strategy, args, fname, **kwargs):
        super(BaitSampling, self).__init__(X, Y, idxs_lb, cl_strategy, args, fname, **kwargs)        
        # self.lamb = args['lamb']
        self.lamb = 1e-2

    def get_exp_grad_embedding(self, X, Y, probs=[], model=[], exp_classes=None):

        model=self.cl_strategy.model     
        embDim = model.get_embedding_dim()
        model.eval()
        if exp_classes is None:
            nLab = self.n_classes
        else:
            nLab = len(exp_classes)

        embedding = np.zeros([len(Y), nLab, embDim * nLab])
        for ind in range(nLab):
            loader_te = DataLoader(DataHandler(X, Y),
                            shuffle=False, batch_size=64, num_workers=self.num_workers)
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    cout, out = model(x, repr=True)
                    out = out.data.cpu().numpy()
                    batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                    for j in range(len(y)):
                        for c in range(nLab):
                            if c == ind:
                                embedding[idxs[j]][ind][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                            else:
                                embedding[idxs[j]][ind][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
                        if len(probs) > 0: embedding[idxs[j]][ind] = embedding[idxs[j]][ind] * np.sqrt(probs[idxs[j]][ind])
                        else: embedding[idxs[j]][ind] = embedding[idxs[j]][ind] * np.sqrt(batchProbs[j][ind])
        return torch.Tensor(embedding)

    def query(self, n, **kwargs):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        # get low-rank point-wise fishers
        xt = self.get_exp_grad_embedding(self.X, self.Y)

        # get fisher
        print('getting fisher matrix...', flush=True)
        batchSize = 16 # should be as large as gpu memory allows
        nClass = torch.max(self.Y).item() + 1
        fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
        rounds = int(np.ceil(len(self.X) / batchSize))
        for i in tqdm(range(int(np.ceil(len(self.X) / batchSize)))):
            xt_ = xt[i * batchSize : (i + 1) * batchSize].to(self.device)
            op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / (len(xt)), 0).detach().cpu()
            fisher = fisher + op
            xt_ = xt_.cpu()
            del xt_, op
            torch.cuda.empty_cache()
            gc.collect()

        # get fisher only for samples that have been seen before
        nClass = torch.max(self.Y).item() + 1
        init = torch.zeros(xt.shape[-1], xt.shape[-1])
        xt2 = xt[self.idxs_lb]
        rounds = int(np.ceil(len(xt2) / batchSize))
        for i in tqdm(range(int(np.ceil(len(xt2) / batchSize)))):
            xt_ = xt2[i * batchSize : (i + 1) * batchSize].to(self.device)
            op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / (len(xt2)), 0).detach().cpu()
            init = init + op
            xt_ = xt_.cpu()
            del xt_, op
            torch.cuda.empty_cache()
            gc.collect()

        chosen = select(xt[idxs_unlabeled], n, fisher, init, self.device, lamb=self.lamb, nLabeled=np.sum(self.idxs_lb))
        return idxs_unlabeled[chosen]
