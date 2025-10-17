import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import jensenshannon

from .strategy import Strategy

class AccuACL(Strategy):
    def __init__(self, X, Y, idxs_lb, cl_strategy, args, fname, lam, exp, **kwargs):
        self.lam = lam
        self.exp_data = exp._dataset.with_transforms('eval')
        super().__init__(X, Y, idxs_lb, cl_strategy, args, fname, **kwargs)
        self.X = torch.stack([data[0] for data in self.exp_data])


    def get_top_idxs(self, i_p, emb_dim=500):
        dim = self.cl_strategy.model.get_embedding_dim()
        num_class = len(i_p)//dim
        k = emb_dim//num_class
        _, indices = i_p.reshape(-1,dim).topk(k, dim=1)
        indices = torch.Tensor([[i*dim] for i in range(num_class)]).int()+indices
        return indices.flatten()

    def query(self, n, n_aug=3, alpha=2.0, debug=False, **kwargs):
        self.cl_strategy.model.eval()
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        idxs_labeled = np.arange(self.n_pool)[self.idxs_lb]
        dim = self.cl_strategy.model.get_embedding_dim()
        topk_dim = 5*self.n_classes

        if self.mem_X is None:
            exp_classes = self.Y.unique()
            cur_fims = self.get_fisher_embedding(self.X[idxs_unlabeled], exp_classes=exp_classes)
            lab_fims = self.get_fisher_embedding(self.X[idxs_labeled], exp_classes=exp_classes)
            
            fims =  torch.concat((cur_fims,lab_fims), dim=0)
            self.fim = cur_fims.mean(0)
            budget = torch.concat([torch.ones(len(idxs_unlabeled)), torch.zeros(len(idxs_labeled))])

        else:
            exp_classes = torch.concat((self.mem_Y.clone().cpu(),self.Y)).unique()

            cur_fims = self.get_fisher_embedding(self.X[idxs_unlabeled], exp_classes=exp_classes)
            lab_fims = self.get_fisher_embedding(self.X[idxs_labeled], exp_classes=exp_classes)
            mem_fims = self.get_fisher_embedding(self.mem_X, exp_classes=exp_classes)

            fims = torch.concat((cur_fims, lab_fims, mem_fims))
            self.fim = self.lam*cur_fims.mean(0)+(1-self.lam)*mem_fims.mean(0)
            budget = torch.concat([torch.ones(len(idxs_unlabeled)), torch.zeros(len(self.mem_X)+len(idxs_labeled))])

        prob_fims = fims[budget.bool()]
        fim_emb_idx = self.get_top_idxs(self.fim, topk_dim)
        self.fim = self.fim[fim_emb_idx]
        fims = fims[:, fim_emb_idx]
        prob_fims = prob_fims[:, fim_emb_idx]

        prob_i_p = F.softmax(self.fim/self.fim.mean(), dim=0)
        prob_fims = F.softmax(prob_fims/self.fim.mean(), dim=1)
        js_div = np.array([jensenshannon(p_f, prob_i_p) for p_f in prob_fims])
        norms = fims[budget.bool()].norm(dim=1,p=2).numpy()

        score = np.exp(-js_div/js_div.mean())
        chosen = np.argsort(score)[-2*n:]
        chosen = chosen[np.argsort(norms[chosen])[-n:]]

        if debug:
            return fims, self.fim, norms, score, chosen
        return idxs_unlabeled[chosen]