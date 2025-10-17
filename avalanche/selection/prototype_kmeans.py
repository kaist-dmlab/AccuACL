import numpy as np
import pandas as pd
from .coresetmethod import CoresetMethod
from .kcentergreedy import k_center_greedy
from .uniform import Uniform
from .uncertainty import Uncertainty
from .methods_utils import euclidean_dist
from torch.utils.data.dataloader import DataLoader
import torch
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

class Prototype_KMeans(CoresetMethod):
    def __init__(self, dst_train, strategy, fraction=0.5, mem=None, index=None, **kwargs):
        super().__init__(dst_train, fraction)
        self.n_train = len(dst_train)
        self.strategy = strategy
        self.mem = mem
        self.data = dst_train

        if index is not None:
            assert len(dst_train) == len(index)
            self.index = index
        else:
            self.index = np.arange(len(dst_train))

    def generate_gaussian_prototypes(self, classes, df):
        prototypes = []
        for t in sorted(classes):
            if len(df[df.target==t])>1:
                gmm = GaussianMixture(1, covariance_type='diag').fit(df[df.target==t]['embs'].to_list())
                prototypes.append(gmm)
        return prototypes

    def map_prototype(self, embs, prototypes, thresh_sigma):
        probs = [g.score_samples(embs) for g in prototypes]
        threshold = [g.score_samples((g.means_+thresh_sigma*g.covariances_).reshape(1,-1))[0] for g in prototypes]
        return probs, threshold


    def construct_matrix(self):
        model = self.strategy.model
        device = self.strategy.device
        mem_matrix = []
        mem_targets = []
        data_matrix = []
        data_entropy = []
        probability = []
        data_margin = []
        mem_loader = DataLoader(self.mem, batch_size=self.strategy.train_mb_size)
        data_loader = DataLoader(self.dst_train, batch_size=self.strategy.train_mb_size)
        model.eval()
        with torch.no_grad():
            for i, (inputs, targets, _) in enumerate(mem_loader):
                mem_matrix.append(model.get_emb(inputs.to(device)).detach().cpu().numpy())
                mem_targets.append(targets.detach().cpu().numpy())
            for i, (inputs, _, _) in enumerate(data_loader):
                data_matrix.append(model.get_emb(inputs.to(device)).detach().cpu().numpy())
                probs = F.softmax(model(inputs.to(device)), dim=1)
                data_entropy.append(Categorical(probs=probs).entropy().detach().cpu().numpy())
                probs_sorted, idxs = probs.sort(descending=True)
                margin = 1-(probs_sorted[:,0] - probs_sorted[:,1]).detach().cpu().numpy()
                data_margin.append(margin)
                probability.append(probs.detach().cpu().numpy())

        mem_matrix = np.vstack(mem_matrix)
        mem_targets = np.concatenate(mem_targets).ravel()
        data_matrix = np.vstack(data_matrix)
        model.train()

        df_mem = pd.DataFrame()
        df_mem['embs'] = [i for i in mem_matrix]
        df_mem['target'] = mem_targets
        df_data = pd.DataFrame()
        df_data['embs'] = [i for i in data_matrix]
        df_data['entropy'] = np.concatenate(data_entropy).ravel()
        df_data['margin'] = np.concatenate(data_margin).ravel()
        probability = np.vstack(probability)
        return df_mem, df_data, data_matrix, probability

    def select(self, thresh_sigma=3, **kwargs):
        if len(self.mem)==0:
            #for first stage, sample randomly
            selection_result = np.random.choice(np.arange(self.n_train), self.coreset_size, replace=False)
            return self.index[selection_result]

        df_mem, df_data, data_matrix, probability = self.construct_matrix()
        # import pdb;pdb.set_trace()
        prototypes = self.generate_gaussian_prototypes(df_mem['target'].unique(), df_mem)
        preds, threshold = self.map_prototype(df_data['embs'].to_list(), prototypes, thresh_sigma=thresh_sigma)
        df_data['sample_weight'] = Categorical(probs=F.softmax(torch.tensor(np.array(preds).T), dim=1)).entropy().numpy()
        _, indices = kmeans_plusplus(
            X=np.vstack(df_data['embs']),
            n_clusters=self.coreset_size,
            sample_weight=df_data['sample_weight'].tolist()
        )

        return self.index[indices]
