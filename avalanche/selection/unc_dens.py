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
from sklearn.neighbors import NearestNeighbors

class DistUnc_KMeans(CoresetMethod):
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
    
    def mean_distance_to_nearest_neighbors(self, data_points, k=10):
        nn = NearestNeighbors(n_neighbors=k+1)
        nn.fit(data_points)
        
        distances, _ = nn.kneighbors(data_points)
        return distances.mean(axis=1)

    def construct_matrix(self):
        model = self.strategy.model
        device = self.strategy.device
        data_entropy = []
        data_matrix = []
        data_loader = DataLoader(self.dst_train, batch_size=self.strategy.train_mb_size)
        model.eval()
        with torch.no_grad():
            for i, (inputs, _, _) in enumerate(data_loader):
                data_matrix.append(model.get_emb(inputs.to(device)).detach().cpu().numpy())
                probs = F.softmax(model(inputs.to(device)), dim=1)
                ent = Categorical(probs=probs).entropy().detach().cpu().numpy()
                data_entropy.append(ent)

        data_matrix = np.vstack(data_matrix)
        model.train()
        df_data = pd.DataFrame()
        df_data['embs'] = [i for i in data_matrix]
        df_data['entropy'] = np.concatenate(data_entropy).ravel()/np.log(len(self.mem.targets.uniques))
        return df_data

    def select(self, thresh_sigma=3, **kwargs):
        if len(self.mem)==0:
            #for first stage, sample randomly
            selection_result = np.random.choice(np.arange(self.n_train), self.coreset_size, replace=False)
            return self.index[selection_result]
        
        df_data = self.construct_matrix()
        df_data['mean_dist'] = self.mean_distance_to_nearest_neighbors(np.vstack(df_data['embs'].tolist()))
        df_data['sample_weight'] = df_data.apply(lambda x: x.entropy/x.mean_dist, axis=1)
        
        # import pdb;pdb.set_trace()

        _, indices = kmeans_plusplus(
            X=np.vstack(df_data['embs']),
            n_clusters=self.coreset_size,
            sample_weight=df_data['sample_weight'].tolist()
        )

        return self.index[indices]
