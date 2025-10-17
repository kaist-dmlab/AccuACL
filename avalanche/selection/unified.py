import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from sklearn.mixture import GaussianMixture
from .coresetmethod import CoresetMethod
from .kcentergreedy import k_center_greedy
from .uncertainty import Uncertainty
from .methods_utils import euclidean_dist

class ACL_Unified(CoresetMethod):
    def __init__(self, dst_train, strategy, fraction=0.5, mem=None, already_selected=[], **kwargs):
        super().__init__(dst_train, fraction)
        self.n_train = len(dst_train)
        self.strategy = strategy
        self.mem = mem
        self.data = dst_train
        self.already_selected = np.array(already_selected)

    def generate_gaussian_prototypes(self, classes, df):
        prototypes = []
        for t in sorted(classes):
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
        mem_loader = DataLoader(self.mem, batch_size=self.strategy.train_mb_size)
        data_loader = DataLoader(self.dst_train, batch_size=self.strategy.train_mb_size)
        model.eval()
        with torch.no_grad():
            for i, (inputs, targets, _) in enumerate(mem_loader):
                mem_matrix.append(model.get_emb(inputs.to(device)).detach().cpu().numpy())
                mem_targets.append(targets.detach().cpu().numpy())
            for i, (inputs, _, _) in enumerate(data_loader):
                data_matrix.append(model.get_emb(inputs.to(device)).detach().cpu().numpy())
        
        mem_matrix = np.vstack(mem_matrix)
        mem_targets = np.concatenate(mem_targets).ravel()
        data_matrix = np.vstack(data_matrix)
        model.train()
    
        df_mem = pd.DataFrame()
        df_mem['embs'] = [i for i in mem_matrix]
        df_mem['target'] = mem_targets
        df_data = pd.DataFrame()
        df_data['embs'] = [i for i in data_matrix]        
        return df_mem, df_data

    def construct_matrix_first(self, index=None):
        model = self.strategy.model
        device = self.strategy.device
        model.eval()
        with torch.no_grad():
            sample_num = self.n_train if index is None else len(index)
            matrix = []
            data_loader = torch.utils.data.DataLoader(self.dst_train if index is None else
                                torch.utils.data.Subset(self.dst_train, index),
                                batch_size=self.strategy.train_mb_size,
                            )
            for i, (inputs, _, _) in enumerate(data_loader):
                matrix.append(model.get_emb(inputs.to(device)))

        model.train()
        return torch.cat(matrix, dim=0)

    def select(self, thresh_sigma=3, **kwargs):
        if len(self.mem)==0:
            matrix = self.construct_matrix_first()
            selection_result = k_center_greedy(
                matrix,
                budget=self.coreset_size,
                metric=euclidean_dist,
                device=self.strategy.device,
                already_selected=self.already_selected,             
            )
            return selection_result

        df_mem, df_data = self.construct_matrix()
        prototypes = self.generate_gaussian_prototypes(df_mem['target'].unique(), df_mem)
        preds, threshold = self.map_prototype(df_data['embs'].to_list(), prototypes, thresh_sigma=thresh_sigma)
        df_data['llh'] = np.max(np.array(preds).T, axis=1)
        df_data['prototype'] = np.argmax(np.array(preds).T, axis=1)
        group = df_data.apply(lambda x: 1 if x['llh'] > threshold[x['prototype']] else 0, axis=1)
        df_original = df_data[~group.astype(bool)].copy()
        df_expressible = df_data[group.astype(bool)].copy()

        original_budget = round(self.coreset_size * (len(df_original)/len(df_data)))
        expressible_budget = self.coreset_size-original_budget
        
        selection_result = np.array([], dtype=np.int32)
        if original_budget==0:
            assert len(df_expressible)==len(df_data)
            expressible_selection = Uncertainty(self.dst_train[np.where(group)[0].tolist()], self.strategy, fraction=self.fraction, index=df_expressible.index).select()
            selection_result = np.append(selection_result, expressible_selection)
        elif expressible_budget == 0:
            assert len(df_original) == len(df_data)
            original_selection = k_center_greedy(
                np.vstack(df_original['embs'].to_list()),
                budget=original_budget,
                metric=euclidean_dist,
                index=df_original.index,
                device=self.strategy.device,
                already_selected=self.already_selected,
            )
            selection_result = np.append(selection_result, original_selection)
        else:
            expressible_selection = Uncertainty(self.dst_train[np.where(group)[0].tolist()], self.strategy, fraction=expressible_budget/len(df_expressible), index=df_expressible.index).select()
            original_selection = k_center_greedy(
                np.vstack(df_original['embs'].to_list()),
                budget=original_budget,
                metric=euclidean_dist,
                index=df_original.index,
                device=self.strategy.device,
                already_selected=self.already_selected
            )
            selection_result = np.append(selection_result, expressible_selection)
            selection_result = np.append(selection_result, original_selection)

        assert len(np.unique(selection_result))==len(selection_result)
        return np.sort(selection_result)
