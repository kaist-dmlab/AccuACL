import numpy as np
import pandas as pd
from .coresetmethod import CoresetMethod
from .kcentergreedy import k_center_greedy
from .uncertainty import Uncertainty
from .methods_utils import euclidean_dist
from torch.utils.data.dataloader import DataLoader
import torch
from sklearn.mixture import GaussianMixture

class ACL_Margin(CoresetMethod):
    def __init__(self, dst_train, strategy, fraction=0.5, mem=None, already_selected=[], index=None, **kwargs):
        super().__init__(dst_train, fraction)
        self.n_train = len(dst_train)
        self.strategy = strategy
        self.mem = mem
        self.data = dst_train
        self.already_selected = np.array(already_selected)
        
        if index is not None:
            assert len(dst_train) == len(index)
            self.index = index
        else:
            self.index = np.arange(len(dst_train))

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

    def select(self, **kwargs):
        if len(self.mem)==0:
            #for first stage, sample randomly
            selection_result = np.random.choice(np.arange(self.n_train), self.coreset_size, replace=False)
            return self.index[selection_result]

        df_mem, df_data = self.construct_matrix()
        # import pdb;pdb.set_trace()
        prototypes = self.generate_gaussian_prototypes(df_mem['target'].unique(), df_mem)
        preds, threshold = self.map_prototype(df_data['embs'].to_list(), prototypes, thresh_sigma=thresh_sigma)
        df_data['llh'] = np.max(np.array(preds).T, axis=1)
        df_data['prototype'] = np.argmax(np.array(preds).T, axis=1)
        group = df_data.apply(lambda x: 1 if x['llh'] > threshold[x['prototype']] else 0, axis=1)
        df_original = df_data[~group.astype(bool)].copy()
        df_expressible = df_data[group.astype(bool)].copy()
        print(f'original: {len(df_original)}')
        print(f'expressible: {len(df_expressible)}')
        assert len(df_expressible) == len(np.where(group)[0])

        original_budget = round(self.coreset_size * (len(df_original)/len(df_data)))
        expressible_budget = self.coreset_size-original_budget
        
        selection_result = np.array([], dtype=np.int32)
        if original_budget==0:
            # assert len(df_expressible)==len(df_data)
            expressible_selection = Uncertainty(self.dst_train[np.where(group)[0].tolist()], self.strategy, fraction=self.fraction, index=df_expressible.index, selection_method='Margin').select()
            selection_result = np.append(selection_result, expressible_selection)
        elif expressible_budget == 0:
            # assert len(df_original) == len(df_data)
            original_selection = k_center_greedy(
                np.vstack(df_original['embs'].to_list()),
                budget=original_budget,
                metric=euclidean_dist,
                index=df_original.index,
                device=self.strategy.device,
                already_selected = self.already_selected,
            )
            selection_result = np.append(selection_result, original_selection)
        else:
            expressible_selection = Uncertainty(self.dst_train[np.where(group)[0].tolist()], self.strategy, fraction=expressible_budget/len(df_expressible), index=df_expressible.index, selection_method='Margin').select()
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
        print(f'total query: {len(selection_result)}')
        return self.index[selection_result]
