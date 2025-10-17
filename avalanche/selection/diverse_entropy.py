import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from .coresetmethod import CoresetMethod
from torch.utils.data import DataLoader


class kth_Uncertainty(CoresetMethod):
    def __init__(self, dst_train, strategy, fraction, selection_method='Entropy', index=None, batch_size=None, **kwargs):
        super().__init__(dst_train, fraction)
        self.strategy = strategy
        self.selection_method = selection_method
        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = self.strategy.train_mb_size
        if index is not None:
            assert len(dst_train) == len(index)
            self.index = index
        else:
            self.index = np.arange(len(dst_train))

    def rank_uncertainty(self, temp=1.0):
        device = self.strategy.device
        model = self.strategy.model
        model.eval()
        with torch.no_grad():
            train_loader = DataLoader(self.dst_train, batch_size=self.batch_size)
            scores = np.array([])
            for i, (inputs, _, _) in tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True):
                if self.selection_method == "LeastConfidence":
                    scores = np.append(scores, model(inputs.to(device)).max(axis=1).values.cpu().numpy())
                elif self.selection_method == "Entropy":
                    preds = torch.nn.functional.softmax(model(inputs.to(device)), dim=1).cpu().numpy()
                    scores = np.append(scores, (np.log(preds + 1e-6) * preds).sum(axis=1))
                elif self.selection_method == 'Margin':
                    preds = torch.nn.functional.softmax(model(inputs.to(device)), dim=1)
                    preds_argmax = torch.argmax(preds, dim=1)
                    max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                    preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                    preds_sub_argmax = torch.argmax(preds, dim=1)
                    scores = np.append(scores, (max_preds - preds[torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy())
                    
        model.train()
        return scores
    
    def select(self, **kwargs):
        scores = self.rank_uncertainty()
        selection_result = np.argsort(scores)[::len(scores)//self.coreset_size]
        return self.index[selection_result]
        







