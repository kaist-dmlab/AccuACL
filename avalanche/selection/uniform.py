import numpy as np
from .coresetmethod import CoresetMethod


class Uniform(CoresetMethod):
    def __init__(self, dst_train, fraction=0.5, balance=False, replace=False, index=None, **kwargs):
        super().__init__(dst_train, fraction)
        self.balance = balance
        self.replace = replace
        self.n_train = len(dst_train)
        if index is not None:
            assert len(dst_train) == len(index)
            self.index = index
        else:
            self.index = np.arange(len(dst_train))

    def select_balance(self):
        """The same sampling proportions were used in each class separately."""
        self.index = np.array([], dtype=np.int64)
        all_index = np.arange(self.n_train)
        for c in range(self.num_classes):
            c_index = (self.dst_train.targets == c)
            self.index = np.append(self.index,
                                   np.random.choice(all_index[c_index], round(self.fraction * c_index.sum().item()),
                                                    replace=self.replace))
        return self.index

    def select_no_balance(self):
        self.index = np.random.choice(np.arange(self.n_train), round(self.n_train * self.fraction),
                                      replace=self.replace)
        return self.index

    def select(self, **kwargs):
        if self.balance:
            return self.index[self.select_balance()]
        else:
            return self.index[self.select_no_balance()]
