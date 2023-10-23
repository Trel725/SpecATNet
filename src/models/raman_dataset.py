import numpy as np
import torch.utils.data as tdata
import torch
import pandas as pd
from sklearn.decomposition import NMF
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def randrows(A, size):
    return A[np.random.randint(A.shape[0], size=size), :]


class RamanSpectraSeqDataset(tdata.Dataset):
    def __init__(self, spectra: pd.DataFrame, labels: pd.DataFrame, groups: pd.DataFrame,
                 balance_groups=None, seq_len=4, min_seq_len=4):
        self.spectra = spectra
        self.labels = labels
        assert len(spectra) == len(labels)
        self.groups = groups
        self.gr_unique = np.unique(groups)
        self.build_x_dict()
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.curr_seq_len = seq_len
        # this defines groups to be used balancing, i.e. every group from
        # balance groups would have the same probability
        self.balance_groups = groups if balance_groups is None else balance_groups
        assert len(self.balance_groups) == len(self.groups)
        self.build_balance_dict()

    def build_balance_dict(self):
        """needed to perform class balancing"""
        self.balance_dict = {}
        for a_, b_ in zip(self.balance_groups, self.groups):
            if a_ not in self.balance_dict:
                self.balance_dict[a_] = [b_]
            else:
                self.balance_dict[a_].append(b_)
        for key in self.balance_dict:
            self.balance_dict[key] = list(set(self.balance_dict[key]))
        self.balance_gr_unique = np.array(list(self.balance_dict.keys()))

    def build_x_dict(self):
        self.x_dict = {}
        for idx, group in enumerate(self.gr_unique):
            label = self.labels[self.groups == group].iloc[0]
            self.x_dict[group] = dict(data=self.spectra[self.groups == group].values,
                                      label=np.squeeze(label),
                                      size=(self.groups == group).sum())

    def on_batch_end(self):
        self.curr_seq_len = np.random.randint(self.min_seq_len, self.seq_len+1)

    def __len__(self):
        return self.spectra.shape[0]

    def __getitem__(self, idx):
        seq_len = self.curr_seq_len
        idx = np.random.choice(self.balance_gr_unique)
        gr = np.random.choice(self.balance_dict[idx])
        x = randrows(self.x_dict[gr]['data'], seq_len)
        y = self.x_dict[gr]['label']
        if type(y) == np.float64:
            y = [y]
        return torch.Tensor(x[:, None, :]), torch.Tensor(y)


class NMFRamanSpectraSeqDataset(RamanSpectraSeqDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nmf = NMF(n_components=5)

    def __getitem__(self, idx):
        seq_len = self.curr_seq_len
        idx = np.random.choice(self.balance_gr_unique)
        gr = np.random.choice(self.balance_dict[idx])
        x = randrows(self.x_dict[gr]['data'], seq_len)
        y = self.x_dict[gr]['label']
        if type(y) == np.float64:
            y = [y]
        x_ = x - x.min(axis=1)[:, None]
        self.nmf.fit_transform(x_)
        x = self.nmf.components_
        return torch.Tensor(x[:, None, :]), torch.Tensor(y)
