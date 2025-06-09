# -*- coding: utf-8 -*-
"""
Dataset：同时返回 (x, y, t0) 三元组，以便后续在 Lightning 里做 event-based 评价。
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from downstream.utils import temporal_interpolation  # 如果你需要插值，可以用这个

class CHBMITDatasetWithT0(Dataset):
    """
    假设已经用 preprocess_chbmit.py 生成了:
      X.npy: shape (N, C, T)
      y.npy: shape (N,)
      t0.npy: shape (N,)
    __getitem__ 返回 (x_i, y_i, t0_i)。
    """
    def __init__(self,
                 x_path: str,
                 y_path: str,
                 t0_path: str,
                 use_avg: bool = True,
                 scale_div: float = 10.0,
                 interpolation_len: int = None):
        """
        x_path: "processed_data/chb01_X.npy"
        y_path: "processed_data/chb01_y.npy"
        t0_path: "processed_data/chb01_t0.npy"
        """
        X = np.load(x_path, allow_pickle=False)  # (N, C, T)
        y = np.load(y_path, allow_pickle=False)  # (N,)
        t0 = np.load(t0_path, allow_pickle=False) # (N,)

        # 转为 tensor
        self.X = torch.tensor(X, dtype=torch.float32)  # [N, C, T]
        self.y = torch.tensor(y, dtype=torch.long)     # [N]
        self.t0 = torch.tensor(t0, dtype=torch.long)   # [N]
        self.training_flag = False
        # 去均值
        if use_avg:
            #self.X = self.X - self.X.mean(dim=-1, keepdim=True)
            self.X = self.X - self.X.mean(dim=-2, keepdim=True)
        # 缩放
        if scale_div is not None:
            self.X = self.X / scale_div

        # 插值到固定长度（可选）
        '''if interpolation_len is not None:
            new_data = []
            for i in range(len(self.X)):
                x_i = self.X[i]  # [C, T]
                x_i = temporal_interpolation(x_i, interpolation_len)  # [C, T_new]
                new_data.append(x_i.unsqueeze(0))
            self.X = torch.cat(new_data, dim=0)  # => [N, C, T_new]'''

    def __len__(self):
        return len(self.y)

    '''def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.t0[idx]'''

    def __getitem__(self, idx):
        x, y, t0 = self.X[idx].clone(), self.y[idx], self.t0[idx]
        return x, y, t0
