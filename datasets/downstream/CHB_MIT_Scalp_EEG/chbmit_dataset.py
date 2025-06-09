# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ======== Lightning ========
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from downstream.utils import temporal_interpolation

class CHBMITDataset(Dataset):
    """
    假设已从 CHB-MIT 提取的样本 X.shape = (N, C, T), y.shape = (N,)
    """
    def __init__(
        self,
        x_path: str,
        y_path: str,
        use_avg: bool = True,
        scale_div: float = 10.0,
        interpolation_len: int = None,
    ):
        # 读 .npy
        X = np.load(x_path)  # (N, C, T)
        y = np.load(y_path)  # (N,)

        # 转为tensor
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        # 可选：对每个通道减去均值
        if use_avg:
            # self.X.mean(dim=-2) => shape (N, C, 1)
            self.X = self.X - self.X.mean(dim=-2, keepdim=True)

        # 可选：简单缩放
        if scale_div is not None:
            self.X = self.X / scale_div

        # 若需要插值，可一次性在 __init__ 做，也可在 __getitem__ 动态做(速度会慢)。
        if interpolation_len is not None:
            # 假设你想把 T=60s*256=15360 压缩到 512 => 方便进网络
            new_data = []
            for i in range(len(self.X)):
                x_i = self.X[i]  # shape [C, T]
                x_i = temporal_interpolation(x_i, interpolation_len)
                new_data.append(x_i.unsqueeze(0))  # => [1, C, T']
            self.X = torch.cat(new_data, dim=0)  # => [N, C, T']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]