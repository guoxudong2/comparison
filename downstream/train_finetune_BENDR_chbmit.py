import random
import os
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
# torch.set_float32_matmul_precision('medium' )#| 'high'
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
import math
from functools import partial
import numpy as np
import random
import os 
import tqdm
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
from pytorch_lightning import loggers as pl_loggers

from Modules.models.dn3_ext import BENDR, ConvEncoderBENDR

from datasets.downstream.CHB_MIT_Scalp_EEG.chbmit_dataset import CHBMITDataset
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import random_split, DataLoader

import torch.nn.functional as F
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(7)


from utils import *
from utils_eval import get_metrics


use_channels_names=[
            'EEG-1', 'EEG-3',
    'EEG-0', 'EEG-1', 'EEG-Fz', 'EEG-3', 'EEG-4',
    'EEG-5', 'EEG-C3', 'EEG-Cz', 'EEG-C4', 'EEG-8',
    'EEG-9', 'EEG-10', 'EEG-Pz', 'EEG-12', 'EEG-13',
            'EEG-14', 'EEG-15'
    ]
ch_names = ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz', 
                'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 
                'EEG-15', 'EEG-16', 'EOG-left', 'EOG-central', 'EOG-right']
# -- get channel id by use chan names
choice_channels = []
for ch in use_channels_names:
    choice_channels.append([x.lower().strip('.') for x in ch_names].index(ch.lower()))
use_channels = choice_channels
# use_channels = None

class ChannelAdapter(nn.Module):
    """
    输入: [B, 23, T]
    输出: [B, 19, T]
    通过 1x1 卷积 + BN + ReLU 实现learnable通道投影
    """
    def __init__(self, in_ch=23, out_ch=19):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        # x: [B, in_ch, T]
        x = self.conv(x)    # -> [B, out_ch, T]
        x = self.bn(x)
        x = F.relu(x)
        return x

class LitBENDR(pl.LightningModule):

    def __init__(self):
        super().__init__()

        encoder = ConvEncoderBENDR(20, encoder_h=512, dropout=0., projection_head=False)
        encoder.load("Modules/models/encoder.pt")
        
        self.model = encoder
        self.scale_param    = torch.nn.Parameter(torch.tensor(1.))
        #self.linear_probe   = torch.nn.Linear(5632, 4)
        #self.linear_probe = torch.nn.Linear(1536, 4)
        self.linear_probe = torch.nn.Linear(1024, 2)
        
        self.drop           = torch.nn.Dropout(p=0.10)
        
        self.loss_fn        = torch.nn.CrossEntropyLoss()

        self.running_scores = {"train": [], "valid": [], "test": []}
        self.is_sanity = True
        self.channel_adapter = ChannelAdapter(in_ch=23, out_ch=19)

    def mixup_data(self, x, y, alpha=None):
        # 随机选择另一个样本来混合数据
        lam = torch.rand(1).to(x) if alpha is None else alpha
        lam = torch.max(lam, 1 - lam)

        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]

        return mixed_x, mixed_y
    
    def forward(self, x):
        if self.global_step == 0 and self.global_rank == 0:
            print(f"[Debug] input x.shape = {x.shape}")  # (B, C, T)
        x = self.channel_adapter(x)
        x = torch.cat([x, self.scale_param.repeat((x.shape[0], 1, x.shape[-1]))], dim=-2)
        h = self.model(x)
        
        h = h.flatten(1)
        h = self.drop(h)
        pred = self.linear_probe(h)
        return x, pred

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        #soft_labels = F.one_hot(y.long(), num_classes=2).float()
        #x,y = self.mixup_data(x,soft_labels)
        
        label = y
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==torch.argmax(label, dim=-1))*1.0).mean()
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_max', x.max(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_min', x.min(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_std', x.std(), on_epoch=True, on_step=False, sync_dist=True)
        
        return loss
        
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity = False
            return super().on_validation_epoch_end()

        label, y_score = [], []
        for x, y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)

        print(label.shape, y_score.shape)

        # logits -> 二分类概率
        y_prob = torch.softmax(y_score, dim=-1)[:, 1]

        metrics = ["accuracy", "balanced_accuracy", "precision",
                   "recall", "cohen_kappa", "f1", "roc_auc", "specificity"]
        results = get_metrics(y_prob.cpu().numpy(), label.cpu().numpy(), metrics, True)

        for key, value in results.items():
            self.log('valid_' + key, value, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False)
        
        self.running_scores["valid"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))
        return loss
    
    def test_step(self, batch, *args, **kwargs):
        
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()

        # Logging to TensorBoard by default
        self.log('test_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)

        self.running_scores["test"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))
        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            list([self.scale_param])+
            list(self.model.parameters())+
            list(self.linear_probe.parameters()),
            weight_decay=0.01)#
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
      
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )

    def on_test_epoch_start(self) -> None:
        self.running_scores["test"] = []
        return super().on_test_epoch_start()

    def on_test_epoch_end(self) -> None:
        label, y_score = [], []
        for x, y in self.running_scores["test"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)

        # logits -> 二分类概率
        y_prob = torch.softmax(y_score, dim=-1)[:, 1]

        pred_class = (y_prob >= 0.5).long()

        print(f"[Test] label.shape: {label.shape}, y_score.shape: {y_score.shape}")
        print("Predicted class distribution:", torch.bincount(pred_class))
        print("True class distribution:", torch.bincount(label))

        metrics = ["accuracy", "balanced_accuracy", "precision",
                   "recall", "cohen_kappa", "f1", "roc_auc", "specificity"]
        results = get_metrics(y_prob.cpu().numpy(), label.cpu().numpy(), metrics, True)

        for key, value in results.items():
            self.log('test_' + key, value, on_epoch=True, on_step=False, sync_dist=True)

        return super().on_test_epoch_end()


'''data_path = "../datasets/downstream/Data/BCIC_2a_0_38HZ"
# load configs
#for sub in range(1,10):
for sub in range(1, 2):
    train_dataset,valid_dataset,test_dataset = get_data(sub,data_path,1,True, use_channels=use_channels)
        
    global max_epochs
    global steps_per_epoch
    global max_lr

    batch_size=64

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    
    #max_epochs = 100
    max_epochs = 1
    steps_per_epoch = math.ceil(len(train_loader) )
    max_lr = 1e-5

    # init model
    model = LitBENDR()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]
    
    trainer = pl.Trainer(accelerator='cuda',
                         max_epochs=max_epochs, 
                         callbacks=callbacks,
                         logger=[pl_loggers.TensorBoardLogger('./logs/', name="BENDR_BCIC2A_tb", version=f"subject{sub}"), 
                                 pl_loggers.CSVLogger('./logs/', name="BENDR_BCIC2A_csv")])

    #trainer.fit(model, train_loader, test_loader, ckpt_path='last')
    trainer.fit(model, train_loader, valid_loader, ckpt_path='last')
    trainer.test(model, dataloaders=test_loader)'''

'''import math
x_path = "../datasets/downstream/CHB_MIT_Scalp_EEG/processed_data/shifted/chb01_X_shift.npy"
y_path = "../datasets/downstream/CHB_MIT_Scalp_EEG/processed_data/shifted/chb01_y_shift.npy"

X = np.load(x_path)
y = np.load(y_path)

print("Data shape:", X.shape, y.shape)
print("Label distribution:", np.bincount(y))

dataset = CHBMITDataset(
    x_path=x_path,
    y_path=y_path,
    use_avg=True,
    scale_div=10.0,
    interpolation_len=256
)

# 拆分 train/val
train_len = int(len(dataset) * 0.8)
val_len = len(dataset) - train_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])

# 构造 Weighted Sampler
train_labels = [dataset.y[i].item() for i in train_ds.indices]
counts = np.bincount(train_labels)
weight_neg = 1.0 / counts[0]
weight_pos = 1.0 / counts[1]
sample_weights = [weight_pos if l == 1 else weight_neg for l in train_labels]
train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=32, sampler=train_sampler, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

max_epochs = 100
steps_per_epoch = math.ceil(len(train_loader) )
max_lr = 1e-5

# init model
model = LitBENDR()

lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
callbacks = [lr_monitor]

trainer = pl.Trainer(accelerator='cuda',
                     max_epochs=max_epochs,
                     callbacks=callbacks,
                     logger=[pl_loggers.TensorBoardLogger('./logs/', name="BENDR_CHBMIT_tb", version=f"chb01"),
                             pl_loggers.CSVLogger('./logs/', name="BENDR_CHBMIT_csv")])

trainer.fit(model, train_loader, ckpt_path='last')
trainer.test(model, dataloaders=val_loader)'''

import os, glob, math, argparse, yaml, random
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint)
from sklearn.utils.class_weight import compute_class_weight

def make_loaders(x_path: str, y_path: str, cfg: dict):
    ds = CHBMITDataset(
        x_path,
        y_path,
        use_avg        = True,
        scale_div      = 10.0,
        interpolation_len = cfg['data']['interpolation_len']
    )

    n_val = int(len(ds) * cfg['data']['val_ratio'])
    tr_ds, va_ds = random_split(ds, [len(ds) - n_val, n_val])

    # class weight & WeightedRandomSampler
    y_tr   = np.array([ds.y[i] for i in tr_ds.indices])
    c_w    = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    pos, neg = y_tr.sum(), len(y_tr) - y_tr.sum()
    samp_w   = [1/pos if l else 1/neg for l in y_tr]

    sampler = WeightedRandomSampler(samp_w, len(samp_w), replacement=True)
    tr_ld   = DataLoader(tr_ds, batch_size=cfg['data']['batch'],
                         sampler=sampler, num_workers=0)
    #tr_ld = DataLoader(tr_ds, batch_size=cfg['data']['batch'], num_workers=0)
    va_ld   = DataLoader(va_ds, batch_size=cfg['data']['batch'],
                         shuffle=False,  num_workers=0)
    return tr_ld, va_ld, c_w

def load_cfg(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))
    cfg['data'].pop('x_path', None)
    cfg['data'].pop('y_path', None)
    return cfg

global max_epochs
global steps_per_epoch
global max_lr
max_epochs = 100
max_lr = 1e-5

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='../conf/mybendr.yaml')
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)

    shift_dir = cfg['data']['shift_dir']
    x_paths = sorted(glob.glob(os.path.join(shift_dir, '*_X_shift.npy')))
    assert x_paths, f'No *_X_shift.npy under {shift_dir}'

    for x_path in x_paths:
        pid = os.path.basename(x_path).split('_')[0]  # chb01
        '''if pid=='chb01':
            print(f'pid={pid}')
            continue'''
        y_path = x_path.replace('_X_shift.npy', '_y_shift.npy')
        if not os.path.exists(y_path):
            print(f'[Skip] {pid}: missing y file');
            continue

        print(f"\n===== Now training on patient {pid} =====")

        # ---------------- Data -----------------
        tr_ld, va_ld, cls_w = make_loaders(x_path, y_path, cfg)
        x, y = next(iter(tr_ld))
        print("Batch shape =", x.shape)  # (B, C, T)

        steps_per_epoch = math.ceil(len(tr_ld))
        model = LitBENDR()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        '''model.loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(cls_w, dtype=torch.float32, device=device)
        )'''

        # ---------------- 日志 -----------------
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir='./logs_bendr/', name='BENDR_CHBMIT_tb', version=pid)
        csv_logger = pl_loggers.CSVLogger(
            save_dir='./logs_bendr/', name='BENDR_CHBMIT_csv', version=pid)

        # ---------------- 回调 -----------------
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks = [lr_monitor]  # + [ckpt_cb]

        # ---------------- Trainer --------------
        trainer = pl.Trainer(
            max_epochs=cfg['train']['epochs'],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=[cfg['train']['gpu']] if torch.cuda.is_available() else 1,
            callbacks=callbacks,
            logger=[tb_logger, csv_logger],
            log_every_n_steps=10,
            enable_checkpointing=False  # 若用 ckpt_cb → True
        )

        print(f'\n=== {pid}: {len(tr_ld.dataset)} train / {len(va_ld.dataset)} val ===')
        trainer.fit(model, tr_ld, va_ld)
        trainer.test(model, va_ld)
        del model
        torch.cuda.empty_cache()
