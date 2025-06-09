import random
import os
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import pytorch_lightning as pl
from functools import partial
import numpy as np
import random
import os
import tqdm
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve
from datasets.downstream.CHB_MIT_Scalp_EEG.chbmit_dataset import CHBMITDataset
import os, glob, random, yaml, argparse, math
import numpy as np, torch, pytorch_lightning as pl
from dotmap import DotMap
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from pytorch_lightning import loggers as pl_loggers

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(7)

from Modules.models.EEGPT_mcae import EEGTransformer
from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
from utils import temporal_interpolation
from sklearn import metrics
from utils_eval import get_metrics

use_channels_names = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3',
                      'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
                      'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ',
                      'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8-1']

class ChannelAdapter(nn.Module):
    """
    输入: [B, 23, T]
    输出: [B, 19, T]
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

class LitEEGPTCausal(pl.LightningModule):

    def __init__(self, max_lr: float, steps_per_epoch: int, max_epochs: int,
                 load_path="../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"):
        super().__init__()
        self.chans_num = len(use_channels_names)
        # init model
        target_encoder = EEGTransformer(
            #img_size=[19, 2 * 256],
            img_size=[23, 2 * 256],
            patch_size=32 * 2,
            patch_stride=32,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.max_lr=max_lr
        self.steps_per_epoch=steps_per_epoch
        self.max_epochs=max_epochs
        self.target_encoder = target_encoder
        #self.chans_id = target_encoder.prepare_chan_ids(use_channels_names)

        #self.channel_adapter = ChannelAdapter(in_ch=23, out_ch=19)
        self.channel_adapter = ChannelAdapter(in_ch=23, out_ch=23)
        self.chans_id = torch.arange(23)

        # -- load checkpoint
        pretrain_ckpt = torch.load(load_path)

        target_encoder_stat = {}
        for k, v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]] = v

        self.target_encoder.load_state_dict(target_encoder_stat)

        #self.chan_conv = Conv1dWithConstraint(19, self.chans_num, 1, max_norm=1)
        #self.chan_conv = Conv1dWithConstraint(23, self.chans_num, 1, max_norm=1)注掉了
        #self.chan_conv = Conv1dWithConstraint(23, 58, 1, max_norm=1)

        #self.linear_probe1 = LinearWithConstraint(2048, 16, max_norm=1)
        #self.linear_probe2 = LinearWithConstraint(240, 2, max_norm=0.25)
        self.linear_probe1 = nn.Sequential(
            LinearWithConstraint(2048, 256, max_norm=1),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.linear_probe2 = LinearWithConstraint(256 * 15, 2, max_norm=0.25)

        self.drop = torch.nn.Dropout(p=0.3)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train": [], "valid": [], "test": []}
        self.is_sanity = True
        self.metric_buf = {"y": [], "p": []}
        self.save_hyperparameters()

    def forward(self, x):
        B, C, T = x.shape
        '''x = x / 10
        x = x - x.mean(dim=-2, keepdim=True)'''
        x = temporal_interpolation(x, 2 * 256)
        x = self.channel_adapter(x)#我加的
        #x = self.chan_conv(x)注掉
        self.target_encoder.eval()
        z = self.target_encoder(x, self.chans_id.to(x))

        h = z.flatten(2)
        h = self.linear_probe1(self.drop(h))
        h = h.flatten(1)
        h = self.linear_probe2(h)

        return x, h

    '''def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()

        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds == label) * 1.0).mean()
        y_score = logit.clone().detach().cpu()
        y_score = torch.softmax(y_score, dim=-1)[:, 1]
        self.running_scores["train"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))
        # rocauc = metrics.roc_auc_score(label.clone().detach().cpu(), y_score)
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"] = []
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

        metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "cohen_kappa", "f1", "roc_auc"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, True)

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
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds == label) * 1.0).mean()

        y_score = logit
        y_score = torch.softmax(y_score, dim=-1)[:, 1]
        self.running_scores["valid"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))

        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False)

        return loss

    def on_train_epoch_start(self) -> None:
        self.running_scores["train"] = []
        return super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:

        label, y_score = [], []
        for x, y in self.running_scores["train"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        rocauc = metrics.roc_auc_score(label, y_score)
        self.log('train_rocauc', rocauc, on_epoch=True, on_step=False)
        return super().on_train_epoch_end()

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
        rocauc = metrics.roc_auc_score(label, y_score)
        self.log('test_rocauc', rocauc, on_epoch=True, on_step=False)

        return super().on_test_epoch_end()

    def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()

        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds == label) * 1.0).mean()
        y_score = logit
        y_score = torch.softmax(y_score, dim=-1)[:, 1]
        self.running_scores["test"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))
        # Logging to TensorBoard by default
        self.log('test_loss', loss, on_epoch=True, on_step=False)
        self.log('test_acc', accuracy, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            list(self.channel_adapter.parameters()) + #我加的
            #list(self.chan_conv.parameters()) +
            list(self.linear_probe1.parameters()) +
            list(self.linear_probe2.parameters()),
            weight_decay=0.01)  #

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch,
                                                           epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler,  # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1,  # The frequency of the scheduler
            'monitor': 'val_loss',  # Metric for `ReduceLROnPlateau` to monitor
            'strict': True,  # Whether to crash the training if `monitor` is not found
            'name': None,  # Custom name for `LearningRateMonitor` to use
        }

        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )'''

    '''def _step(self, batch, stage: str):
        x, y = batch
        _, logits = self.forward(x)
        loss = self.loss_fn(logits, y.long())

        prob = torch.softmax(logits,dim=-1)[:,1]
        self.metric_buf["y"].append(y.detach().cpu())
        self.metric_buf["p"].append(prob.detach().cpu())

        preds = (prob > 0.5).int()
        acc = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_epoch=True, prog_bar=True)
        return loss'''

    '''我的错误写法
        def _step(self, batch, stage):
        x, y = batch
        _, logits = self.forward(x)
        loss = self.loss_fn(logits, y.long())

        prob = torch.softmax(logits, dim=-1)[:,1].detach().cpu()  # ← 这里
        self.metric_buf["y"].append(y.detach().cpu())
        self.metric_buf["p"].append(prob)

        preds = prob.argmax(dim=-1)
        acc = (preds == y.cpu()).float().mean()
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_epoch=True, prog_bar=True)
        return loss'''

    def _step(self, batch, stage):
        x, y = batch
        _, logits = self.forward(x)
        #print(f'----------yyyyyy:{y}')
        #print(f'----------logits:{logits}')
        loss = self.loss_fn(logits, y.long())

        self.metric_buf["y"].append(y.detach().cpu())
        self.metric_buf["p"].append(logits.detach().cpu())  # 保存 logits，而不是 softmax 后

        #preds = logits.argmax(dim=-1).cpu()
        preds = torch.argmax(logits,dim=-1).cpu()
        acc = (preds == y.cpu()).float().mean()
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    #def validation_step(self, batch, batch_idx):
    #    return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    '''def _epoch_end(self, stage: str):
        """
        每个 epoch 结束时聚合指标
        ---------------------------------------------------------------
        - 预测标签用 argmax
        - AUC 用 class-1 概率
        """
        print(stage, "buffer len =", len(self.metric_buf["y"]))
        if len(self.metric_buf["y"]) == 0:
            return

        ys = torch.cat(self.metric_buf["y"]).numpy()  # (N,)
        ps = torch.cat(self.metric_buf["p"]).numpy()  # (N,2)

        # --- 离散预测 & class-1 概率 ---
        preds = np.argmax(ps, axis=1)  # argmax → 0/1
        prob_class1 = ps[:, 1]  # 取第 1 类概率

        # --- 计算指标 ---
        acc = metrics.accuracy_score(ys, preds)
        auc = metrics.roc_auc_score(ys, prob_class1) if len(np.unique(ys)) == 2 else 0.0
        prec = metrics.precision_score(ys, preds, zero_division=0)
        rec = metrics.recall_score(ys, preds, zero_division=0)  # sensitivity
        f1 = metrics.f1_score(ys, preds, zero_division=0)
        kappa = metrics.cohen_kappa_score(ys, preds)

        tn, fp, fn, tp = metrics.confusion_matrix(ys, preds, labels=[0, 1]).ravel()
        spec = tn / (tn + fp + 1e-8)  # specificity

        # --- log 到 Lightning ---
        self.log_dict({
            f"{stage}_acc": acc,
            f"{stage}_auc": auc,
            f"{stage}_precision": prec,
            f"{stage}_sensitivity": rec,
            f"{stage}_specificity": spec,
            f"{stage}_f1": f1,
            f"{stage}_kappa": kappa,
        }, prog_bar=True)

        unique_pred, cnt_pred = np.unique(preds, return_counts=True)
        unique_true, cnt_true = np.unique(ys, return_counts=True)
        print(f"preds: {dict(zip(unique_pred, cnt_pred))}  "
              f"ys: {dict(zip(unique_true, cnt_true))}")

        # 清空缓冲区
        self.metric_buf = {"y": [], "p": []}'''

    '''def _epoch_end(self, stage: str):
        if not self.metric_buf["y"]:  # 兼容 sanity check
            return
        y_true = torch.cat(self.metric_buf["y"]).numpy()
        logits = torch.cat(self.metric_buf["p"])  # (N,)
        preds = logits.argmax(dim=-1).cpu().numpy()
        prob1 = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        # --- metrics ---
        acc = metrics.accuracy_score(y_true, preds)
        auc = metrics.roc_auc_score(y_true, prob1) if len(np.unique(y_true)) == 2 else .0
        prec = metrics.precision_score(y_true, preds, zero_division=0)
        rec = metrics.recall_score(y_true, preds, zero_division=0)
        f1 = metrics.f1_score(y_true, preds, zero_division=0)
        kappa = metrics.cohen_kappa_score(y_true, preds)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        spec = tn / (tn + fp + 1e-8)

        self.log_dict({
            f"{stage}_acc": acc, f"{stage}_auc": auc,
            f"{stage}_precision": prec, f"{stage}_sensitivity": rec,
            f"{stage}_specificity": spec, f"{stage}_f1": f1, f"{stage}_kappa": kappa},
            prog_bar=True)

        self.metric_buf = {"y": [], "p": []}'''

    def _epoch_end(self, stage: str):
        if not self.metric_buf["y"]:  # 兼容 sanity check
            return
        y_true = torch.cat(self.metric_buf["y"]).numpy()
        logits = torch.cat(self.metric_buf["p"])  # (N,)
        preds = logits.argmax(dim=-1).cpu().numpy()
        prob1 = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        # --- metrics ---
        acc = metrics.accuracy_score(y_true, preds)
        auc = metrics.roc_auc_score(y_true, prob1) if len(np.unique(y_true)) == 2 else .0
        prec = metrics.precision_score(y_true, preds, zero_division=0)
        rec = metrics.recall_score(y_true, preds, zero_division=0)
        f1 = metrics.f1_score(y_true, preds, zero_division=0)
        kappa = metrics.cohen_kappa_score(y_true, preds)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        spec = tn / (tn + fp + 1e-8)

        self.log_dict({
            f"{stage}_acc": acc, f"{stage}_auc": auc,
            f"{stage}_precision": prec, f"{stage}_sensitivity": rec,
            f"{stage}_specificity": spec, f"{stage}_f1": f1, f"{stage}_kappa": kappa},
            prog_bar=True)

        self.metric_buf = {"y": [], "p": []}

    def on_train_epoch_end(self):
        self._epoch_end('train')

    def on_validation_epoch_end(self):
        self._epoch_end('val')

    def on_test_epoch_end(self):
        self._epoch_end('test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.channel_adapter.parameters()) + #我加的
            #list(self.chan_conv.parameters()) +
            list(self.linear_probe1.parameters()) +
            list(self.linear_probe2.parameters()),
            weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.max_lr, steps_per_epoch=self.steps_per_epoch,
            epochs=self.trainer.max_epochs, pct_start=0.2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

# load configs
# -- LOSO
#from utils import *
#import math

#seed_torch(9)
#path = "../datasets/downstream"

#global max_epochs
#global steps_per_epoch
#global max_lr

#batch_size = 64
#max_epochs = 100

#Folds = {
#    1: ([12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26], [1, 3, 4, 5, 8, 9, 10, 15, 19, 25]),
#    2: ([2, 6, 7, 11, 17, 18, 20, 21, 22, 23, 24, 26], [1, 3, 4, 5, 8, 9, 10, 15, 19, 25]),
#    3: ([2, 6, 7, 11, 12, 13, 14, 16, 22, 23, 24, 26], [1, 3, 4, 5, 8, 9, 10, 15, 19, 25]),
#    4: ([2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21], [1, 3, 4, 5, 8, 9, 10, 15, 19, 25]),
#}

#for k, v in Folds.items():
    # init model
#    model = LitEEGPTCausal()
#    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
#    callbacks = [lr_monitor]
#    training = read_kaggle_ern_train(path, subjects=v[0])
#    validation = read_kaggle_ern_test(path, subjects=v[1])
#    train_loader = torch.utils.data.DataLoader(training, batch_size=batch_size, num_workers=0, shuffle=True)
#    valid_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size, num_workers=0, shuffle=False)

#    steps_per_epoch = math.ceil(len(train_loader))
#    max_lr = 4e-4
#    trainer = pl.Trainer(accelerator='cuda',
#                         max_epochs=max_epochs,
#                         callbacks=callbacks,
#                         enable_checkpointing=False,
#                         logger=[pl_loggers.TensorBoardLogger('./logs/', name="EEGPT_ERN_tb", version=f"fold{k}"),
#                                 pl_loggers.CSVLogger('./logs/', name="EEGPT_ERN_csv")])

#   trainer.fit(model, train_loader, valid_loader, ckpt_path='last')'''

def load_cfg(cfg_path: str) -> DotMap:
    cfg = DotMap(yaml.safe_load(open(cfg_path, 'r', encoding='utf-8')))
    cfg.data.pop('x_path', None)
    cfg.data.pop('y_path', None)
    return cfg

#没有分层，即训练集和验证集的preictal/interictal可能很不一样
'''def make_loaders(x_path, y_path, cfg):
    ds = CHBMITDataset(x_path, y_path,
                       use_avg=True, scale_div=10.0,
                       interpolation_len=cfg.data.interpolation_len)
    n_val = int(len(ds)*cfg.data.val_ratio)
    tr_ds, va_ds = random_split(ds, [len(ds)-n_val, n_val])

    y_tr = np.array([ds.y[i] for i in tr_ds.indices])
    class_w = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr).astype(float)

    pos, neg = y_tr.sum(), len(y_tr)-y_tr.sum()
    samp_w = [1/pos if l else 1/neg for l in y_tr]
    print("Train class counts:", np.bincount(y_tr))

    sampler = WeightedRandomSampler(samp_w, len(samp_w), replacement=True)

    tr_ld = DataLoader(tr_ds, batch_size=cfg.data.batch, sampler=sampler, num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=cfg.data.batch, shuffle=False, num_workers=0)
    return tr_ld, va_ld, class_w'''

def make_loaders(
    x_path: str,
    y_path: str,
    cfg
) -> tuple[DataLoader, DataLoader]:
    ds = CHBMITDataset(
        x_path, y_path,
        use_avg=True,
        scale_div=10.0,
        interpolation_len=cfg.data.interpolation_len
    )
    print(f'x_path:{x_path}')
    print(f'y_path:{y_path}')
    y = ds.y.numpy()  # shape (N,)
    print(f'yyyyy y.shape:{y.shape}')
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=cfg.data.val_ratio,
        random_state=42
    )
    train_idx, val_idx = next(
        splitter.split(np.zeros(len(y)), y)
    )

    tr_ds = Subset(ds, train_idx)
    va_ds = Subset(ds, val_idx)

    y_tr = y[train_idx]
    pos = y_tr.sum()
    neg = len(y_tr) - pos
    sample_weights = [ (1/pos if lab==1 else 1/neg) for lab in y_tr ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    '''tr_ld = DataLoader(
        tr_ds,
        batch_size=cfg.data.batch,
        sampler=sampler,
        num_workers=0
    )'''
    tr_ld = DataLoader(
        tr_ds,
        batch_size=cfg.data.batch,
        num_workers=0
    )
    va_ld = DataLoader(
        va_ds,
        batch_size=cfg.data.batch,
        shuffle=False,
        num_workers=0
    )

    return tr_ld, va_ld

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='../conf/eegpt_causal.yaml')
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)

    #x_paths = sorted(glob.glob(os.path.join(cfg.data.shift_dir, '*_X_shift.npy')))
    #assert x_paths, f'No *_X_shift.npy under {cfg.data.shift_dir}'

    x_paths = sorted(glob.glob(os.path.join(cfg.data.shift_dir, '*_X.npy')))
    print(f'x_paths: {x_paths}')
    assert x_paths, f'No *_X.npy under {cfg.data.shift_dir}'

    for x_path in x_paths:
        pid    = os.path.basename(x_path).split('_')[0]          # chb01
        #y_path = x_path.replace('_X_shift.npy', '_y_shift.npy')
        y_path = x_path.replace('_X.npy', '_y.npy')
        if not os.path.exists(y_path):
            print(f'[Skip] {pid}: missing y file'); continue

        if pid != 'chb01':
            print(f'{pid} not chb04, continue')
            continue
        # ---------------- Data -----------------
        tr_ld, va_ld = make_loaders(x_path, y_path, cfg)

        steps_per_epoch = math.ceil(len(tr_ld))
        # ---------------- Model ----------------
        model = LitEEGPTCausal(max_lr=float(cfg.train.lr), steps_per_epoch=steps_per_epoch,max_epochs=cfg.train.epochs)   # 如有预训练 ckpt
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------------- Logger ---------------
        tb_logger  = pl_loggers.TensorBoardLogger(
            './logs_eegpt/', name="EEGPT_CHBMIT_tb",  version=pid)
        csv_logger = pl_loggers.CSVLogger(
            './logs_eegpt/', name="EEGPT_CHBMIT_csv", version=pid)

        # ---------------- Trainer --------------
        trainer = pl.Trainer(
            max_epochs=cfg.train.epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=[cfg.train.gpu] if torch.cuda.is_available() else 1,
            logger=[tb_logger, csv_logger],
            log_every_n_steps=10,
            enable_checkpointing=False
        )

        print(f'\n=== {pid}: {len(tr_ld.dataset)} train / {len(va_ld.dataset)} val ===')

        #-----------------------
        all_tr_y = []
        for _, y in tr_ld:
            all_tr_y.append(y.numpy())
        all_tr_y = np.concatenate(all_tr_y, axis=0)

        all_va_y = []
        for _, y in va_ld:
            all_va_y.append(y.numpy())
        all_va_y = np.concatenate(all_va_y, axis=0)

        print("Train label counts:", np.bincount(all_tr_y))
        print("Valid label counts:", np.bincount(all_va_y))
        #-----------------------
        trainer.fit(model, tr_ld, va_ld)
        trainer.test(model, va_ld)



