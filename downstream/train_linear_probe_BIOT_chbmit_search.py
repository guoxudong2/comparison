import random 
import os
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

from datasets.downstream.CHB_MIT_Scalp_EEG.chbmit_dataset import CHBMITDataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

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

from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from Modules.BIOT.biot import BIOTClassifier
from utils_eval import get_metrics


# --------------------------- 1. 1×1 通道适配 --------------------------- #
class ChannelAdapter(nn.Module):
    """
    输入 : [B, C_in, T]   输出 : [B, C_out, T]
    通过 1×1 Conv + BN + ReLU 实现通道投影
    """
    def __init__(self, in_ch: int = 23, out_ch: int = 19):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

class LitEEGPTCausal(pl.LightningModule):
    """
    LightningModule for CHB-MIT seizure prediction with BIOT backbone.
    可调超参:
        n_unfreeze : 解冻骨干最后 N 个 Transformer block
        head_lr    : 适配器 + 分类头学习率
        ft_lr      : 微调骨干学习率
        dropout    : 额外 Dropout 概率
        total_epochs: 供 CosineLR 计算周期
    """
    def __init__(
        self,
        pretrain_model_choice: int = 0,
        n_unfreeze: int = 0,
        head_lr: float = 2e-4,
        ft_lr: float = 2e-5,
        dropout: float = 0.1,
        total_epochs: int = 60
    ):
        super().__init__()
        self.save_hyperparameters()
        ckpts = [
            "Modules/BIOT/EEG-PREST-16-channels.ckpt",
            "Modules/BIOT/EEG-SHHS+PREST-18-channels.ckpt",
            "Modules/BIOT/EEG-six-datasets-18-channels.ckpt",
        ]
        in_ch = 16 if pretrain_model_choice == 0 else 18
        self.channel_adapter = ChannelAdapter(in_ch=23, out_ch=in_ch)
        self.extra_drop      = nn.Dropout(dropout)

        biot = BIOTClassifier(
            n_classes=2, n_channels=in_ch, n_fft=200, hop_length=100
        )
        biot.biot.load_state_dict(torch.load(ckpts[pretrain_model_choice]))
        self.backbone = biot
        print(f"✅ loaded pretrained weights from {ckpts[pretrain_model_choice]}")

        for p in self.backbone.biot.parameters():
            p.requires_grad = False
        if n_unfreeze > 0:
            print(f'biot.biot:')
            print(biot.biot)
            blocks = self.backbone.biot.transformer.layers.layers
            for blk in blocks[-n_unfreeze:]:
                for p in blk.parameters():
                    p.requires_grad = True

        self.loss_fn    = nn.CrossEntropyLoss()
        self.metrics    = {"train": [], "valid": [], "test": []}
        self.is_sanity  = True
        self.best_thr = 0.5

    def forward(self, x):
        if x.shape[-1] % 200:
            x = x[..., : x.shape[-1] - x.shape[-1] % 200]
        x = self.channel_adapter(x)
        x = self.extra_drop(x)
        return x, self.backbone(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        y    = y.long()
        _, logit = self.forward(x)
        loss = self.loss_fn(logit, y)
        acc  = (logit.argmax(-1) == y).float().mean()

        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc",  acc,  on_epoch=True, prog_bar=True)
        if stage != "train":
            y_prob = torch.softmax(logit, -1)[:, 1]
            self.metrics[stage].append((y.cpu(), y_prob.cpu()))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()

        _, logits = self.forward(x)
        y_prob = torch.softmax(logits, -1)[:, 1]  # 概率

        y_pred = (y_prob > self.best_thr).long()

        loss = self.loss_fn(logits, y)
        acc = (y_pred == y).float().mean()

        # Lightning 日志
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        self.metrics["test"].append((y.cpu(), y_prob.cpu()))
        return loss

    # ---------- 2.4 计算完整指标 ----------
    def _epoch_end_metrics(self, stage: str):
        if stage == "train" or self.is_sanity:
            return
        ys, ps = zip(*self.metrics[stage])
        y_true = torch.cat(ys).numpy()
        y_prob = torch.cat(ps).numpy()
        names  = ["accuracy", "balanced_accuracy", "precision",
                  "recall", "cohen_kappa", "f1", "roc_auc", "specificity"]
        res = get_metrics(y_prob, y_true, names, True)
        for k, v in res.items():
            self.log(f"{stage}_{k}", v, prog_bar=(stage == "valid"))

        if stage == "valid":
            from sklearn.metrics import f1_score, balanced_accuracy_score

            best_thr, best_bacc, best_f1 = 0.5, 0.0, 0.0
            for t in np.linspace(0.05, 0.95, 37):
                y_pred = (y_prob > t)
                cur_bacc = balanced_accuracy_score(y_true, y_pred)
                cur_f1 = f1_score(y_true, y_pred)
                if (cur_bacc > best_bacc) or (cur_bacc == best_bacc and cur_f1 > best_f1):
                    best_thr, best_bacc, best_f1 = t, cur_bacc, cur_f1

            self.best_thr = best_thr

            self.log("valid_bacc_best", best_bacc, prog_bar=True, sync_dist=True)
            self.log("valid_f1_best", best_f1, prog_bar=False)
            self.log("valid_thr_best", best_thr, prog_bar=False)

            print("Label cnt:", np.bincount(y_true))
            print("Prob mean/std:", y_prob.mean(), y_prob.std())
            print("Best thr/bacc/f1:", best_thr, best_bacc, best_f1)

    def on_validation_epoch_end(self):
        self._epoch_end_metrics("valid")
        self.metrics["valid"].clear()
        self.is_sanity = False

    def on_test_epoch_end(self):
        self._epoch_end_metrics("test")
        self.metrics["test"].clear()

    def configure_optimizers(self):
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"trainable params: {trainable / 1e6:.3f} M / {total / 1e6:.3f} M")

        head_params = (
                list(self.channel_adapter.parameters())
                + list(self.extra_drop.parameters())
                + list(self.backbone.classifier.parameters())
        )


        ft_params = [
            p for n, p in self.backbone.named_parameters()
            if p.requires_grad and not n.startswith("classifier.")
        ]

        opt = torch.optim.AdamW(
            [
                {"params": head_params, "lr": self.hparams.head_lr},
                {"params": ft_params, "lr": self.hparams.ft_lr},
            ],
            weight_decay=1e-3,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.total_epochs
        )
        return {"optimizer": opt, "lr_scheduler": sched}


# load configs
# -- LOSO 

# load configs
# -- LOSO 
'''from utils import *
import math
data_path = "../datasets/downstream/Data/BCIC_2b_0_38HZ/"
pretrain_model_choice=0
seed_torch(7)
for i in range(1,10):
    all_subjects = [i]
    all_datas = []
    train_dataset,valid_dataset,test_dataset = get_data(i,data_path,1, is_few_EA = True, target_sample=200*4)
    
    global max_epochs
    global steps_per_epoch
    global max_lr

    batch_size=64

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    
    #max_epochs = 100
    max_epochs = 1
    steps_per_epoch = math.ceil(len(train_loader) )
    max_lr = 4e-4

    # init model
    model = LitEEGPTCausal(pretrain_model_choice)

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]
    
    trainer = pl.Trainer(accelerator='cuda',
                         max_epochs=max_epochs, 
                         callbacks=callbacks,
                         logger=[pl_loggers.TensorBoardLogger('./log/', name="BIOT_BCIC2B_tb", version="subject{}_model{}".format(i,pretrain_model_choice)), 
                                 pl_loggers.CSVLogger('./logs/', name="BIOT_BCIC2B_csv")])

    trainer.fit(model, train_loader, valid_loader, ckpt_path='last')
    trainer.test(model, test_loader)'''

# ============ A) 加载 .npy 数据路径 ============ #
'''import math
x_path = "../datasets/downstream/CHB_MIT_Scalp_EEG/processed_data/shifted/chb06_X_shift.npy"
y_path = "../datasets/downstream/CHB_MIT_Scalp_EEG/processed_data/shifted/chb06_y_shift.npy"
#x_path = "../datasets/downstream/CHB_MIT_Scalp_EEG/shifted/chb01_X_shift.npy"
#y_path = "../datasets/downstream/CHB_MIT_Scalp_EEG/shifted/chb01_y_shift.npy"

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

# DataLoader
train_loader = DataLoader(train_ds, batch_size=32, sampler=train_sampler, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

model = LitEEGPTCausal(pretrain_model_choice=0)

max_epochs = 1
steps_per_epoch = math.ceil(len(train_loader))
max_lr = 4e-4

lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
callbacks = [lr_monitor]

trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator='gpu',
    callbacks=callbacks,
    logger=[
        pl_loggers.TensorBoardLogger('./logs_biot/', name="BIOT_CHBMIT_tb", version="chb01"),
        pl_loggers.CSVLogger('./logs_biot/', name="BIOT_CHBMIT_csv")
    ]
)

trainer.fit(model, train_loader)
trainer.test(model, val_loader)'''

import os, glob, math, gc, argparse, yaml
import random, numpy as np, torch, pytorch_lightning as pl
from torch.utils.data import WeightedRandomSampler, DataLoader, random_split
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import optuna
from dotmap import DotMap
from sklearn.utils.class_weight import compute_class_weight
from datasets.downstream.CHB_MIT_Scalp_EEG.chbmit_dataset import CHBMITDataset

def load_cfg(p):
    c = yaml.safe_load(open(p, "r", encoding="utf-8"))
    c["data"].pop("x_path", None); c["data"].pop("y_path", None)
    return DotMap(c)

def make_loaders(x_path, y_path, cfg):
    ds = CHBMITDataset(x_path, y_path, use_avg=True,
                       scale_div=10.0, interpolation_len=cfg.data.interpolation_len)
    n_val = int(len(ds) * cfg.data.val_ratio)
    tr_ds, va_ds = random_split(ds, [len(ds)-n_val, n_val])

    y_tr = np.array([ds.y[i] for i in tr_ds.indices])
    pos, neg = y_tr.sum(), len(y_tr)-y_tr.sum()
    samp_w = [1/pos if l else 1/neg for l in y_tr]
    sampler = WeightedRandomSampler(samp_w, len(samp_w), replacement=True)

    tr_ld = DataLoader(tr_ds, batch_size=cfg.data.batch, sampler=sampler, num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=cfg.data.batch, shuffle=False, num_workers=0)

    from collections import Counter
    xb, yb = next(iter(tr_ld))
    print("Sampler first-batch label count →", Counter(yb.tolist()))
    return tr_ld, va_ld

# ---------- 3. Optuna objective ----------
def objective(trial, tr_ld, va_ld, max_epochs=30):
    import pytorch_lightning as pl
    pl.seed_everything(trial.number, workers=True)
    hp = dict(
        n_unfreeze = trial.suggest_int("n_unfreeze", 0, 12),
        head_lr    = trial.suggest_loguniform("head_lr", 1e-5, 2e-3),
        ft_lr      = trial.suggest_loguniform("ft_lr",   1e-6, 2e-4),
        dropout    = trial.suggest_float("dropout", 0.0, 0.3),
    )
    model = LitEEGPTCausal(pretrain_model_choice=0, **hp)
    #early = EarlyStopping(monitor="valid_bacc_best", mode="max", patience=4, verbose=False)
    early = EarlyStopping(monitor="valid_bacc_best", mode="max", patience=999, verbose=False)
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices=1,
                         logger=False, enable_checkpointing=False,
                         #callbacks=[early], enable_progress_bar=False)
                         enable_progress_bar=False)
    trainer.fit(model, tr_ld, va_ld)
    return trainer.callback_metrics["valid_bacc_best"].item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="../conf/mybiot.yaml")
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)

    shift_dir = cfg.data.shift_dir
    x_files = sorted(glob.glob(os.path.join(shift_dir, "*_X_shift.npy")))
    assert x_files, f"No *_X_shift.npy in {shift_dir}"

    for x_path in x_files:
        pid = os.path.basename(x_path).split("_")[0]
        '''if pid=='chb01' or pid=='chb03' or pid=='chb05' or pid=='chb06' or pid=='chb07':
            continue'''
        if pid!='chb14':
            continue
        y_path = x_path.replace("_X_shift.npy", "_y_shift.npy")
        if not os.path.exists(y_path): continue
        print(f"\n===========  Patient {pid}  ===========")

        tr_ld, va_ld = make_loaders(x_path, y_path, cfg)
        from collections import Counter
        cnt = Counter()
        for _, yb in tr_ld:
            cnt.update(yb.tolist())

        print(f"Epoch label count → {cnt}")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, tr_ld, va_ld),
                       n_trials=args.trials, timeout=3600)
        best_params = study.best_params
        print("Best params:", best_params)

        best_model = LitEEGPTCausal(pretrain_model_choice=0, **best_params)
        tb_logger  = pl_loggers.TensorBoardLogger("./logs_biot/", "BIOT_CHBMIT_tb", pid)
        csv_logger = pl_loggers.CSVLogger("./logs_biot/", "BIOT_CHBMIT_csv", pid)
        lr_mon     = LearningRateMonitor(logging_interval="epoch")

        trainer = pl.Trainer(max_epochs=100, accelerator="gpu", devices=1,
                             logger=[tb_logger, csv_logger],
                             callbacks=[lr_mon], log_every_n_steps=20,
                             enable_checkpointing=False)
        trainer.fit(best_model, tr_ld, va_ld)
        trainer.test(best_model, va_ld)

        del best_model, tr_ld, va_ld
        torch.cuda.empty_cache(); gc.collect()

if __name__ == "__main__":
    main()
