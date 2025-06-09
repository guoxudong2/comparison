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

from Modules.BIOT.biot import (
    BIOTClassifier,
)
import torch
from utils_eval import get_metrics

class ChannelAdapter(nn.Module):
    """
    输入: [B, in_ch, T]
    输出: [B, out_ch, T]
    """
    def __init__(self, in_ch=23, out_ch=19):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = self.conv(x)    # -> [B, out_ch, T]
        x = self.bn(x)
        x = F.relu(x)
        return x


class LitEEGPTCausal(pl.LightningModule):

    def __init__(self, pretrain_model_choice = 0):
        super().__init__()
        self.save_hyperparameters()
        pretrain_models = ["Modules/BIOT/EEG-PREST-16-channels.ckpt",
                           "Modules/BIOT/EEG-SHHS+PREST-18-channels.ckpt",
                           "Modules/BIOT/EEG-six-datasets-18-channels.ckpt"]
        if pretrain_model_choice == 0: in_channels = 16
        elif pretrain_model_choice == 1: in_channels = 18
        elif pretrain_model_choice == 2: in_channels = 18
        else: raise ValueError("pretrain_model_choice should be 0, 1, or 2")
        
        #self.chan_conv      = Conv1dWithConstraint(3, in_channels, 1, max_norm=1)
        self.channel_adapter = ChannelAdapter(in_ch=23, out_ch=in_channels)
        model = BIOTClassifier(
                    n_classes=2, #丛4改为2
                    # set the n_channels according to the pretrained model if necessary
                    n_channels=in_channels,
                    n_fft=200,
                    hop_length=100,
                )
        model.biot.load_state_dict(torch.load(pretrain_models[pretrain_model_choice]))
        print(f"load pretrain model from {pretrain_models[pretrain_model_choice]}")
        for p in model.biot.parameters():
            p.requires_grad = False
        self.feature        = model
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity=True
    
    def forward(self, x):
        B, C, T = x.shape
        if T%200!=0: 
            x = x[:,:,0:T-T%200]
            T = T-T%200

        #x = self.chan_conv(x)
        x = self.channel_adapter(x)
        pred = self.feature(x)
        return x, pred

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        #y = F.one_hot(y.long(), num_classes=4).float() 我先注释掉了
        y = y.long()#我加的
        label = y
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        #accuracy = ((torch.argmax(logit, dim=-1)==torch.argmax(label, dim=-1))*1.0).mean()
        accuracy = ((torch.argmax(logit, dim=-1) == label) * 1.0).mean()
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)
        
        return loss
        
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()
    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()
            
        label, y_score = [], []
        for x,y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        print(label.shape, y_score.shape)
        
        metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "cohen_kappa", "f1", "roc_auc", "specificity"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, True)
        
        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)
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
        
        y_score =  logit
        y_score =  torch.softmax(y_score, dim=-1)[:,1]
        self.running_scores["valid"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))

        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            #list(self.chan_conv.parameters())+
            list(self.channel_adapter.parameters()) +
            list(self.feature.parameters()),
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

    def test_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()

        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1) == label) * 1.0).mean()

        self.log('test_loss', loss, on_epoch=True, on_step=False)
        self.log('test_acc', accuracy, on_epoch=True, on_step=False)

        y_score = torch.softmax(logit, dim=-1)[:, 1]  # 取第1类的概率，适用于二分类情况
        self.running_scores["test"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))

        pppred = torch.argmax(logit, dim=-1)
        print("Pred10:", pppred[:10].cpu().numpy())
        print("True10:", label[:10].cpu().numpy())

        print("Logits10:", logit[:10].cpu().detach().numpy())
        print("Softmax prob10:", torch.softmax(logit, dim=-1)[:10].cpu().detach().numpy())

        return loss

    def on_test_epoch_end(self):
        label, y_score = [], []
        for x, y in self.running_scores["test"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)

        # 计算测试指标
        metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "cohen_kappa", "f1", "roc_auc", "specificity"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, True)

        for key, value in results.items():
            self.log('test_' + key, value, on_epoch=True, on_step=False, sync_dist=True)
        print(f"Test results: {results}")

        pppred = (y_score > 0.5).long()  # 对于二分类
        print("Predicted labels:", pppred.numpy())
        print("True labels:", label.numpy())

        from sklearn.metrics import confusion_matrix
        print("confusion matrix")
        print(confusion_matrix(label.numpy(), pppred.numpy()))


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

# ====================================================================== #
# 🏃 main：一次跑完 shift_dir 里所有 *_X_shift.npy
# ====================================================================== #
import math
import argparse, yaml, glob, gc
from dotmap import DotMap
from sklearn.utils.class_weight import compute_class_weight


def load_cfg(p):
    c = yaml.safe_load(open(p, "r", encoding="utf-8"))
    c["data"].pop("x_path", None)
    c["data"].pop("y_path", None)
    return DotMap(c)

def make_loaders(x_path, y_path, cfg):
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
    return tr_ld, va_ld, class_w

global max_epochs
global steps_per_epoch
global max_lr

if __name__ == "__main__":
    # -------- 读取配置 --------
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="../conf/mybiot.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)

    # -------- 找到所有 *_X_shift.npy --------
    shift_dir = cfg.data.shift_dir
    x_files = sorted(glob.glob(os.path.join(shift_dir, "*_X_shift.npy")))
    assert x_files, f"No *_X_shift.npy under {shift_dir}"

    for x_path in x_files:
        pid = os.path.basename(x_path).split("_")[0]      # chb01
        if pid=='chb01' or pid=='chb03':
            continue
        y_path = x_path.replace("_X_shift.npy", "_y_shift.npy")
        if not os.path.exists(y_path):
            print(f"[Skip] {pid}: y file missing"); continue

        print(f"\n================ Patient {pid} ================")

        # ------------- DataLoader & class_weight -------------
        tr_ld, va_ld, cls_w = make_loaders(x_path, y_path, cfg)
        print("Train class counts:", np.bincount([tr_ld.dataset.dataset.y[i] for i in tr_ld.dataset.indices]))
        print("Class weights     :", cls_w)

        max_epochs = 100
        steps_per_epoch = math.ceil(len(tr_ld))
        max_lr = 4e-4

        # ------------- Model -------------
        steps_per_epoch = math.ceil(len(tr_ld))
        model = LitEEGPTCausal(pretrain_model_choice=0)

        # 把 class weight 传给 loss
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        '''model.loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(cls_w, dtype=torch.float32, device=device)
        )'''

        # ------------- Logger / Trainer -------------
        tb_logger  = pl_loggers.TensorBoardLogger(
            "./logs_biot/", name="BIOT_CHBMIT_tb",  version=pid)
        csv_logger = pl_loggers.CSVLogger(
            "./logs_biot/", name="BIOT_CHBMIT_csv", version=pid)
        lr_mon = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

        trainer = pl.Trainer(
            max_epochs=cfg.train.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=cfg.train.gpu if torch.cuda.is_available() else 1,
            logger=[tb_logger, csv_logger],
            callbacks=[lr_mon],
            log_every_n_steps=10,
            enable_checkpointing=False
        )

        print(f"=== {pid}: {len(tr_ld.dataset)} train / {len(va_ld.dataset)} val ===")
        trainer.fit(model, tr_ld, va_ld)
        trainer.test(model, va_ld)

        # ---------- 清理显存 ----------
        del model, tr_ld, va_ld
        torch.cuda.empty_cache()
        gc.collect()
