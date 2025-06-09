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

from torch.utils.data import random_split, DataLoader
from datasets.downstream.CHB_MIT_Scalp_EEG.chbmit_dataset import CHBMITDataset
from torch.utils.data import WeightedRandomSampler

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

'''use_channels_names = [
    'FP1', 'FP2',
    'F7', 'F3', 'FZ', 'F4', 'F8',
    'T7', 'C3', 'CZ', 'C4', 'T8',
    'P7', 'P3', 'PZ', 'P4', 'P8',
    'O1', 'O2'
]'''
use_channels_names = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3',
                      'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
                      'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ',
                      'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8-1']

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

class LitEEGPTCausal(pl.LightningModule):

    def __init__(self, load_path="../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"):
        super().__init__()
        self.chans_num = len(use_channels_names)
        # init model
        target_encoder = EEGTransformer(
            #img_size=[19, 2 * 256],
            img_size=[23, 2 * 256],#我改的
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

        self.target_encoder = target_encoder
        #self.chans_id = target_encoder.prepare_chan_ids(use_channels_names)

        #self.channel_adapter = ChannelAdapter(in_ch=23, out_ch=19)
        self.chans_id = torch.arange(19)

        # -- load checkpoint
        pretrain_ckpt = torch.load(load_path)

        target_encoder_stat = {}
        for k, v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]] = v

        self.target_encoder.load_state_dict(target_encoder_stat)

        self.chan_conv = Conv1dWithConstraint(23, self.chans_num, 1, max_norm=1)
        #self.chan_conv = Conv1dWithConstraint(23, self.chans_num, 1, max_norm=1)注掉了
        #self.chan_conv = Conv1dWithConstraint(23, 58, 1, max_norm=1)

        self.linear_probe1 = LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2 = LinearWithConstraint(240, 2, max_norm=0.25)

        self.drop = torch.nn.Dropout(p=0.50)

        # 类别数=2
        '''class_weights = [1.0, 33.36]  # 例如 [1, (2335/70)]
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        self.loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)'''

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train": [], "valid": [], "test": []}
        self.is_sanity = True

    def forward(self, x):
        B, C, T = x.shape
        x = x / 10
        x = x - x.mean(dim=-2, keepdim=True)
        x = temporal_interpolation(x, 2 * 256)

        #x = self.channel_adapter(x)#我加的
        x = self.chan_conv(x)
        self.target_encoder.eval()
        z = self.target_encoder(x, self.chans_id.to(x))

        h = z.flatten(2)

        h = self.linear_probe1(self.drop(h))

        h = h.flatten(1)

        h = self.linear_probe2(h)

        return x, h

    def training_step(self, batch, batch_idx):
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
            #list(self.channel_adapter.parameters()) + #我加的
            list(self.chan_conv.parameters()) +
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
        )


'''# load configs
# -- LOSO
from utils import *
import math

seed_torch(9)
path = "../datasets/downstream"

global max_epochs
global steps_per_epoch
global max_lr

batch_size = 64
max_epochs = 100

Folds = {
    1: ([12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26], [1, 3, 4, 5, 8, 9, 10, 15, 19, 25]),
    2: ([2, 6, 7, 11, 17, 18, 20, 21, 22, 23, 24, 26], [1, 3, 4, 5, 8, 9, 10, 15, 19, 25]),
    3: ([2, 6, 7, 11, 12, 13, 14, 16, 22, 23, 24, 26], [1, 3, 4, 5, 8, 9, 10, 15, 19, 25]),
    4: ([2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21], [1, 3, 4, 5, 8, 9, 10, 15, 19, 25]),
}

for k, v in Folds.items():
    # init model
    model = LitEEGPTCausal()
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]
    training = read_kaggle_ern_train(path, subjects=v[0])
    validation = read_kaggle_ern_test(path, subjects=v[1])
    train_loader = torch.utils.data.DataLoader(training, batch_size=batch_size, num_workers=0, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size, num_workers=0, shuffle=False)

    steps_per_epoch = math.ceil(len(train_loader))
    max_lr = 4e-4
    trainer = pl.Trainer(accelerator='cuda',
                         max_epochs=max_epochs,
                         callbacks=callbacks,
                         enable_checkpointing=False,
                         logger=[pl_loggers.TensorBoardLogger('./logs/', name="EEGPT_ERN_tb", version=f"fold{k}"),
                                 pl_loggers.CSVLogger('./logs/', name="EEGPT_ERN_csv")])

    trainer.fit(model, train_loader, valid_loader, ckpt_path='last')'''

def shift_labels(y: np.ndarray, step: int) -> np.ndarray:
    """
    把标签向后平移 step 个窗口:
      y[i] <- y[i + step]
    最后 step 个标签无效 => 置 -1 方便过滤
    """
    y_shift = np.roll(y, -step)
    y_shift[-step:] = -1
    return y_shift

import math

global max_epochs
global steps_per_epoch
global max_lr

batch_size = 64
max_epochs = 1


# ============ A) 定义好 Dataset 的文件路径 & 参数 ============
path = "../datasets/downstream/CHB_MIT_Scalp_EEG/"
x_path = path + "chb01_X.npy"  # 这里换成你的实际输出 .npy 路径
y_path = path + "chb01_y.npy"
X = np.load(x_path)
y = np.load(y_path)

print("Before shift =>", X.shape, y.shape)
print("y diistribution:",np.bincount(y))

K = 1  # 用当前段预测1段之后
y_shifted = shift_labels(y, K)
mask = (y_shifted != -1)
X_shifted = X[mask]
y_shifted = y_shifted[mask]
print("After shift =>", X_shifted.shape, y_shifted.shape)

# ========== 3) 把 (X_shifted, y_shifted) 存成新文件 "_shift.npy" ==========
shifted_path = path + "shifted/"
os.makedirs(shifted_path, exist_ok=True)
x_shifted_path = shifted_path + "chb01_X_shift.npy"
y_shifted_path = shifted_path + "chb01_y_shift.npy"
np.save(x_shifted_path, X_shifted)
np.save(y_shifted_path, y_shifted)

dataset = CHBMITDataset(
    x_path=x_shifted_path,
    y_path=y_shifted_path,
    use_avg=True,
    scale_div=10.0,
    interpolation_len=256  # 例如：把 T=15360 压缩到 256
)

# 数据量可能不多，这里简单拆 80% / 20% 做 train/val
train_len = int(len(dataset) * 0.8)
val_len = len(dataset) - train_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])

# 1) 在生成 train_ds, val_ds 之后，统计标签
train_labels = []
for idx in train_ds.indices:
    train_labels.append(dataset.y[idx].item())
train_labels = np.array(train_labels)

counts = np.bincount(train_labels)
count_neg, count_pos = counts[0], counts[1]
weight_neg = 1.0 / count_neg
weight_pos = 1.0 / count_pos

print(f'counts:{counts}')

sample_weights = [weight_pos if lab==1 else weight_neg for lab in train_labels]
train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# ============ B) 构建 DataLoader ============
train_loader = DataLoader(train_ds, batch_size=32, sampler=train_sampler, num_workers=0) #sample和shuffle冲突，不能同时用
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

# ============ C) 初始化模型 ============
# 注意: in_ch=23, in_time=256, num_classes=2 需与你的数据形状匹配
model = LitEEGPTCausal()

steps_per_epoch = math.ceil(len(train_loader))
max_lr = 4e-4
# ============ D) Lightning Trainer & fit ============
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
callbacks = [lr_monitor]
trainer = pl.Trainer(
    max_epochs=max_epochs,
    callbacks=callbacks,
    enable_checkpointing=False,
    accelerator='gpu',  # 若有GPU环境，否则改成'cpu'
    devices=1,
    logger=[pl_loggers.TensorBoardLogger('./logs/', name="EEGPT_CHBMIT_tb", version=f"foldxxx"),
                                 pl_loggers.CSVLogger('./logs/', name="EEGPT_CHBMIT_csv")],
    log_every_n_steps=10
)
#trainer.fit(model, train_loader, val_loader)
trainer.fit(model, train_loader)
trainer.test(model, val_loader)

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def plot_true_vs_pred_over_time(model, data_loader, time_interval=4.0, save_path="label_comparison.png"):
    """
    画出“随真实时间推移，标签如何变化”的图：
      如果数据集没有提供时间戳，则假设每个样本间隔 time_interval（单位秒）。
    使用阶梯图（step plot）来展示离散的 0/1 标签随时间的变化。

    参数：
      model: 训练好的模型
      data_loader: 测试/验证数据加载器，期望返回 (x, y) 或 (x, y, time) 三元组
      time_interval: 如果数据集中没有时间信息，则认为每个样本间隔多少秒（默认1.0秒）
      save_path: 保存图像的路径
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_times = []
    sample_index = 0  # 用于生成时间戳（如果没有提供）

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting over time"):
            # 如果batch中包含时间戳，则假设格式为 (x, y, time)
            if len(batch) == 3:
                x, y, times = batch
                # 假设 times 为 tensor，此处转换为 numpy 数组
                times = times.cpu().numpy()
            else:
                x, y = batch
                batch_size = x.size(0)
                # 没有时间信息则生成连续的时间戳：sample_index * time_interval
                times = np.arange(sample_index, sample_index + batch_size) * time_interval
                sample_index += batch_size

            # 转移到对应设备
            x = x.to(model.device) if hasattr(model, 'device') else x
            _, logits = model(x)
            preds = torch.argmax(logits, dim=-1)

            print("y =", y, "preds =", preds.cpu())

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_times.append(times)

    # 拼接所有batch的数据
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_times = np.concatenate(all_times)

    # 如果时间顺序混乱，则按时间排序
    sorted_indices = np.argsort(all_times)
    all_times = all_times[sorted_indices]
    all_labels = all_labels[sorted_indices]
    all_preds = all_preds[sorted_indices]

    # 绘制阶梯图，使得标签变化显示为阶梯式跳变
    plt.figure(figsize=(12, 6))
    plt.scatter(all_times, all_labels, label="True Labels", marker='o', alpha=0.8)
    plt.scatter(all_times, all_preds, label="Predicted Labels", marker='x', alpha=0.8)

    plt.yticks([0, 1], ['0', '1'])
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Time (s)")
    plt.ylabel("Label")
    plt.title("True vs. Predicted Labels Over Time")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

'''# 在构造 val_loader 后面
val_labels = []
for _, y in val_loader:
    val_labels.extend(y.numpy())
print("Val set label distribution:", np.bincount(val_labels))

plot_true_vs_pred_over_time(model, val_loader, save_path="label_comparison.png")'''


