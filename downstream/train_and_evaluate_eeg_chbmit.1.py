# -*- coding: utf-8 -*-
"""
训练 + 同时输出 Segment-based & Event-based 指标（兼容 Lightning v2.x，从配置文件自动读取 pid）。
直接点击 Run 按钮即可，无需命令行参数。
"""

import os
import math
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binom
from typing import Optional
from functools import partial

from dotmap import DotMap

# 请确认以下模块都在你的工程目录下，并且路径正确：
# - chbmit_dataset_with_t0.py
# - Modules/models/EEGPT_mcae.py
# - Modules/Network/utils.py
# - downstream/utils.py
# - utils_eval.py

from datasets.downstream.CHB_MIT_Scalp_EEG.chbmit_dataset_with_t0 import CHBMITDatasetWithT0
from Modules.models.EEGPT_mcae import EEGTransformer
from Modules.Network.utils import LinearWithConstraint
from downstream.utils import temporal_interpolation
from utils_eval import get_metrics  # 如有自定义评估函数，可自行扩展



# =========================== Lightning Module ===========================
class LitEEGPT(pl.LightningModule):
    def __init__(self,
                 max_lr: float,
                 steps_per_epoch: int,
                 max_epochs: int,
                 pid: str,
                 pretrain_ckpt: Optional[str] = None
                 ):
        """
        max_lr: 学习率
        steps_per_epoch: 每个 epoch 里有多少个 batch
        max_epochs: 总共跑多少 Epoch
        pid: patient id，用于在 test 时加载 seizure_times.npy
        pretrain_ckpt: 如果有预训练模型，就给路径，否则传 None
        """
        super().__init__()
        self.save_hyperparameters()

        self.pid = pid
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = max_epochs

        # ----- 1) 初始化模型 -----
        self.target_encoder = EEGTransformer(
            img_size=[23, 2*256],
            patch_size=32*2,
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
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        if pretrain_ckpt is not None and os.path.exists(pretrain_ckpt):
            ckpt = torch.load(pretrain_ckpt, map_location="cpu")
            state_dict = {}
            for k, v in ckpt['state_dict'].items():
                if k.startswith("target_encoder."):
                    state_dict[k.replace("target_encoder.", "")] = v
            self.target_encoder.load_state_dict(state_dict)

        self.channel_adapter = nn.Conv1d(23, 23, kernel_size=1, bias=False)
        self.bn_adapter = nn.BatchNorm1d(23)
        self.dropout = nn.Dropout(0.3)

        self.linear_probe1 = nn.Sequential(
            LinearWithConstraint(2048, 256, max_norm=1),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.linear_probe2 = LinearWithConstraint(256 * 15, 2, max_norm=0.25)

        self.loss_fn = nn.CrossEntropyLoss()

        # 用于在 validation & test 阶段累积预测结果
        self.val_logits_buf = []
        self.val_labels_buf = []

        self.test_preds = []
        self.test_logits = []
        self.test_preds_logits = []
        self.test_t0s   = []
        self.test_labels= []


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入 x: [B, 23, T]（T=1024 原始，或 512 插值后）
        返回 logits: [B, 2]
        """
        x = temporal_interpolation(x, 2*256)   # → [B, 23, 512]
        x = self.channel_adapter(x)
        x = self.bn_adapter(x)
        x = F.relu(x)

        self.target_encoder.eval()
        z = self.target_encoder(x, torch.arange(23).to(x.device))  # [B, num_tokens, dim]

        h = z.flatten(2)                       # [B, dim, num_tokens]
        h = self.linear_probe1(self.dropout(h))# → [B, 256, num_tokens]
        h = h.flatten(1)                       # → [B, 256*num_tokens]
        logits = self.linear_probe2(h)         # → [B, 2]

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.max_lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.max_epochs,
            pct_start=0.2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    # =================== 训练 & 验证 ===================
    def training_step(self, batch, batch_idx):
        x, y, t0 = batch
        logits = self.forward(x)

        print(f'----------yyyyyy:{y}')
        print(f'----------logits:{logits}')

        loss = self.loss_fn(logits, y.long())
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, t0 = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y.long())
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

        # 缓存 logits 和 labels，用于 epoch 末计算 aggregate 指标
        self.val_logits_buf.append(logits.detach().cpu())
        self.val_labels_buf.append(y.detach().cpu())

        return

    def on_validation_epoch_end(self) -> None:
        # 如果是 sanity check，就直接清空并返回
        if not self.val_labels_buf:
            self.val_logits_buf.clear()
            self.val_labels_buf.clear()
            return

        logits = torch.cat(self.val_logits_buf, dim=0).numpy()  # (N_val,2)
        labels = torch.cat(self.val_labels_buf, dim=0).numpy()  # (N_val,)
        preds  = np.argmax(logits, axis=1)
        prob1  = logits[:, 1]

        acc  = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec  = recall_score(labels, preds, zero_division=0)
        f1   = f1_score(labels, preds, zero_division=0)
        auc  = roc_auc_score(labels, prob1) if len(np.unique(labels)) == 2 else 0.0

        self.log("val_seg_acc", acc, prog_bar=True)
        self.log("val_seg_prec", prec)
        self.log("val_seg_rec", rec)
        self.log("val_seg_f1", f1)
        self.log("val_seg_auc", auc)

        self.val_logits_buf.clear()
        self.val_labels_buf.clear()

    # ==================== 测试 ====================
    def test_step(self, batch, batch_idx):
        x, y, t0 = batch
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=-1)

        # 缓存下来
        self.test_preds.append(preds.detach().cpu().numpy())
        self.test_logits.append(logits.detach().cpu().numpy())
        self.test_t0s.append(t0.detach().cpu().numpy())
        self.test_labels.append(y.detach().cpu().numpy())
        self.test_preds_logits.append(logits.detach().cpu().numpy())

        return {"logits": logits.detach().cpu(), "y": y.detach().cpu()}

    def on_test_epoch_end(self) -> None:
        # 1) Segment-based Test
        all_preds  = np.concatenate(self.test_preds, axis=0)   # (N_test,)
        all_labels = np.concatenate(self.test_labels, axis=0)  # (N_test,)
        all_logits = np.concatenate(self.test_logits, axis=0)  # 正确，2D logits

        # 注意：Lightning v2.x 中，要访问 test_step 返回的 outputs，可在 trainer.test(...) 之后，
        # 也可以像这里直接用 self.test_step_outputs（需要调用 .reset_test_step_outputs() 来清空，见下方）

        prob1 = all_logits[:, 1]
        seg_acc  = accuracy_score(all_labels, all_preds)
        seg_prec = precision_score(all_labels, all_preds, zero_division=0)
        seg_rec  = recall_score(all_labels, all_preds, zero_division=0)
        seg_f1   = f1_score(all_labels, all_preds, zero_division=0)
        seg_auc  = roc_auc_score(all_labels, prob1) if len(np.unique(all_labels)) == 2 else 0.0

        self.log("test_seg_acc", seg_acc)
        self.log("test_seg_prec", seg_prec)
        self.log("test_seg_rec", seg_rec)
        self.log("test_seg_f1", seg_f1)
        self.log("test_seg_auc", seg_auc)

        unique_true, cnt_true = np.unique(all_labels, return_counts=True)
        print(f"[Test Set] True label counts: {dict(zip(unique_true, cnt_true))}")

        print("---------- Segment-based Test Results ----------")
        print(f"Accuracy = {seg_acc*100:.2f}%  "
              f"Precision = {seg_prec*100:.2f}%  Recall = {seg_rec*100:.2f}%  "
              f"F1 = {seg_f1*100:.2f}%  AUC = {seg_auc:.4f}")
        print("------------------------------------------------\n")

        # 2) Event-based Test
        all_preds = np.concatenate(self.test_preds, axis=0)   # (N_test,)
        all_t0    = np.concatenate(self.test_t0s, axis=0)     # (N_test,)

        # 加载真实 seizure_times.npy
        seizure_times = np.load(os.path.join("../datasets/downstream/CHB_MIT_Scalp_EEG/processed_data", f"{self.pid}_seizure_times.npy"))

        SPH_seconds = 3 * 60
        SOP_seconds = 30 * 60
        REFRACTORY  = 30 * 60

        # k-of-n 策略
        K = 18
        N = 30

        alarm_times = []
        last_alarm_time = -1e9
        for i in range(len(all_preds)):
            if i < N - 1:
                continue
            window = all_preds[i-N+1 : i+1]
            if window.sum() >= K:
                alarm_t = int(all_t0[i-N+1])
                if alarm_t - last_alarm_time >= REFRACTORY:
                    alarm_times.append(alarm_t)
                    last_alarm_time = alarm_t
        alarm_times = np.array(alarm_times, dtype=np.int64)

        TP_event = 0
        FN_event = 0
        for t_ictal in seizure_times:
            mask_valid = (alarm_times >= t_ictal - SOP_seconds) & (alarm_times <= t_ictal - SPH_seconds)
            if np.any(mask_valid):
                TP_event += 1
            else:
                FN_event += 1

        legal_mask = np.zeros_like(alarm_times, dtype=bool)
        for t_ictal in seizure_times:
            low  = t_ictal - SOP_seconds
            high = t_ictal - SPH_seconds
            legal_mask |= ((alarm_times >= low) & (alarm_times <= high))
        FP_event = int(np.sum(~legal_mask))

        PREICTAL_LEN  = 1800
        POSTICTAL_LEN = 14400
        RESERVE_TIME  = 60
        skip_per_event = PREICTAL_LEN + POSTICTAL_LEN + 2*RESERVE_TIME
        total_skip = skip_per_event * len(seizure_times)

        start_test = int(all_t0.min())
        end_test   = int(all_t0.max()) + 4
        total_sec = end_test - start_test
        inter_sec  = max(0, total_sec - total_skip)
        inter_h    = inter_sec / 3600.0

        FPR_event_per_h = FP_event / inter_h if inter_h > 0 else float("nan")

        pred_times = []
        for t_ictal in seizure_times:
            mask_valid = (alarm_times >= t_ictal - SOP_seconds) & (alarm_times <= t_ictal - SPH_seconds)
            if np.any(mask_valid):
                first_alarm = np.min(alarm_times[mask_valid])
                pred_times.append(t_ictal - first_alarm)
        avg_pred_sec = float(np.mean(pred_times)) if len(pred_times) > 0 else float("nan")

        num_events = len(seizure_times)
        sens_event = TP_event / num_events if num_events > 0 else 0.0

        λ = FP_event / inter_h if inter_h > 0 else 0.0
        SOP_h = SOP_seconds / 3600.0
        p_hit = 1.0 - np.exp(-λ * SOP_h + 1e-15)
        p_value = float(binom.sf(TP_event - 1, num_events, p_hit)) if num_events > 0 else float("nan")

        self.log("test_event_sens", sens_event * 100.0)
        self.log("test_event_FPR_per_h", FPR_event_per_h)
        self.log("test_event_avgPredSec", avg_pred_sec)
        self.log("test_event_p_value", p_value)

        print("\n========== Event-based Test Results ==========")
        print(f"#Events = {num_events}, TP={TP_event}, FN={FN_event}, FP={FP_event}")
        print(f"Sensitivity_event = {sens_event*100.0:.2f}%")
        print(f"FPR_event/h = {FPR_event_per_h:.4f}")
        print(f"AvgPredTime (sec) = {avg_pred_sec:.1f}")
        print(f"p-value = {p_value:.4e}")
        print("==============================================\n")

        # 清空缓存
        self.test_preds.clear()
        self.test_logits.clear()
        self.test_t0s.clear()
        self.test_labels.clear()
        self.test_preds_logits.clear()

# =========================== DataModule ===========================
class CHBMITDataModule(pl.LightningDataModule):
    def __init__(self, pid: str, cfg: DotMap):
        """
        pid: 病人 ID，例如 "chb04"
        cfg: 配置字典，需包含 data.interpolation_len, data.batch, data.val_ratio, data.shift_dir
        """
        super().__init__()
        self.pid = pid
        self.cfg = cfg

    def setup(self, stage=None):
        """
        这里示例使用“causal split”：
        - 用最后一个发作簇附近的数据作为测试集（test）
        - 剩余数据做 5-fold StratifiedKFold，用其中一折做 val，剩下做 train
        """
        base = os.path.join(self.cfg.data.shift_dir, self.pid)
        X = np.load(base + "_X.npy")            # (N, 23, 1024)
        y = np.load(base + "_y.npy")            # (N,)
        t0= np.load(base + "_t0.npy")           # (N,)
        seizure_times = np.load(base + "_seizure_times.npy")  # (M,)

        last_ictal = int(seizure_times[-1])
        boundary = last_ictal - 1800 - 60

        test_mask = (t0 >= boundary)
        idx_test  = np.where(test_mask)[0]
        idx_train_val = np.where(~test_mask)[0]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx_sub, val_idx_sub = next(skf.split(idx_train_val, y[idx_train_val]))
        idx_train = idx_train_val[train_idx_sub]
        idx_val   = idx_train_val[val_idx_sub]

        full_ds = CHBMITDatasetWithT0(
            x_path=base + "_X.npy",
            y_path=base + "_y.npy",
            t0_path=base + "_t0.npy",
            use_avg=True,
            scale_div=10.0,
            interpolation_len=self.cfg.data.interpolation_len
        )
        self.train_ds = Subset(full_ds, idx_train.tolist())
        self.val_ds   = Subset(full_ds, idx_val.tolist())
        self.test_ds  = Subset(full_ds, idx_test.tolist())

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.cfg.data.batch,
                          shuffle=True,
                          num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.cfg.data.batch,
                          shuffle=False,
                          num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.cfg.data.batch,
                          shuffle=False,
                          num_workers=4)


# =========================== 主程序入口 ===========================
def load_cfg(cfg_path: str) -> DotMap:
    assert os.path.exists(cfg_path), f"找不到配置文件：{cfg_path}"
    cfg_dict = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))
    return DotMap(cfg_dict)


if __name__ == "__main__":
    # 直接从配置文件读取，无需命令行参数
    cfg_path = "../conf/eegpt_causal.yaml"
    cfg = load_cfg(cfg_path)

    pid = cfg.train.pid
    assert pid is not None and len(pid) > 0, "请在 conf/eegpt_causal.yaml 中填写 train.pid，例如 pid: chb04"

    dm = CHBMITDataModule(pid=pid, cfg=cfg)
    dm.setup()

    steps_per_epoch = math.ceil(len(dm.train_ds) / cfg.data.batch)
    model = LitEEGPT(
        max_lr=float(cfg.train.lr),
        steps_per_epoch=steps_per_epoch,
        max_epochs=cfg.train.epochs,
        pid=pid,
        pretrain_ckpt=cfg.train.pretrain_ckpt  # 如果为 null，则传 None
    )

    tb_logger  = pl.loggers.TensorBoardLogger("./logs_eegpt/", name="EEGPT", version=pid)
    csv_logger = pl.loggers.CSVLogger("./logs_eegpt/", name="EEGPT_csv", version=pid)

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[cfg.train.gpu] if torch.cuda.is_available() else 1,
        logger=[tb_logger, csv_logger],
        log_every_n_steps=10,
        enable_checkpointing=False
    )

    print(f"== PID={pid} ,  #train={len(dm.train_ds)}, #val={len(dm.val_ds)}, #test={len(dm.test_ds)} ==")
    trainer.fit(model, dm)
    trainer.test(model, dm)
