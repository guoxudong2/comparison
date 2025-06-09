# -*- coding: utf-8 -*-
"""
训练 + 同时输出 Segment-based & Event-based 指标（兼容 Lightning v2.x，从配置文件自动读取 pid）。
暂时改为 不按簇 划分，而是用随机分（StratifiedShuffleSplit），以便验证用法是否正确。
"""

import os
import math
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.stats import binom
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from typing import Optional
from functools import partial
from sklearn.model_selection import StratifiedShuffleSplit

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

# 统计 val/test 每场发作在 SOP–SPH 内剩多少片段
def _count_preictal_segments(all_t0: np.ndarray,
                             seizure_times: np.ndarray,
                             sop_sec: int = 1800,
                             sph_sec: int = 60):
    """
    打印每场发作的预警窗口内剩余多少个 segment
    """
    print("\n=== DEBUG: 每场发作在 SOP–SPH 内的 segment 数 ===")
    for idx, t_seiz in enumerate(seizure_times):
        mask = (all_t0 >= t_seiz - sop_sec) & (all_t0 <= t_seiz - sph_sec)
        print(f"Seizure {idx + 1:2d}: {mask.sum():3d} / "
              f"{(sop_sec - sph_sec) // 10:3d} 片段")
    print("==========================================\n")
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

        # 评估时 target_encoder 固定，关闭 Dropout
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
        seizure_times = np.load(os.path.join(
            "../datasets/downstream/CHB_MIT_Scalp_EEG/processed_data",
            f"{self.pid}_seizure_times.npy"
        ))

        _count_preictal_segments(all_t0, seizure_times,
                                 sop_sec=1800,
                                 sph_sec=60)

        SPH_seconds = 60
        SOP_seconds = 30 * 60
        #REFRACTORY  = 30 * 60
        REFRACTORY = 0
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print("=== 验证模型在「任何一个1就报警」条件下，预警窗口内有无 1 ===")
        for idx, t_ictal in enumerate(seizure_times):
            low = t_ictal - SOP_seconds
            high = t_ictal - SPH_seconds
            # 找到所有落在 [t_ictal-1800, t_ictal-180] 的 segment
            mask_pre = (all_t0 >= low) & (all_t0 <= high)
            preds_pre = all_preds[mask_pre]
            print(f"Event #{idx + 1} at t={t_ictal}s：窗内共有 {mask_pre.sum()} 个 segment，"
                  f"其中模型预测为 1 的数量 = {preds_pre.sum()}")
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # k-of-n 策略
        K = 10
        N = 30

        alarm_times = []
        last_alarm_time = -1e9
        for i in range(len(all_preds)):
            if i < N - 1:
                continue
            window = all_preds[i-N+1 : i+1]
            if window.sum() >= K:
                alarm_t = int(all_t0[i - N + 1])
                if alarm_t - last_alarm_time >= REFRACTORY:
                    alarm_times.append(alarm_t)
                    last_alarm_time = alarm_t
        alarm_times = np.array(alarm_times, dtype=np.int64)

        TP_event = 0
        FN_event = 0
        for t_ictal in seizure_times:
            print(f't_ictal:{t_ictal}')
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

        # 假设一切都算好了：seizure_times (M,), alarm_times (K,)
        # SPH_seconds, SOP_seconds 也都已经定义

        print("所有 alarm_times:", alarm_times)
        print("\n—— 开始检查每个发作的预警窗口 vs 报警时刻 ——")
        for idx, t_ictal in enumerate(seizure_times):
            window_low = t_ictal - SOP_seconds  # 预警窗口最早允许报警的时间
            window_high = t_ictal - SPH_seconds  # 预警窗口最晚允许报警的时间

            # 先打印出这一事件的窗口范围
            print(f"Event #{idx + 1} @ t_ictal={t_ictal}s： 窗口 = [{window_low}, {window_high}]")

            # 再遍历所有 alarm_times，看哪些报警落在这个窗口内
            hits = alarm_times[(alarm_times >= window_low) & (alarm_times <= window_high)]
            if len(hits) > 0:
                print(f"   → 窗内的报警时刻：{hits.tolist()}")
            else:
                # 如果 hits 为空，说明这个事件里“没有一次报警严格落在该窗口”
                # 我们也想知道：有没有报警落到 window_high 之后，或彻底落到 window_low 之前
                later = alarm_times[alarm_times > window_high]
                earlier = alarm_times[alarm_times < window_low]
                if len(later) > 0:
                    print(f"   → 所有报警都太“晚”了（比窗口上限 {window_high}s 晚）：{later.min()}s 等")
                if len(earlier) > 0:
                    print(f"   → 有报警落在“窗口外过早”（比窗口下限 {window_low}s 早）：{earlier.max()}s 等")
                if len(hits) == 0 and len(later) == 0 and len(earlier) == 0:
                    # 如果用户没有任何报警 (alarm_times 可能空)，就提示一下
                    print(f"   → 目前根本没有任何报警。")
            print("------------------------------------------------")
        print("—— 检查结束 ——\n")


# =========================== DataModule ===========================
class CHBMITDataModule(pl.LightningDataModule):
    def __init__(self, pid: str, cfg: DotMap):
        """
        pid: 病人 ID，例如 "chb04"
        cfg: 配置字典，需包含 data.interpolation_len, data.batch, data.shift_dir, data.val_ratio
        """
        super().__init__()
        self.pid = pid
        self.cfg = cfg

    def setup(self, stage=None):
        """
        将所有样本进行随机分层划分：
        - 用 `StratifiedShuffleSplit` 把整体数据分成 train/val（test 先用 val 来跑）。
        - 测试时直接用 val 集合做测试。
        """
        base = os.path.join(self.cfg.data.shift_dir, self.pid)
        # 直接加载所有片段的 X, y, t0
        X = np.load(base + "_X.npy")            # (N, 23, 1024)
        y = np.load(base + "_y.npy")            # (N,)
        # t0 仅作存储，训练/测试时只在 test 阶段用于 event-level 评估
        t0 = np.load(base + "_t0.npy")          # (N,)

        # 构造完整 Dataset
        full_ds = CHBMITDatasetWithT0(
            x_path=base + "_X.npy",
            y_path=base + "_y.npy",
            t0_path=base + "_t0.npy",
            use_avg=True,
            scale_div=10.0,
            interpolation_len=self.cfg.data.interpolation_len
        )

        # --- 下面开始用 StratifiedShuffleSplit 做随机分 ---
        labels = y  # shape (N,)
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.cfg.data.val_ratio,
            random_state=42
        )
        train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))

        # 创建 Subset
        self.train_ds = Subset(full_ds, train_idx.tolist())
        # self.val_ds = Subset(full_ds, val_idx.tolist())
        val_idx_sorted = np.sort(val_idx)
        self.val_ds   = Subset(full_ds, val_idx_sorted.tolist())


        # 先把训练时增强关闭标志都设置正确
        full_ds.training_flag = False
        self.train_ds.dataset.training_flag = True
        self.val_ds.dataset.training_flag = False

        # 暂时直接把 val 当成 test，用于后续 .test(...)
        self.test_ds = self.val_ds

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
        # 这里把测试集也当作 val_ds
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

    # 不再使用 fold_idx 参数
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

    tb_logger  = pl.loggers.TensorBoardLogger("./logs_eegpt/", name="EEGPT", version=pid+"_rand")
    csv_logger = pl.loggers.CSVLogger("./logs_eegpt/", name="EEGPT_csv", version=pid+"_rand")

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[cfg.train.gpu] if torch.cuda.is_available() else 1,
        logger=[tb_logger, csv_logger],
        log_every_n_steps=10,
        enable_checkpointing=False
    )

    print(f"== PID={pid} (随机分),  #train={len(dm.train_ds)}, #val={len(dm.val_ds)}, #test={len(dm.test_ds)} ==")
    trainer.fit(model, dm)
    # 这里 test 也跑同一个 val_ds
    trainer.test(model, dm)

