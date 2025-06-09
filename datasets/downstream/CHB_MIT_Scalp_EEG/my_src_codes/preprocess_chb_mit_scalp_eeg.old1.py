# -*- coding: utf-8 -*-
"""
EEG 数据预处理：从 CHB-MIT 数据集中提取 preictal 和 interictal 样本
===============================================================

1) 从 summary.txt 或 .edf.seizures 文件中读取发作时间 (seizure_info)
2) 遍历 edf_dir 下所有 .edf 文件（无论是否记录发作），为每10秒窗口打标签：
   - preictal (1)
   - interictal (0)
   - ictal + postictal (跳过)
3) 输出 X.npy, y.npy
4) 绘制 "Labels Over Time" 图，其中横坐标是窗口的起始时间(秒)
5) 可选：检查每段发作是否被正确覆盖 preictal

你可以在代码最后的 if __name__ == "__main__": 部分，修改
edf_dir, summary_path, use_summary 等变量来进行实际处理。
"""
from __future__ import annotations

import os
import re
import numpy as np
import mne
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from collections import Counter

from numba.core.typing.builtins import Print

# ============================= 参数配置 =============================
SFREQ = 256       # 采样率 Hz
WINDOW_SIZE = 4  # 每段样本长度（秒）
STRIDE = 4       # 滑窗步长（秒）
PREICTAL_LEN = 1800    # 30 分钟 (1800)
POSTICTAL_LEN = 14400   # this is not postical_len, we choose 14 patients already considered POSTICTAL_LEN=14400
RESERVE_TIME = 60       # 保留区（1分钟）
################################################################################
#                   1) 读取发作时间：summary.txt 或 .edf.seizures               #
################################################################################

def read_summary(summary_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    从 summary.txt 中解析:
    File Name: chb01_03.edf
    Number of Seizures in File: 1
    Seizure Start Time: 2996 seconds
    Seizure End Time:   3036 seconds

    返回: {"chb01_03.edf": [(2996, 3036)], ...}
    """
    seizure_info = {}
    current_file = None
    with open(summary_path, 'r') as f:
        for line in f:
            if line.startswith("File Name:"):
                current_file = line.split()[-1]
            elif "Number of Seizures" in line:
                seizure_info[current_file] = []
            elif re.match(r"Seizure \d* *Start Time", line):
                start_time = int(re.findall(r'\d+', line)[-1])
            elif re.match(r"Seizure \d* *End Time", line):
                end_time = int(re.findall(r'\d+', line)[-1])
                seizure_info[current_file].append((start_time, end_time))
    print(f'seizure_info: {seizure_info}')
    return seizure_info

def read_all_seizure_files(edf_dir: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    从同目录下每个 .edf 对应的 .seizures 文件中解析发作时间。
    返回: {"chb01_03.edf": [(2996, 3036), (....)], ...}
    """
    seizure_info = {}
    for file in os.listdir(edf_dir):
        if file.endswith(".edf"):
            edf_path = os.path.join(edf_dir, file)
            seizure_file = edf_path + ".seizures"
            key = os.path.basename(file)
            seizure_info[key] = []
            if os.path.exists(seizure_file):
                with open(seizure_file, 'r', encoding='ISO-8859-1') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 2 and parts[0].isdigit():
                            seizure_info[key].append((int(parts[0]), int(parts[1])))
    return seizure_info

def get_label_global(t: int,
                     sz_global: list[tuple[int, int]]) -> int | None:
    """
    1 → preictal   0 → interictal   None → 舍弃
    逻辑完全遵循论文：
      • preictal: seizure_onset 前 30 min（reservation 1 min 排除）
      • interictal: 距离任何一次发作 ≥ 4 h
      • ictal / reservation / postictal(4 h) → None
    """
    # ---------- 判定 preictal (直接返回 1) -------------------------------
    for start, _ in sz_global:
        pre_s = start - PREICTAL_LEN          # 30 min 前
        res_s = start - RESERVE_TIME           # reservation 起点
        if pre_s <= t < res_s:                # 落在纯 preictal
            return 1

    # ---------- 判定 ictal / reservation / postictal → None -------------
    for start, end in sz_global:
        # reservation+ictal
        if (start - RESERVE_TIME) <= t < end:
            return None
        # postictal 4 h
        if end <= t < end + POSTICTAL_LEN:
            return None

    # ---------- interictal：距离所有发作 ≥ 4 h ---------------------------
    for start, end in sz_global:
        if abs(t - start) < POSTICTAL_LEN or abs(t - end) < POSTICTAL_LEN:
            return None                       # 距离不足 4 h
    return 0

import datetime as dt
import re

# ---------- 1) 解析文件的真实起止时刻 -------------------
def parse_clock_time(line: str) -> dt.datetime:
    # line: "File Start Time: 19:08:32"
    hh, mm, ss = map(int, re.findall(r'(\d+):(\d+):(\d+)', line)[0])
    return dt.datetime(2000, 1, 1, hh, mm, ss)   # 年月日随便给

def build_real_offsets(summary_path: str) -> dict[str, int]:
    """
    返回 {edf_name: global_seconds_offset}，时间轴以
    该患者 **第一条 EDF 的 start_time** 为零点。
    """
    base_time = None
    offsets = {}
    with open(summary_path) as f:
        current_file = None
        for ln in f:
            if ln.startswith('File Name:'):
                current_file = ln.split()[-1].strip()
            elif ln.startswith('File Start Time:'):
                t = parse_clock_time(ln)
                if base_time is None:
                    base_time = t
                offsets[current_file] = int((t - base_time).total_seconds())
    return offsets

def filter_short_gap_seizures(sz: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    输入必须按 start 升序；若相邻两次发作 start 间隔 < 30 min，则丢弃后一次
    """
    if len(sz) <= 1:
        return sz
    kept = [sz[0]]
    for s, e in sz[1:]:
        if s - kept[-1][0] >= PREICTAL_LEN + RESERVE_TIME:
            kept.append((s, e))
    return kept

TARGET_CH = [   # 23-channel 的常用顺序
    'FP1-F7','F7-T7','T7-P7','P7-O1','FP1-F3','F3-C3','C3-P3','P3-O1',
    'FP2-F4','F4-C4','C4-P4','P4-O2','FP2-F8','F8-T8','T8-P8-0','P8-O2',
    'FZ-CZ','CZ-PZ','P7-T7','T7-FT9','FT9-FT10','FT10-T8','T8-P8-1'  # 末尾两条可用占位
]
N_CH = len(TARGET_CH)          # 23，与 chb01 对齐

def process_patient(edf_dir: str, seizure_info: Dict[str, List[Tuple[int, int]]], save_prefix: str):
    X_list, y_list = [], []
    segment_start_times = []
    segment_end_times = []

    edf_start_times = build_real_offsets(os.path.join(edf_dir, f'{save_prefix}-summary.txt'))

    # 合并生成 global_seizure_times
    global_seizure_times = []
    for edf_file, times in seizure_info.items():
        times = sorted(times, key=lambda x: x[0])
        times = filter_short_gap_seizures(times)
        file_offset = edf_start_times.get(edf_file, 0)
        for start, end in times:
            global_seizure_times.append((file_offset + start, file_offset + end))

    # 遍历所有 .edf 文件
    all_edf_files = sorted(f for f in os.listdir(edf_dir) if f.endswith(".edf"))

    for edf_file in all_edf_files:
        edf_path = os.path.join(edf_dir, edf_file)
        file_offset = edf_start_times.get(edf_file, 0)

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.filter(0.5, 60, fir_design='firwin', verbose=False)
        missing = [ch for ch in TARGET_CH if ch not in raw.ch_names]
        if missing:
            pad = np.zeros((len(missing), raw.n_times))
            raw_add = mne.io.RawArray(pad,
                                      mne.create_info(missing, raw.info['sfreq'], ch_types='eeg'))
            raw.add_channels([raw_add])

        # 只保留且按顺序重排到 TARGET_CH
        raw.reorder_channels(TARGET_CH)
        data = raw.get_data()
        duration = data.shape[1] / SFREQ

        file_y = []
        for start_sec in range(0, int(duration - WINDOW_SIZE + 1), STRIDE):
            global_start_sec = file_offset + start_sec
            label = get_label_global(global_start_sec, global_seizure_times)
            if label is None:
                continue

            start_idx = int(start_sec * SFREQ)
            end_idx = start_idx + int(WINDOW_SIZE * SFREQ)
            segment = data[:, start_idx:end_idx]

            X_list.append(segment)
            y_list.append(label)
            segment_start_times.append(global_start_sec)
            segment_end_times.append(global_start_sec + WINDOW_SIZE)

            file_y.append(label)

        if file_y:
            c = Counter(file_y)
            print(f"[{edf_file}] => interictal={c.get(0,0)}, preictal={c.get(1,0)}")

    # 转为 numpy
    T_fixed = WINDOW_SIZE * SFREQ  # 4 s × 256 Hz = 1024 采样点
    fixed_segments = []
    for seg in X_list:  # seg.shape = (C, T_var)
        if seg.shape[1] < T_fixed:  # 不足则右侧补零
            pad = np.zeros((seg.shape[0], T_fixed - seg.shape[1]),
                           dtype=seg.dtype)
            seg = np.concatenate([seg, pad], axis=1)
        elif seg.shape[1] > T_fixed:  # 过长则截断
            seg = seg[:, :T_fixed]
        fixed_segments.append(seg)

    # 转成规则三维数组 (N, C, T_fixed)，再保存
    X = np.stack(fixed_segments, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    print(f"Final: X.shape={X.shape}, y.shape={y.shape}")
    c_all = Counter(y)
    print(f"Overall label count: interictal={c_all.get(0,0)}, preictal={c_all.get(1,0)}")

    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)

    # 存到子目录下
    x_path = os.path.join(output_dir, f"{save_prefix}_X.npy")
    y_path = os.path.join(output_dir, f"{save_prefix}_y.npy")
    np.save(x_path, X)
    np.save(y_path, y)

    # 绘图
    '''if len(segment_start_times) > 0:
        plt.figure(figsize=(350, 5))

        # 1) step-plot 显示所有样本的 label (0/1)
        plt.plot(segment_start_times, y, drawstyle='steps-post', label='Label(0/1)')

        # 2) 给每个分块画起止竖线(可选)
        for s, e in zip(segment_start_times, segment_end_times):
            plt.axvline(x=s, color='gray', alpha=0.2, linewidth=0.5)
            plt.axvline(x=e, color='gray', alpha=0.2, linewidth=0.5)

        # 3) 在 y=1 位置，画出 label=1 的分块
        for i, lab in enumerate(y):
            if lab == 1:
                s = segment_start_times[i]
                e = segment_end_times[i]
                # 用水平线表示这段是 preictal
                plt.hlines(y=1, xmin=s, xmax=e, color='green', linewidth=2, alpha=0.8)
                # 在顶部加个文字标注
                mid = (s + e) / 2
                plt.text(mid, 1.05, f"{s}~{e}", color='green', ha='center', va='bottom', rotation=30, fontsize=8)

        # 4) 画 seizure 区间
        for start, end in global_seizure_times:
            plt.axvspan(start, end, color='red', alpha=0.2)
            plt.text((start + end) / 2, 1.2, f"{start}-{end}", ha='center', va='bottom', fontsize=8, rotation=45, color='red')

        for s in segment_start_times:
            plt.text(s, -0.15, str(s), rotation=90, fontsize=6, ha='center', va='top', color='gray')

        plt.title("Segment vs Label (with Seizure in Red, Preictal Segments in Green)")
        plt.xlabel("Global Time (s)")
        plt.ylabel("Label (0=interictal, 1=preictal)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"{save_prefix}_label_timeline.png")
        plt.savefig(fig_path)
        print(f"Plot saved: {fig_path}")'''


def shift_labels(arr: np.ndarray, step: int) -> np.ndarray:
    """
    把标签向后平移 step 个窗口；最后 step 个位置置 -1 方便过滤。
    """
    out = np.roll(arr, -step)
    out[-step:] = -1
    return out


def shift_and_save_patient(
        x_path: str,
        y_path: str,
        step: int = 1,
        out_dir: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    读入 chbXX_X.npy / _y.npy → 右移 step，滤掉 -1，保存到 out_dir。
    返回 (X_shifted, y_shifted) 以便后续直接使用或统计。
    """
    X = np.load(x_path,allow_pickle=True)
    y = np.load(y_path,allow_pickle=True)
    y_shift = shift_labels(y, step)
    mask = (y_shift != -1)

    X_shift = X[mask]
    y_shift = y_shift[mask]

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(x_path), "shifted")
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.basename(x_path).replace("_X.npy", "")
    np.save(os.path.join(out_dir, f"{base}_X_shift.npy"), X_shift)
    np.save(os.path.join(out_dir, f"{base}_y_shift.npy"), y_shift)

    print(f"[{base}] shift={step},   keep={len(y_shift)}  "
          f"(pos={y_shift.sum()}, neg={len(y_shift)-y_shift.sum()})")
    return X_shift, y_shift

if __name__ == "__main__":
    base_dir = "./physionet.org/files/chbmit/1.0.0"
    # True：使用 summary.txt；False：使用 .edf.seizures
    use_summary = True

    # 仅限论文 Table-1 中的 14 位患者
    patient_ids = {
        "chb01", "chb03", "chb05", "chb06", "chb07", "chb08", "chb09",
        "chb10", "chb14", "chb20", "chb21", "chb23"
    }

    # 遍历符合目录 + 在名单里的患者
    patient_dirs = sorted(
        d for d in os.listdir(base_dir)
        if d in patient_ids and os.path.isdir(os.path.join(base_dir, d))
    )

    for patient_id in patient_dirs:
        edf_dir = os.path.join(base_dir, patient_id)
        summary_path = os.path.join(edf_dir, f"{patient_id}-summary.txt")
        print(f"\n=== Processing {patient_id} ===")

        try:
            # 读取发作信息
            if use_summary:
                seizure_info = read_summary(summary_path)
            else:
                seizure_info = read_all_seizure_files(edf_dir)

            # 批量预处理并保存：chbXX_X.npy & chbXX_y.npy
            process_patient(edf_dir, seizure_info, save_prefix=patient_id)

        except Exception as e:
            print(f"[Error] {patient_id}: {e}")

    SHIFT_STEP = 1  # 向后平移一个 4-s 窗口
    OUT_SUBDIR = "shifted"  # 保存到 processed_data/shifted 目录
    # ------------------------------------------------------------------

    for pid in patient_ids:
        x_path = os.path.join('./processed_data', f"{pid}_X.npy")
        y_path = os.path.join('./processed_data', f"{pid}_y.npy")
        shift_and_save_patient(
            x_path, y_path,
            step=SHIFT_STEP,
            out_dir=os.path.join('./processed_data', OUT_SUBDIR)
        )

