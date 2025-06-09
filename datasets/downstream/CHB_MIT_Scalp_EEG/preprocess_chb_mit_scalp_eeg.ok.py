# -*- coding: utf-8 -*-
"""
EEG 数据预处理：从 CHB-MIT 数据集中提取 preictal 和 interictal 样本
===============================================================

1) 从 summary.txt 或 .edf.seizures 文件中读取发作时间 (seizure_info)
2) 遍历 edf_dir 下所有 .edf 文件（无论是否记录发作），为每4秒窗口打标签：
   - preictal (1)：步长 2 秒（50% overlap）
   - interictal (0)：步长 4 秒（无 overlap）
   - ictal + postictal (跳过)
3) 收集完所有 segment → 对 interictal 做等距采样到与 preictal 数量相同
4) 输出 X.npy, y.npy
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

# ============================= 参数配置 =============================
SFREQ = 256       # 采样率 Hz
WINDOW_SIZE = 4   # 每段样本长度（秒）
PREICTAL_STRIDE = 2    # preictal 的小步长（50% overlap）
INTERICTAL_STRIDE = 4  # interictal 的大步长（no overlap）
PREICTAL_LEN = 1800    # 30 分钟 (1800s)
RESERVE_TIME = 60      # 保留区（1分钟）
POSTICTAL_LEN = 14400  # 4 小时 (postictal)
LEAD_LEN = 1800        # 聚簇时判定相隔 30 分钟内的发作为一簇

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
    seizure_info: Dict[str, List[Tuple[int,int]]] = {}
    current_file: str | None = None
    with open(summary_path, 'r') as f:
        for line in f:
            if line.startswith("File Name:"):
                current_file = line.split()[-1]
            elif "Number of Seizures" in line:
                # 确保 key 存在，后面会 append
                seizure_info[current_file] = []
            elif re.match(r"Seizure \d* *Start Time", line):
                start_time = int(re.findall(r'\d+', line)[-1])
            elif re.match(r"Seizure \d* *End Time", line):
                end_time = int(re.findall(r'\d+', line)[-1])
                seizure_info[current_file].append((start_time, end_time))
    print(f"读取到的 seizure_info: {seizure_info}")
    return seizure_info

def read_all_seizure_files(edf_dir: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    从同目录下每个 .edf 对应的 .seizures 文件中解析发作时间。
    返回: {"chb01_03.edf": [(2996, 3036), (...)]}
    """
    seizure_info: Dict[str, List[Tuple[int,int]]] = {}
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
                     regions: list[tuple[str,int,int]]) -> int | None:
    """
    根据标签区间返回 0/1/None
        1   — preictal
        0   — interictal
        None— ictal + postictal(4 h) 或 保留区（reserve）
    """
    for tag, s, e in regions:
        if s <= t < e:
            if tag == 'pre':
                return 1
            else:  # 'skip'
                return None
    return 0

import datetime as dt
import re

def parse_clock_time(line: str) -> dt.datetime:
    """
    解析 summary.txt 里记录的 “File Start Time: HH:MM:SS” 为 datetime 对象，
    这里只随便给一个同一天的年月日作为占位。
    """
    hh, mm, ss = map(int, re.findall(r'(\d+):(\d+):(\d+)', line)[0])
    return dt.datetime(2000, 1, 1, hh, mm, ss)

def _hms_to_sec(hh, mm, ss) -> int:
    return int(hh)*3600 + int(mm)*60 + int(ss)

def build_real_offsets(edf_dir: str) -> dict[str, int]:
    """
    读取每条 EDF 的绝对测量起点(info['meas_date'])，以第一条文件的 meas_date 为零点，
    返回 {edf_name: offset_sec}. 这样可完整保留跨文件的间隔(跨午夜也OK)。
    """
    edf_files = sorted(f for f in os.listdir(edf_dir) if f.endswith('.edf'))
    offsets: dict[str,int] = {}

    # 拿第一条 EDF 的 meas_date 作为基准
    first_raw = mne.io.read_raw_edf(os.path.join(edf_dir, edf_files[0]),
                                    preload=False, verbose=False)
    t0 = first_raw.info['meas_date']  # datetime.datetime
    first_raw.close()

    for fname in edf_files:
        raw = mne.io.read_raw_edf(os.path.join(edf_dir, fname),
                                  preload=False, verbose=False)
        t      = raw.info['meas_date']
        dur    = raw.n_times / raw.info['sfreq']
        offset = (t - t0).total_seconds()
        offsets[fname] = int(offset)
        raw.close()

    return offsets

def cluster_seizures(times: list[tuple[int,int]],
                     lead_len: int = LEAD_LEN) -> list[list[tuple[int,int]]]:
    """
    同一簇判定：上一场 end + 30 min > 下一场 start
    返回分簇后的 seizure 时间列表，每个簇内部的 seizure 时间互相间隔 < lead_len (1800s)。
    """
    if not times:
        return []
    clusters: list[list[tuple[int,int]]] = [[times[0]]]
    last_end = times[0][1]
    for s, e in times[1:]:
        if last_end + lead_len > s:
            clusters[-1].append((s, e))
        else:
            clusters.append([(s, e)])
        last_end = e
    return clusters

# 23 通道的常用顺序 (与 CHB01 数据对齐)
TARGET_CH = [
    'FP1-F7','F7-T7','T7-P7','P7-O1','FP1-F3','F3-C3','C3-P3','P3-O1',
    'FP2-F4','F4-C4','C4-P4','P4-O2','FP2-F8','F8-T8','T8-P8-0','P8-O2',
    'FZ-CZ','CZ-PZ','P7-T7','T7-FT9','FT9-FT10','FT10-T8','T8-P8-1'
]
N_CH = len(TARGET_CH)  # 23，与 CHB01 对齐

def process_patient(edf_dir: str,
                    seizure_info: Dict[str, List[Tuple[int, int]]],
                    save_prefix: str):
    """
    对单个患者 (edf_dir)：
    1) 读取各个 .edf 的 meas_date 得到全局偏移 offsets
    2) 合并得到 global_seizure_times (单位：秒)
    3) 聚簇、生成 label_regions (preictal, skip)
    4) 对每个 EDF 做 2 种滑窗扫描（preictal / interictal），先 collect 所有 segment
    5) 扫描完后，把 interictal 等距采样到跟 preictal 数量相同，最终得到 X, y
    6) 返回 interictal 数量、preictal 数量、clusters、seizure_report
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    segment_start_times: list[int] = []
    segment_end_times: list[int] = []

    # 1) 读取每个 EDF 的偏移量 (单位 s)
    edf_start_times = build_real_offsets(edf_dir)

    # 2) 合并所有发作时间，构造全局 (offset + start, offset + end)
    global_seizure_times: list[tuple[int,int]] = []
    for edf_file, times in seizure_info.items():
        offset = edf_start_times.get(edf_file, 0)
        for (s, e) in times:
            global_seizure_times.append((offset + s, offset + e))
    global_seizure_times.sort(key=lambda x: x[0])

    # 3) 4h 聚簇
    clusters = cluster_seizures(global_seizure_times, LEAD_LEN)
    print("clusters =", clusters)

    # 生成 label_regions: [('pre', start, end), ('skip', start, end), ...]
    label_regions: list[tuple[str,int,int]] = []
    for cl in clusters:
        lead_start, _ = cl[0]
        _, tail_end = cl[-1]
        # preictal：lead_start - 1800 到 lead_start - 60
        label_regions.append((
            'pre',
            lead_start - PREICTAL_LEN - RESERVE_TIME,
            lead_start - RESERVE_TIME
        ))
        # skip：lead_start - 60 到 tail_end + 14400
        label_regions.append((
            'skip',
            lead_start - RESERVE_TIME,
            tail_end + POSTICTAL_LEN
        ))

    # 打印一下各个 preictal region 的 segment 数量预估
    for region in label_regions:
        if region[0] == 'pre':
            s, e = region[1], region[2]
            n = (e - s) // PREICTAL_STRIDE
            print(f"preictal region: {s}-{e}, segment count (approx) = {n}")

    # 4) 遍历所有 .edf 文件，分别以两种 stride 扫描，先把所有 segment 收集起来
    all_edf_files = sorted(f for f in os.listdir(edf_dir) if f.endswith(".edf"))
    for edf_file in all_edf_files:
        edf_path = os.path.join(edf_dir, edf_file)
        file_offset = edf_start_times.get(edf_file, 0)

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        # 如果缺少某些通道就补零
        missing = [ch for ch in TARGET_CH if ch not in raw.ch_names]
        if missing:
            pad = np.zeros((len(missing), raw.n_times))
            raw_add = mne.io.RawArray(pad,
                                      mne.create_info(missing, raw.info['sfreq'], ch_types='eeg'))
            raw.add_channels([raw_add])

        # 只保留且按 TARGET_CH 顺序排列
        raw.reorder_channels(TARGET_CH)
        # 滤波 & notch
        raw.filter(l_freq=1.0, h_freq=None, fir_design='firwin', picks='eeg')
        raw.notch_filter(freqs=[60, 120], picks='eeg', notch_widths=6)

        data = raw.get_data()  # shape = (N_channels, N_times)
        duration = data.shape[1] / SFREQ  # 单位秒

        file_y = []

        # 两个循环：先 preictal (stride=2s, label=1)，再 interictal (stride=4s, label=0)
        for stride, target_label in [(PREICTAL_STRIDE, 1), (INTERICTAL_STRIDE, 0)]:
            for start_sec in range(0, int(duration - WINDOW_SIZE + 1), stride):
                global_start_sec = file_offset + start_sec
                label = get_label_global(global_start_sec, label_regions)
                # 只保留目标标签
                if label != target_label:
                    continue

                start_idx = int(start_sec * SFREQ)
                end_idx = start_idx + int(WINDOW_SIZE * SFREQ)
                segment = data[:, start_idx:end_idx]  # (C, 1024)

                X_list.append(segment)
                y_list.append(label)
                segment_start_times.append(global_start_sec)
                segment_end_times.append(global_start_sec + WINDOW_SIZE)
                file_y.append(label)

        if file_y:
            c = Counter(file_y)
            print(f"[{edf_file}] => interictal={c.get(0,0)}, preictal={c.get(1,0)}")

    # 5) 把收集到的 segment 转 numpy，(N, C, T_fixed) -> X; (N,) -> y
    T_fixed = WINDOW_SIZE * SFREQ  # 4s × 256Hz = 1024
    fixed_segments: list[np.ndarray] = []
    for seg in X_list:  # seg.shape = (C, T_var)
        if seg.shape[1] < T_fixed:
            pad = np.zeros((seg.shape[0], T_fixed - seg.shape[1]), dtype=seg.dtype)
            seg = np.concatenate([seg, pad], axis=1)
        elif seg.shape[1] > T_fixed:
            seg = seg[:, :T_fixed]
        fixed_segments.append(seg)

    X = np.stack(fixed_segments, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    # 6) interictal 等距采样到与 preictal 数量相同
    idx_pos = np.where(y == 1)[0]  # preictal 索引
    idx_neg = np.where(y == 0)[0]  # interictal 索引
    n_pos = len(idx_pos)

    if len(idx_neg) > n_pos:
        # np.linspace 在 [0, len(idx_neg)-1] 区间均匀采样 n_pos 个点
        # dtype=int 会向下取整
        choice_indices = np.linspace(0, len(idx_neg)-1, n_pos, dtype=int)
        keep_neg = idx_neg[choice_indices]
        keep_idx = np.sort(np.concatenate([idx_pos, keep_neg]))
        X = X[keep_idx]
        y = y[keep_idx]

    # =============================================================
    print(f"Final: X.shape = {X.shape}, y.shape = {y.shape}")
    c_all = Counter(y)
    inter_cnt = c_all.get(0, 0)
    pre_cnt = c_all.get(1, 0)
    print(f"Overall label count: interictal = {inter_cnt}, preictal = {pre_cnt}")

    # 保存到 processed_data 子目录
    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)
    x_path = os.path.join(output_dir, f"{save_prefix}_X.npy")
    y_path = os.path.join(output_dir, f"{save_prefix}_y.npy")
    print(f'preprocess X.shape:{X.shape}')
    print(f'preprocess y.shape:{y.shape}')
    np.save(x_path, X)
    np.save(y_path, y)

    # 7) 统计每场 seizure 实际保留的 preictal 段
    all_edf_files = sorted(f for f in os.listdir(edf_dir) if f.endswith(".edf"))
    seizure_report = report_pre_segments(clusters,
                                         label_regions,
                                         edf_start_times,
                                         all_edf_files,
                                         edf_dir)
    return (inter_cnt, pre_cnt, clusters, seizure_report)


def report_pre_segments(clusters,
                        label_regions,
                        edf_start_times,
                        all_edf_files,
                        edf_dir):
    """
    对每个 cluster 内的每个 seizure，统计真正被保留的 preictal 窗数量。
    返回 [(c_idx, z_idx, s, e, expect, kept), ...]
    """
    report: list[tuple[int,int,int,int,int,int]] = []
    for c_idx, cl in enumerate(clusters, 1):
        print(f"\n▶ Cluster {c_idx}")
        for z_idx, (s, e) in enumerate(cl, 1):
            expect = PREICTAL_LEN // PREICTAL_STRIDE  # 理论≈900（如果 stride=2）
            pre_s = s - PREICTAL_LEN - RESERVE_TIME
            pre_e = s - RESERVE_TIME

            kept = 0
            for f in all_edf_files:
                offset = edf_start_times[f]
                raw = mne.io.read_raw_edf(os.path.join(edf_dir, f),
                                          preload=False, verbose=False)
                dur = raw.n_times / raw.info['sfreq']
                raw.close()
                # 只在 preictal 区间内扫描，stride 用 PREICTAL_STRIDE
                for st in range(0, int(dur - WINDOW_SIZE + 1), PREICTAL_STRIDE):
                    g = offset + st
                    if pre_s <= g < pre_e and get_label_global(g, label_regions) == 1:
                        kept += 1

            print(f"  seizure {z_idx:<2d} ({s:>6d}-{e:<6d})  "
                  f"expected={expect:<3d}   kept={kept}")
            report.append((c_idx, z_idx, s, e, expect, kept))
    return report


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
    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
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
        "chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08",
        "chb09", "chb10", "chb11", "chb12", "chb13", "chb14", "chb15", "chb16",
        "chb17", "chb18", "chb19", "chb20", "chb21", "chb22", "chb23"
    }

    # 遍历符合目录 + 在名单里的患者
    patient_dirs = sorted(
        d for d in os.listdir(base_dir)
        if d in patient_ids and os.path.isdir(os.path.join(base_dir, d))
    )

    interictal_preictal_cnts = []
    patient_summaries: Dict[str, Dict[str, object]] = {}
    for patient_id in patient_dirs:
        edf_dir = os.path.join(base_dir, patient_id)
        if patient_id!='chb01'and patient_id!='chb02'and patient_id!='chb03':
            continue
        print(f"\n====== patient id: {patient_id} ======")
        summary_path = os.path.join(edf_dir, f"{patient_id}-summary.txt")
        try:
            if use_summary:
                seizure_info = read_summary(summary_path)
            else:
                seizure_info = read_all_seizure_files(edf_dir)

            inter_cnt, pre_cnt, cl, report = process_patient(
                edf_dir, seizure_info, save_prefix=patient_id
            )
            interictal_preictal_cnts.append((patient_id, inter_cnt, pre_cnt))
            patient_summaries[patient_id] = {
                'clusters': cl,
                'report': report
            }
        except Exception as e:
            print(f"[Error] {patient_id}: {e}")

    print("\n================  INTERICTAL / PREICTAL  NUM  ================\n")
    for pid, inter_cnt, pre_cnt in interictal_preictal_cnts:
        print(f"{pid:<6s}  {inter_cnt:>10d}  {pre_cnt:>9d}")

    print("\n================  CLUSTER / PREICTAL  DETAIL  ================\n")
    for pid, summary in patient_summaries.items():
        print(f"========== Patient {pid} ==========")
        print("clusters =", summary['clusters'])
        for (c_idx, z_idx, s, e, exp, kept) in summary['report']:
            print(f"  C{c_idx:02d}-S{z_idx:02d} ({s:>6d}-{e:<6d}) "
                  f"expected={exp:<3d}  kept={kept}")
        print()

    SHIFT_STEP = 1  # 向后平移一个 4-s 窗口
    OUT_SUBDIR = "shifted"  # 保存到 processed_data/shifted 目录

    # 如果后面需要 shift-and-save，再打开这个循环：
    # for pid in patient_ids:
    #     x_path = os.path.join('./processed_data', f"{pid}_X.npy")
    #     y_path = os.path.join('./processed_data', f"{pid}_y.npy")
    #     shift_and_save_patient(
    #         x_path, y_path,
    #         step=SHIFT_STEP,
    #         out_dir=os.path.join('./processed_data', OUT_SUBDIR)
    #     )
