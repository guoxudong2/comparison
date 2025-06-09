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
PREICTAL_STRIDE = 2   # preictal 的小步长（overlap）
INTERICTAL_STRIDE = 4 # interictal 的大步长（no overlap）
PREICTAL_LEN = 1800    # 30 分钟 (1800)
RESERVE_TIME = 60       # 保留区（1分钟）
POSTICTAL_LEN = 14400
LEAD_LEN = 1800
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
                     regions: list[tuple[str,int,int]]) -> int | None:
    """
    根据标签区间返回 0/1/None
        1   — preictal
        0   — interictal
        None— ictal + postictal(4 h) 或缓冲区
    """
    for tag, s, e in regions:
        if s <= t < e:
            if tag == 'pre':
                return 1
            else:               # 'skip'
                return None
    return 0


import datetime as dt
import re

# ---------- 1) 解析文件的真实起止时刻 -------------------
def parse_clock_time(line: str) -> dt.datetime:
    # line: "File Start Time: 19:08:32"
    hh, mm, ss = map(int, re.findall(r'(\d+):(\d+):(\d+)', line)[0])
    return dt.datetime(2000, 1, 1, hh, mm, ss)   # 年月日随便给

'''def build_real_offsets(summary_path: str) -> dict[str, int]:
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
    return offsets'''

import mne, re, os, datetime as dt

def _hms_to_sec(hh, mm, ss):
    return int(hh) * 3600 + int(mm) * 60 + int(ss)

'''def build_real_offsets(edf_dir: str, summary_path: str) -> dict[str, int]:
    """
    返回 {edf_name: offset_sec}，保证：
       • offset 单调递增、不为负
       • 不依赖 summary 中是否含有 Duration
    """
    offsets = {}
    cur_offset = 0

    # 1) 先把目录里所有 EDF 按文件名排序
    edf_files = sorted([f for f in os.listdir(edf_dir) if f.endswith('.edf')])

    # 2) 把 summary 信息一次读完做索引（可没有）
    summary_txt = open(summary_path, encoding='utf-8', errors='ignore').read()

    for fname in edf_files:
        offsets[fname] = cur_offset

        # ===== 2.1 先找 Duration =====
        p_dur = rf'File Name:\s*{re.escape(fname)}.*?Duration:\s*(\d+)\s*hour.*?(\d+)\s*minute.*?(\d+)\s*second'
        m = re.search(p_dur, summary_txt, flags=re.S)
        if m:
            h, m_, s = map(int, m.groups())
            dur = h*3600 + m_*60 + s
        else:
            # ===== 2.2 再找 Start / End Time =====
            p_start = rf'File Name:\s*{re.escape(fname)}.*?File Start Time:\s*([\d:]+)'
            p_end   = rf'File Name:\s*{re.escape(fname)}.*?File End Time:\s*([\d:]+)'
            ms = re.search(p_start, summary_txt, flags=re.S)
            me = re.search(p_end,   summary_txt, flags=re.S)
            if ms and me:
                sh, sm, ss = map(int, ms.group(1).split(':'))
                eh, em, es = map(int, me.group(1).split(':'))
                t_start = _hms_to_sec(sh, sm, ss)
                t_end   = _hms_to_sec(eh, em, es)
                # 跨午夜则 +24h
                dur = (t_end - t_start) % 86400
            else:
                # ===== 2.3 最保险：去读 EDF Header =====
                raw_head = mne.io.read_raw_edf(
                    os.path.join(edf_dir, fname),
                    preload=False, verbose=False
                )
                dur = raw_head.n_times / raw_head.info['sfreq']
                ### NEW —— 如果当前文件的开始时间比上一条还早，说明跨午夜
                if cur_offset > 0 and offsets[fname] < list(offsets.values())[-1]:
                    offsets[fname] += 86400  # 补一天
                cur_offset = offsets[fname] + int(dur)

        cur_offset += int(dur)

    return offsets'''

import datetime as dt
import mne, os

def build_real_offsets(edf_dir: str) -> dict[str, int]:
    """
    读取每条 EDF 的绝对测量起点（info['meas_date']），
    以第一条文件的 meas_date 为零点，返回 {edf_name: offset_sec}.
    这样可完整保留跨文件的间隔（跨午夜也OK）。
    """
    edf_files = sorted(f for f in os.listdir(edf_dir) if f.endswith('.edf'))
    offsets = {}

    # 先拿第一条 EDF 的测量时间作为基准
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


'''def cluster_seizures(times, gap=POSTICTAL_LEN):
    """
    times: 已按 start 排序的 [(s,e), ...]
    gap : postictal 时长 (≥4h) —— 任意发作距 **簇首** start 超过 gap 就另起一簇
    """
    if not times:
        return []

    clusters = [[times[0]]]
    lead_start = times[0][0]          # 记录簇首的 start

    for s, e in times[1:]:
        if s - lead_start < gap:
            clusters[-1].append((s, e))
        else:
            clusters.append([(s, e)])
            lead_start = s            # 更新新的簇首
    return clusters'''

def cluster_seizures(times: list[tuple[int,int]],
                     lead_len: int = 1800) -> list[list[tuple[int,int]]]:
    """
    同一簇判定：上一场 end + 30 min > 下一场 start
    """
    if not times:
        return []

    clusters = [[times[0]]]
    last_end = times[0][1]                 # 用上一场结束时间做基准

    for s, e in times[1:]:
        if last_end + lead_len > s:        # 关键比较   ### NEW
            clusters[-1].append((s, e))
        else:
            clusters.append([(s, e)])      # 开新簇     ### NEW
        last_end = e                       # 更新基准   ### NEW
    return clusters

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

    edf_start_times = build_real_offsets(edf_dir)
    #edf_start_times = build_real_offsets(os.path.join(edf_dir, f'{save_prefix}-summary.txt'))
    #edf_start_times = build_real_offsets(edf_dir, os.path.join(edf_dir, f'{save_prefix}-summary.txt'))

    # 合并生成 global_seizure_times
    global_seizure_times = []
    for edf_file, times in seizure_info.items():
        offset = edf_start_times.get(edf_file, 0)
        global_seizure_times.extend([(offset + s, offset + e) for s, e in times])

    global_seizure_times = sorted(global_seizure_times, key=lambda x: x[0])

    # ② 4-h 聚簇
    clusters = cluster_seizures(global_seizure_times, LEAD_LEN)  # ← 现在得到二维列表
    print("clusters =", clusters)
    # ③ 根据每个簇生成可用的“标签区间”列表
    #    格式: [(tag, start_sec, end_sec), ...]
    #    tag = 'pre'   →  30 min preictal
    #    tag = 'skip'  →  ictal + postictal(4 h)
    label_regions: list[tuple[str, int, int]] = []
    for cl in clusters:
        lead_start, _ = cl[0]  # 簇首
        _, tail_end = cl[-1]  # 簇尾

        # preictal：lead_start 前 30 min，排除最后 1 min
        label_regions.append(('pre',
                              lead_start - PREICTAL_LEN - RESERVE_TIME,
                              lead_start - RESERVE_TIME))

        # ictal + postictal：从 lead_start 前 1 min 起直到簇尾后 4 h
        label_regions.append(('skip',
                              lead_start - RESERVE_TIME,
                              tail_end + POSTICTAL_LEN))

    for region in label_regions:
        if region[0] == 'pre':
            s, e = region[1], region[2]
            n = (e - s) // PREICTAL_STRIDE
            print(f"preictal region: {s}-{e}, segment count: {n}")

    # 遍历所有 .edf 文件
    all_edf_files = sorted(f for f in os.listdir(edf_dir) if f.endswith(".edf"))
    # ③ label_regions 构建完毕后
    report_pre_segments(clusters, label_regions, edf_start_times, all_edf_files, edf_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for edf_file in all_edf_files:
        edf_path = os.path.join(edf_dir, edf_file)
        file_offset = edf_start_times.get(edf_file, 0)

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        missing = [ch for ch in TARGET_CH if ch not in raw.ch_names]
        if missing:
            pad = np.zeros((len(missing), raw.n_times))
            raw_add = mne.io.RawArray(pad,
                                      mne.create_info(missing, raw.info['sfreq'], ch_types='eeg'))
            raw.add_channels([raw_add])

        # 只保留且按顺序重排到 TARGET_CH
        raw.reorder_channels(TARGET_CH)

        raw.filter(l_freq=1.0, h_freq=None, fir_design='firwin', picks='eeg')
        raw.notch_filter(freqs=[60, 120], picks='eeg', notch_widths=6)
        data = raw.get_data()
        duration = data.shape[1] / SFREQ

        file_y = []
        '''for start_sec in range(0, int(duration - WINDOW_SIZE + 1), STRIDE):
            global_start_sec = file_offset + start_sec
            label = get_label_global(global_start_sec, label_regions)

            if label is None:
                continue

            start_idx = int(start_sec * SFREQ)
            end_idx = start_idx + int(WINDOW_SIZE * SFREQ)
            segment = data[:, start_idx:end_idx]

            X_list.append(segment)
            y_list.append(label)
            segment_start_times.append(global_start_sec)
            segment_end_times.append(global_start_sec + WINDOW_SIZE)

            file_y.append(label)'''

        '''for stride, target_label in [(PREICTAL_STRIDE, 1), (INTERICTAL_STRIDE, 0)]:
            for start_sec in range(0, int(duration - WINDOW_SIZE + 1), stride):
                global_start_sec = file_offset + start_sec
                label = get_label_global(global_start_sec, label_regions)
                # 只保留目标标签
                if label != target_label:
                    continue
                # ↓↓↓ 下面跟原本的一样
                start_idx = int(start_sec * SFREQ)
                end_idx = start_idx + int(WINDOW_SIZE * SFREQ)
                segment = data[:, start_idx:end_idx]

                X_list.append(segment)
                y_list.append(label)
                segment_start_times.append(global_start_sec)
                segment_end_times.append(global_start_sec + WINDOW_SIZE)
                file_y.append(label)'''
        # ── ❶ 预扫：直接保存 preictal；只收集 interictal 索引 ──────────────────
        inter_idx_buf = []  # 保存 (global_start_sec, local_start_sec)
        n_preictal_total = 0

        # ① 先按 PREICTAL_STRIDE 扫一遍 —— 只为了找 preictal
        for start_sec in range(0,
                               int(duration - WINDOW_SIZE + 1),
                               PREICTAL_STRIDE):
            g_start = file_offset + start_sec
            label = get_label_global(g_start, label_regions)
            if label != 1:  # 不是 preictal 就跳过
                continue

            start_idx = start_sec * SFREQ
            seg = data[:, start_idx: start_idx + WINDOW_SIZE * SFREQ]
            X_list.append(seg)
            y_list.append(1)
            segment_start_times.append(g_start)
            segment_end_times.append(g_start + WINDOW_SIZE)
            n_preictal_total += 1
            file_y.append(1)

        # ② 再用 INTERICTAL_STRIDE 扫一遍 —— 记录 interictal 候选起点
        for start_sec in range(0,
                               int(duration - WINDOW_SIZE + 1),
                               INTERICTAL_STRIDE):
            g_start = file_offset + start_sec
            if get_label_global(g_start, label_regions) == 0:
                inter_idx_buf.append((g_start, start_sec))

        # ── ❷ 等距抽样 interictal ──────────────────────────────────────────────
        if inter_idx_buf and n_preictal_total:
            step_pick = max(1, len(inter_idx_buf) // n_preictal_total)
            picked = inter_idx_buf[::step_pick][:n_preictal_total]  # 1:1

            for g_start, start_sec in picked:
                start_idx = start_sec * SFREQ
                seg = data[:, start_idx: start_idx + WINDOW_SIZE * SFREQ]
                X_list.append(seg)
                y_list.append(0)
                segment_start_times.append(g_start)
                segment_end_times.append(g_start + WINDOW_SIZE)
                file_y.append(0)

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

    # ============ interictal 下采样到 1:1 ===============
    n_pos = int((y == 1).sum())  # preictal 数量
    idx_neg = np.where(y == 0)[0]  # interictal 索引

    '''if len(idx_neg) > n_pos:  # 只有 interictal 多时才抽样
        rng = np.random.default_rng(42)  # 固定随机种子方便复现
        keep_neg = rng.choice(idx_neg, n_pos, replace=False)
        keep_pos = np.where(y == 1)[0]
        keep_idx = np.sort(np.concatenate([keep_pos, keep_neg]))

        X = X[keep_idx]
        y = y[keep_idx]'''

    # （如果想同步裁剪 segment_*_times，可同样用 keep_idx 取子集）
    # =============================================================
    print(f"Final: X.shape={X.shape}, y.shape={y.shape}")
    c_all = Counter(y)
    inter_cnt = c_all.get(0, 0)  # ★ 新增
    pre_cnt = c_all.get(1, 0)
    print(f"Overall label count: interictal={c_all.get(0,0)}, preictal={c_all.get(1,0)}")

    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)

    # 存到子目录下
    x_path = os.path.join(output_dir, f"{save_prefix}_X.npy")
    y_path = os.path.join(output_dir, f"{save_prefix}_y.npy")
    np.save(x_path, X)
    np.save(y_path, y)

    # ===== 统计每场 seizure 实际保留的 preictal 段 =====
    all_edf_files = sorted(f for f in os.listdir(edf_dir) if f.endswith(".edf"))
    seizure_report = report_pre_segments(clusters,label_regions, edf_start_times, all_edf_files, edf_dir)
    return (inter_cnt, pre_cnt,
            clusters,  # ①
            seizure_report)  # ②

    #return c_all.get(0,0),c_all.get(1,0)
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

def report_pre_segments(clusters, label_regions, edf_start_times, all_edf_files, edf_dir):
    report = []
    for c_idx, cl in enumerate(clusters, 1):
        print(f"\n▶ Cluster {c_idx}")
        for z_idx, (s, e) in enumerate(cl, 1):
            expect = PREICTAL_LEN // PREICTAL_STRIDE      # 理论≈450
            pre_s, pre_e = s - PREICTAL_LEN - RESERVE_TIME, s - RESERVE_TIME

            kept = 0
            for f in all_edf_files:
                offset = edf_start_times[f]
                raw = mne.io.read_raw_edf(os.path.join(edf_dir, f),
                                          preload=False, verbose=False)
                dur = raw.n_times / raw.info['sfreq']
                raw.close()
                for st in range(0, int(dur - WINDOW_SIZE + 1), PREICTAL_STRIDE):
                    g = offset + st
                    if pre_s <= g < pre_e and get_label_global(g, label_regions) == 1:
                        kept += 1

            print(f"  seizure {z_idx:<2d} ({s:>6d}-{e:<6d})  "
                  f"expected={expect:<3d}   kept={kept}")
            report.append((c_idx,z_idx,s,e,expect,kept))
    return report

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

    interictal_preictal_cnts=[]
    patient_summaries = {}  # {pid: {'clusters':…, 'report':…}}
    for patient_id in patient_dirs:
        edf_dir = os.path.join(base_dir, patient_id)
        print(f'patient id: {patient_id}')
        if patient_id != 'chb01' and patient_id != 'chb02'and patient_id!='chb03':
            continue
        summary_path = os.path.join(edf_dir, f"{patient_id}-summary.txt")
        print(f"\n=== Processing {patient_id} ===")
        try:
            # 读取发作信息
            if use_summary:
                seizure_info = read_summary(summary_path)
            else:
                seizure_info = read_all_seizure_files(edf_dir)

            # 批量预处理并保存：chbXX_X.npy & chbXX_y.npy
            inter_cnt, pre_cnt, cl, report = process_patient(edf_dir, seizure_info, save_prefix=patient_id)
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

    '''for pid in patient_ids:
        x_path = os.path.join('./processed_data', f"{pid}_X.npy")
        y_path = os.path.join('./processed_data', f"{pid}_y.npy")
        shift_and_save_patient(
            x_path, y_path,
            step=SHIFT_STEP,
            out_dir=os.path.join('./processed_data', OUT_SUBDIR)
        )'''

