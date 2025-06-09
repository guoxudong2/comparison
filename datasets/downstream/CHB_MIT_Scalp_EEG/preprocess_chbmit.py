# -*- coding: utf-8 -*-
"""
EEG 数据预处理：从 CHB-MIT 数据集中提取 preictal 和 interictal 样本，同时
保存每个 segment 的起始时间戳（t0）和每个病人完整的 seizure_times 列表。

输出到 processed_data/ 子目录下：
  {pid}_X.npy            # (N, C, T_fixed)
  {pid}_y.npy            # (N,)
  {pid}_t0.npy           # (N,)  – 每个样本在全局时间轴上的起始秒数（以第一条 EDF 为时间零点）
  {pid}_seizure_times.npy  # (M,) – 真实发作簇的“第一个引领发作”开始时刻列表（单位：秒）
"""

from __future__ import annotations

import os
import re
import numpy as np
import mne
from typing import List, Tuple, Dict
from collections import Counter
import datetime as dt

# ========================= 参数配置 (与论文中一致) =========================
SFREQ = 256        # 采样率 256 Hz
WINDOW_SIZE = 4    # 每个样本长度 4 秒
PREICTAL_STRIDE = 2    # preictal 区域采样步长 2 秒（50% overlap）
INTERICTAL_STRIDE = 4  # interictal 区域采样步长 4 秒（no overlap）
PREICTAL_LEN = 1800    # 30 分钟 = 1800 s
RESERVE_TIME = 60      # 保留区（reserve time）= 1 分钟 = 60 s
POSTICTAL_LEN = 14400  # 4 小时 = 14400 s
LEAD_LEN = 1800        # 30 分钟 = 1800 s，用于“聚簇”判断

# 23 通道顺序 (用于补齐 CHB01 常用通道)
TARGET_CH = [
    'FP1-F7','F7-T7','T7-P7','P7-O1','FP1-F3','F3-C3','C3-P3','P3-O1',
    'FP2-F4','F4-C4','C4-P4','P4-O2','FP2-F8','F8-T8','T8-P8-0','P8-O2',
    'FZ-CZ','CZ-PZ','P7-T7','T7-FT9','FT9-FT10','FT10-T8','T8-P8-1'
]
N_CH = len(TARGET_CH)  # 23

################################################################################
# 1) 从 summary.txt 读取发作时间
################################################################################
def read_summary(summary_path: str) -> Dict[str, List[Tuple[int,int]]]:
    """
    从 summary.txt 中解析出每个 EDF 文件里真正发作的(开始, 结束)秒数。
    返回：{"chb01_03.edf": [(2996, 3036), (xxx,xxx)], ...}
    """
    seizure_info: Dict[str, List[Tuple[int,int]]] = {}
    current_file: str | None = None

    with open(summary_path, 'r') as f:
        for line in f:
            if line.startswith("File Name:"):
                current_file = line.split()[-1]  # 例如 "chb01_03.edf"
            elif "Number of Seizures" in line:
                seizure_info[current_file] = []
            elif re.match(r"Seizure \d* *Start Time", line):
                start_time = int(re.findall(r'\d+', line)[-1])
            elif re.match(r"Seizure \d* *End Time", line):
                end_time = int(re.findall(r'\d+', line)[-1])
                seizure_info[current_file].append((start_time, end_time))
    print(f"[read_summary] seizure_info = {seizure_info}")
    return seizure_info


################################################################################
# 2) 计算全局偏移：把每条 EDF 的 meas_date 对齐到第一个文件为 t0=0
################################################################################
def build_real_offsets(edf_dir: str) -> Dict[str,int]:
    """
    读取每个 .edf 文件的 meas_date（绝对 UTC 时间），以第一条为零点，
    返回 { "chb01_01.edf": 0, "chb01_02.edf": 1800, ... } 偏移秒数，
    这样跨文件的时间可以无缝拼接（例如跨午夜也没问题）。
    """
    edf_files = sorted([f for f in os.listdir(edf_dir) if f.endswith('.edf')])
    offsets: Dict[str,int] = {}

    # 以列表里第一个文件为参考 t0
    first_raw = mne.io.read_raw_edf(os.path.join(edf_dir, edf_files[0]),
                                    preload=False, verbose=False)
    t0 = first_raw.info['meas_date']  # datetime 对象
    first_raw.close()

    for fname in edf_files:
        raw = mne.io.read_raw_edf(os.path.join(edf_dir, fname), preload=False, verbose=False)
        t_meas = raw.info['meas_date']  # 这条记录的测量起点 datetime
        offset = (t_meas - t0).total_seconds()
        offsets[fname] = int(offset)
        raw.close()

    print(f"[build_real_offsets] offsets = {offsets}")
    return offsets


################################################################################
# 3) 聚簇：如果“上一次发作结束 + LEAD_LEN” > 下一次发作开始，就同簇
################################################################################
def cluster_seizures(times: List[Tuple[int,int]], lead_len: int = LEAD_LEN) -> List[List[Tuple[int,int]]]:
    """
    times: [(s1,e1), (s2,e2), ...] 都是全局秒数 (offset + 本地秒数)
    按照 lead_len (1800s) 将相邻发作聚在一起。
    返回 list of clusters，例如：
      [
        [(100,110), (1400,1450)],   # 如果 110 + 1800 > 1400，则同簇
        [(5000,5050)],
        ...
      ]
    """
    if not times:
        return []

    clusters = [[times[0]]]
    last_end = times[0][1]
    for (s,e) in times[1:]:
        if last_end + lead_len > s:
            clusters[-1].append((s,e))
        else:
            clusters.append([(s,e)])
        last_end = e

    print(f"[cluster_seizures] clusters = {clusters}")
    return clusters


################################################################################
# 4) 根据“簇”生成标签区间：preictal 区间 vs skip 区间
################################################################################
def get_label_global(t: int, regions: List[Tuple[str,int,int]]) -> int | None:
    """
    给定一个全局时间点 t，判断它属于哪些区间：
      - 如果落在某个 'pre' 区间里，返回 1
      - 如果落在 'skip' 区间里（ictal + postictal + reserve），返回 None
      - 否则返回 0（interictal）。
    regions: list[ ( 'pre',  start_sec, end_sec ), ( 'skip', start_sec, end_sec ), ... ]
    """
    for tag, s, e in regions:
        if s <= t < e:
            if tag == 'pre':
                return 1
            else:  # 'skip'
                return None
    return 0


################################################################################
# 5) 对单个患者做预处理：按“滑窗采样”收集 (X, y, t0)，对 interictal 做均衡，
#    最后保存 X.npy, y.npy, t0.npy, seizure_times.npy 并返回统计信息
################################################################################
def process_patient(edf_dir: str,
                    seizure_info: Dict[str,List[Tuple[int,int]]],
                    save_prefix: str):
    """
    步骤：
      1) 读取各 EDF 全局偏移 offsets
      2) 合并所有 (offset+s, offset+e) → global_seizure_times
      3) 对 global_seizure_times 按 LEAD_LEN 做聚簇 clusters
      4) 生成 label_regions = [('pre', pre_start, pre_end), ('skip', skip_start, skip_end), ...]
         其中：
           pre_start = lead_start - PREICTAL_LEN - RESERVE_TIME
           pre_end   = lead_start - RESERVE_TIME
           skip_start = lead_start - RESERVE_TIME
           skip_end   = tail_end + POSTICTAL_LEN
      5) 依次遍历每个 EDF 文件，做两次滑窗：一次 stride=PREICTAL_STRIDE (label=1)，一次 stride=INTERICTAL_STRIDE(label=0)。
         逐个 segment 计算它的全局起始秒 global_start_sec = offset + start_sec
         然后调用 get_label_global(global_start_sec, label_regions) 决定要不要收录到 X_list, y_list, t0_list。
      6) 把收集好的 X_list（(C, T_var)）都 pad/truncate 到 (C, T_fixed) → X.shape=(N,C,T_fixed)；y.shape=(N,)；t0.shape=(N,)
      7) 对 interictal(0) 做 undersampling（或 oversampling）让正负例平衡
      8) 保存：
         processed_data/{save_prefix}_X.npy
         processed_data/{save_prefix}_y.npy
         processed_data/{save_prefix}_t0.npy
         processed_data/{save_prefix}_seizure_times.npy
    """
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    t0_list: List[int] = []

    # 1) 读取每个 EDF 文件的全局偏移
    edf_start_times = build_real_offsets(edf_dir)

    # 2) 合并 “offset + start, offset + end” → global_seizure_times
    global_seizure_times: List[Tuple[int,int]] = []
    for edf_file, times in seizure_info.items():
        offset = edf_start_times.get(edf_file, 0)
        for (s,e) in times:
            global_seizure_times.append((offset + s, offset + e))
    global_seizure_times.sort(key=lambda x: x[0])

    # 3) 按 LEAD_LEN（1800s） 做聚簇
    clusters = cluster_seizures(global_seizure_times, LEAD_LEN)
    print(f'clusters:{clusters}')
    cluster_id_list: List[int] = []

    # seizure_times: 每个簇第一个引领发作的开始秒，用于 event-based 评价
    seizure_times: List[int] = [cl[0][0] for cl in clusters]

    # 4) 构造 label_regions（preictal + skip）
    label_regions: List[Tuple[str,int,int]] = []
    for cl in clusters:
        lead_start, _ = cl[0]
        _, tail_end = cl[-1]
        # preictal 区间
        pre_s = lead_start - PREICTAL_LEN - RESERVE_TIME
        pre_e = lead_start - RESERVE_TIME
        label_regions.append(("pre", pre_s, pre_e))

        # skip 区间 = [lead_start - RESERVE_TIME, tail_end + POSTICTAL_LEN)
        skip_s = lead_start - RESERVE_TIME
        skip_e = tail_end + POSTICTAL_LEN
        label_regions.append(("skip", skip_s, skip_e))

    # 5) 遍历每个 EDF 文件，先做 preictal(1) 滑窗，再做 interictal(0) 滑窗
    all_edf_files = sorted([f for f in os.listdir(edf_dir) if f.endswith(".edf")])
    for edf_file in all_edf_files:
        edf_path = os.path.join(edf_dir, edf_file)
        offset = edf_start_times.get(edf_file, 0)  # 该文件在全局时间轴上的秒偏移

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        # 如果缺少通道就补零
        missing = [ch for ch in TARGET_CH if ch not in raw.ch_names]
        if missing:
            pad = np.zeros((len(missing), raw.n_times))
            raw_add = mne.io.RawArray(pad,
                                      mne.create_info(missing, raw.info['sfreq'], ch_types='eeg'))
            raw.add_channels([raw_add])

        # 保留并按 TARGET_CH 顺序排列
        raw.reorder_channels(TARGET_CH)
        raw.filter(l_freq=1.0, h_freq=None, fir_design='firwin', picks='eeg')
        raw.notch_filter(freqs=[60, 120], picks='eeg', notch_widths=6)

        data = raw.get_data()  # shape = (23, n_times)
        dur  = data.shape[1] / SFREQ  # 秒

        # local_y_count 只是为了打印统计
        file_y: List[int] = []
        INTER_WINDOW = PREICTAL_LEN
        # 两个循环：一次 stride=PREICTAL_STRIDE, label=1；一次 stride=INTERICTAL_STRIDE, label=0
        for (stride, target_label) in [(PREICTAL_STRIDE, 1), (INTERICTAL_STRIDE, 0)]:
            for start_sec in range(0, int(dur - WINDOW_SIZE + 1), stride):
                global_start_sec = offset + start_sec
                lbl = get_label_global(global_start_sec, label_regions)
                if lbl is None or lbl != target_label:
                    continue

                start_idx = int(start_sec * SFREQ)
                end_idx   = start_idx + int(WINDOW_SIZE * SFREQ)  # WINDOW_SIZE=4s, SFREQ=256 => 1024个点
                segment = data[:, start_idx:end_idx]  # shape = (23, 1024)

                X_list.append(segment.astype(np.float32))
                y_list.append(int(lbl))
                t0_list.append(int(global_start_sec))
                file_y.append(int(lbl))

                '''if lbl == 1:
                    cid = next((i for i, cl in enumerate(clusters) if
                                cl[0][0] - PREICTAL_LEN - RESERVE_TIME <= global_start_sec < cl[-1][1] + POSTICTAL_LEN), -1)
                    if cid != -1:
                        cluster_id_list.append(cid)'''

                if lbl == 1:  # 只给正样本打簇标签
                    cid = next(i for i, cl in enumerate(clusters)
                               if cl[0][0] - PREICTAL_LEN - RESERVE_TIME
                               <= global_start_sec
                               < cl[0][0])  # **到 lead_start 截止，不往后拉 4h**

                else:
                    cid = -1  # 默认归为纯背景
                    for i, cl in enumerate(clusters):  # 看它是否在 INTER_WINDOW 内
                        lead = cl[0][0]
                        left = lead - PREICTAL_LEN - RESERVE_TIME - INTER_WINDOW
                        right = lead - PREICTAL_LEN - RESERVE_TIME
                        if left <= global_start_sec < right:
                            cid = i  # 归到同一个簇
                            break
                cluster_id_list.append(cid)
        if file_y:
            c = Counter(file_y)
            print(f"[{edf_file}] => interictal={c.get(0,0)}, preictal={c.get(1,0)}")

        raw.close()

    # 6) 把 (N, C, T_var) 统一 pad/truncate 到 (N, C, T_fixed=WINDOW_SIZE×SFREQ)
    T_fixed = WINDOW_SIZE * SFREQ  # 4s × 256Hz = 1024
    fixed_segments: List[np.ndarray] = []
    for seg in X_list:
        if seg.shape[1] < T_fixed:
            pad = np.zeros((seg.shape[0], T_fixed - seg.shape[1]), dtype=seg.dtype)
            seg = np.concatenate([seg, pad], axis=1)
        elif seg.shape[1] > T_fixed:
            seg = seg[:, :T_fixed]
        fixed_segments.append(seg)

    X = np.stack(fixed_segments, axis=0)  # (N, 23, 1024)
    y = np.array(y_list, dtype=np.int64)   # (N,)
    t0 = np.array(t0_list, dtype=np.int64) # (N,)

    print(f"Before balancing: interictal={np.sum(y==0)}, preictal={np.sum(y==1)}")

    # 7) 对 interictal(0) 做 undersampling，让正负例平衡：
    '''idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_pos = len(idx_pos)
    if len(idx_neg) > n_pos:
        # 在 idx_neg 里等距采样 n_pos 个
        choice_neg = np.linspace(0, len(idx_neg)-1, n_pos, dtype=int)
        keep_neg = idx_neg[choice_neg]
        keep_idx = np.sort(np.concatenate([idx_pos, keep_neg]))
        X = X[keep_idx]
        y = y[keep_idx]
        t0 = t0[keep_idx]
        #cluster_id_list = [cluster_id_list[i] for i in keep_idx]
        cluster_id_list = list(np.array(cluster_id_list)[keep_idx])'''

    print(f"After balancing: interictal={np.sum(y==0)}, preictal={np.sum(y==1)}")
    print(f"Final shapes: X={X.shape}, y={y.shape}, t0={t0.shape}")

    # 8) 保存到 processed_data/
    out_dir = "processed_data"
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, f"{save_prefix}_X.npy"), X)
    np.save(os.path.join(out_dir, f"{save_prefix}_y.npy"), y)
    np.save(os.path.join(out_dir, f"{save_prefix}_t0.npy"), t0)
    np.save(os.path.join(out_dir, f"{save_prefix}_seizure_times.npy"),
            np.array(seizure_times, dtype=np.int64))
    cluster_id = np.array(cluster_id_list, dtype=np.int16)
    np.save(os.path.join(out_dir, f"{save_prefix}_cluster.npy"), cluster_id)

    print(f"[process_patient] {save_prefix}  saved: X.npy, y.npy, t0.npy, seizure_times.npy")
    return (np.sum(y==0), np.sum(y==1), clusters, seizure_times)

################################################################################
# 6) 用来调试或批量处理多个患者
################################################################################
if __name__ == "__main__":
    base_dir = "./physionet.org/files/chbmit/1.0.0"  # CHB-MIT 数据根目录
    use_summary = True  # True: 使用 summary.txt；False: 使用 .edf.seizures

    # 仅处理 chb01, chb02, chb03（示例），可自行扩展到 chb01~chb23
    patient_ids = ["chb01"]#, "chb02", "chb03"]

    for pid in patient_ids:
        edf_dir = os.path.join(base_dir, pid)
        if not os.path.isdir(edf_dir):
            continue

        summary_path = os.path.join(edf_dir, f"{pid}-summary.txt")
        print(f'summary_path:{summary_path}')
        try:
            if use_summary:
                seizure_info = read_summary(summary_path)
            else:
                seizure_info = read_all_seizure_files(edf_dir)  # 如果你有 .seizures 文件

            inter_cnt, pre_cnt, clusters, seizure_times = process_patient(
                edf_dir, seizure_info, save_prefix=pid
            )
            print(f"[{pid}] interictal={inter_cnt}, preictal={pre_cnt}, #clusters={len(clusters)}")
        except Exception as e:
            print(f"[Error {pid}] 报错内容如下：")
            import traceback
            traceback.print_exc()

    print("所有患者处理完毕，文件保存在 processed_data/ 目录下。")
