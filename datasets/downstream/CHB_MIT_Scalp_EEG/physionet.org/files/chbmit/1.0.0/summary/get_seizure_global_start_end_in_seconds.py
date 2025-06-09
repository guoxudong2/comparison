import os
import re
import pandas as pd

def parse_time_sec(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

records = []
data_dir = './'

# 记录每个病人的 offset 和 seizure_id
offset_map = {}
seizure_id_map = {}

# Iterate over all summary files in the data directory
for fname in sorted(os.listdir(data_dir)):
    print("读取到文件：", fname)
    if not fname.endswith('-summary.txt'):
        continue
    patient = fname.split('-')[0]
    path = os.path.join(data_dir, fname)

    # 初始化该病人的 offset 和 seizure_id
    if patient not in offset_map:
        offset_map[patient] = 0
    if patient not in seizure_id_map:
        seizure_id_map[patient] = 1

    file_start = None
    file_end = None
    rel_starts = []
    rel_ends = []

    with open(path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith('File Name:'):
            # Process previous file block
            if file_start is not None:
                duration = parse_time_sec(file_end) - parse_time_sec(file_start)
                for rs, re_sec in zip(rel_starts, rel_ends):
                    records.append({
                        'patient': patient,
                        'seizure_id': seizure_id_map[patient],
                        'global_start_s': offset_map[patient] + rs,
                        'global_end_s': offset_map[patient] + re_sec
                    })
                    seizure_id_map[patient] += 1
                offset_map[patient] += duration

            # Reset for new file
            file_start = None
            file_end = None
            rel_starts = []
            rel_ends = []

        if line.startswith('File Start Time:'):
            file_start = line.split(': ', 1)[1].strip()
        if line.startswith('File End Time:'):
            file_end = line.split(': ', 1)[1].strip()

        m_start = re.search(r'Seizure(?: \d+)? Start Time: (\d+) seconds', line)
        if m_start:
            rel_starts.append(int(m_start.group(1)))
        m_end = re.search(r'Seizure(?: \d+)? End Time: (\d+) seconds', line)
        if m_end:
            rel_ends.append(int(m_end.group(1)))

    # Process last file block
    if file_start is not None:
        duration = parse_time_sec(file_end) - parse_time_sec(file_start)
        for rs, re_sec in zip(rel_starts, rel_ends):
            records.append({
                'patient': patient,
                'seizure_id': seizure_id_map[patient],
                'global_start_s': offset_map[patient] + rs,
                'global_end_s': offset_map[patient] + re_sec
            })
            seizure_id_map[patient] += 1
        offset_map[patient] += duration

# Create DataFrame and display
df = pd.DataFrame(records)
df = df.sort_values(by=['patient', 'global_start_s']).reset_index(drop=True)
print(df)
df.to_csv("global_seizure_times.csv", index=False)
