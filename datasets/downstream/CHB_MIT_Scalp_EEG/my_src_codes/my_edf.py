import mne
import matplotlib.pyplot as plt

raw = mne.io.read_raw_edf('./physionet.org/files/chbmit/1.0.0/chb01/chb01_01.edf', preload=True)

'''print(raw.info)
print(raw.ch_names)

data = raw.get_data() #提取原始数据
print(data.shape) #(n_channels,n_times)

sfreq = raw.info['sfreq'] #采样频率
print(f'采样率:{sfreq} Hz')
raw.plot()
plt.show()

#提取时间段数据
start = int(10 * sfreq)
stop = int(20 * sfreq)
segment = data[:, start:stop]

print(raw.ch_names)

#选指定通道
channels_of_interest = ['FP1-F7', 'F7-T7']
picks = mne.pick_channels(raw.info['ch_names'], include=channels_of_interest)

# 获取信号数据和时间轴
data, times = raw[picks, :]  # data shape: (n_channels, n_times)
data = data*1e6 #转为微伏
# 绘图
plt.figure(figsize=(12, 4))
for i, ch_name in enumerate(channels_of_interest):
    plt.plot(times, data[i] + i * 100, label=ch_name)  # 上下平移避免重叠（可调）

plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.title('Selected EEG Channels')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()'''

t_start = 0
ser = 0
duration = 4   # 每隔4秒取一次
tmax = 4     # 每段取4秒数据

# 生成事件点（每4秒一个事件）
events = mne.make_fixed_length_events(
    raw, start=t_start, stop=60, duration=duration, overlap=0.0
)

# 打印事件点
print("事件格式：[sample_index, 0, event_id]")
print(events[:5])
print(f"共生成 {len(events)} 个事件")

# 创建 epochs（每个 epoch 长 8 秒）
epochs = mne.Epochs(
    raw, events, event_id=None, tmin=0.0, tmax=tmax, baseline=None,
    preload=True, verbose=False
)

print(f"切出的 epoch 数量：{len(epochs)}")
print(f"epoch 数据 shape：{epochs.get_data().shape} （n_epochs, n_channels, n_times）")

epochs.plot(n_epochs=1, n_channels=5)
plt.show()
