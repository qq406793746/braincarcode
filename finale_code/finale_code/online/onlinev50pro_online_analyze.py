# v75_online_simulation_robust.py
# 包含：
# 1. 智能滤波器状态初始化 (消除启动震荡)
# 2. 正确的单位 (Volts) 和通道数 (25)
# 3. 严格的时间对齐 (1001 samples)

import os, sys, joblib, traceback, time
import numpy as np
import scipy.io
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne

# ================= 配置区域 =================
ARTIFACTS_DIR =r"C:\Users\Administrator\Desktop\finale_code\online\onlinev50pro"
DATA_DIR_GDF = r"D:\Backup\Downloads\bciciv_2a_gdf"
DATA_DIR_MAT = r"D:\Backup\Downloads\A0xE"

CHUNK_SIZE = 10       
WINDOW_SIZE = 1001    # 严格匹配训练时的 1001 点
FS = 250.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: 'Left', 1: 'Right', 2: 'Foot', 3: 'Tongue'}

# ================= 模型定义 =================
# (保持不变，必须包含)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class SmallTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, activation='gelu')
        self.trm = nn.TransformerEncoder(enc, num_layers)
        self.pe = PositionalEncoding(d_model)
    def forward(self, x):
        x = self.pe(x)
        return self.trm(x)

class TabNetHeadPlaceholder(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(), nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        return self.layers(x)

class EEGNetLight(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=4, csp_dim=6):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.GELU()
        )
        self.proj = nn.Linear(64, 128)
        self.trm = SmallTransformer(d_model=128, nhead=4, num_layers=2)
        self.csp_fc = nn.Sequential(nn.Linear(csp_dim, 64), nn.GELU(), nn.Dropout(0.4))
        input_dim = 128 + 64 
        self.head = TabNetHeadPlaceholder(input_dim, n_classes)
    def forward(self, x, csp_feat):
        t = self.temporal(x)
        t = t.permute(0, 2, 1)
        t = self.proj(t)
        t = self.trm(t)
        tpool = t.mean(dim=1)
        csp_out = self.csp_fc(csp_feat)
        cat = torch.cat([tpool, csp_out], dim=1)
        return self.head(cat)

# ================= 增强版在线滤波器 =================

class OnlineFilter:
    def __init__(self, n_channels, fs=250.0):
        nyq = fs / 2
        self.b_bp, self.a_bp = butter(4, [0.5/nyq, 100.0/nyq], btype='bandpass')
        self.b_notch, self.a_notch = iirnotch(50.0, 30.0, fs)
        self.n_channels = n_channels
        
        # 预计算稳态单位响应
        self.zi_bp_unit = lfilter_zi(self.b_bp, self.a_bp)
        self.zi_notch_unit = lfilter_zi(self.b_notch, self.a_notch)
        
        # 实际状态 (将在 init_state 中被赋值)
        self.zi_bp = None
        self.zi_notch = None

    def init_state(self, first_sample_vector):
        """
        关键修复：根据数据的第一个采样点初始化滤波器状态。
        这消除了 '假设从0开始' 带来的巨大阶跃响应。
        first_sample_vector shape: (n_channels,)
        """
        # 扩展维度以便广播
        # zi shape: (order, channels) approx -> Scipy format is (order,) typically
        # lfilter_zi returns shape (order,). We need (n_channels, order) for lfilter loop
        
        self.zi_bp = np.zeros((self.n_channels, len(self.zi_bp_unit)))
        self.zi_notch = np.zeros((self.n_channels, len(self.zi_notch_unit)))
        
        for i in range(self.n_channels):
            # 核心逻辑: zi = zi_unit * x[0]
            self.zi_bp[i] = self.zi_bp_unit * first_sample_vector[i]
            # Notch 级联在 Bandpass 之后。Bandpass 对 DC 的响应是 0 (因为隔直)，
            # 但瞬态时 BP 输出不为 0。为了简单，我们假设 BP 输出的初始值接近 0 (Bandpass 特性)
            # 或者更严谨地：notch 的输入是 BP 的输出。
            # 在“冷启动”瞬间，我们假设 Notch 处于 0 状态更为安全，
            # 因为 Bandpass 已经负责去除了 DC 偏移。
            self.zi_notch[i] = self.zi_notch_unit * 0.0 

    def process(self, chunk):
        n_ch = chunk.shape[0]
        filtered_chunk = np.zeros_like(chunk)
        for i in range(n_ch):
            # 如果状态未初始化（防止意外），回退到 0
            if self.zi_bp is None: 
                self.zi_bp = np.tile(self.zi_bp_unit, (self.n_channels, 1)) * 0
                self.zi_notch = np.tile(self.zi_notch_unit, (self.n_channels, 1)) * 0
                
            out_bp, self.zi_bp[i] = lfilter(self.b_bp, self.a_bp, chunk[i], zi=self.zi_bp[i])
            out_notch, self.zi_notch[i] = lfilter(self.b_notch, self.a_notch, out_bp, zi=self.zi_notch[i])
            filtered_chunk[i] = out_notch
        return filtered_chunk

class BCIStreamSimulator:
    def __init__(self, model_path, csp_path, scaler_path, n_channels=25):
        print(f"[System] Initializing Simulator with {n_channels} channels...")
        self.csp = joblib.load(csp_path)
        self.scalers = joblib.load(scaler_path)
        
        self.model = EEGNetLight(n_channels=n_channels, n_times=WINDOW_SIZE, n_classes=4, csp_dim=6).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        
        self.filter = OnlineFilter(n_channels)
        self.buffer = np.zeros((n_channels, WINDOW_SIZE), dtype=np.float32)
        self.n_channels = n_channels
        self.is_session_started = False
        
    def start_session(self, first_chunk_sample):
        """新 Trial 开始时调用"""
        self.filter.init_state(first_chunk_sample)
        self.buffer = np.zeros((self.n_channels, WINDOW_SIZE), dtype=np.float32)
        self.is_session_started = True

    def push_chunk(self, raw_chunk):
        if not self.is_session_started:
            # 自动启动
            self.start_session(raw_chunk[:, 0])
            
        filtered_chunk = self.filter.process(raw_chunk)
        n_new = filtered_chunk.shape[1]
        self.buffer = np.roll(self.buffer, -n_new, axis=1)
        self.buffer[:, -n_new:] = filtered_chunk
        
    def predict(self):
        try:
            trial_data = self.buffer[np.newaxis, :, :] 
            
            # Scaler
            temp_input = self.buffer.copy() 
            for ch in range(self.n_channels):
                scaler_idx = ch if ch < len(self.scalers) else -1
                temp_input[ch] = self.scalers[scaler_idx].transform(temp_input[ch].reshape(1, -1)).flatten()
            
            # 调试检查：Scaler 输出是否爆炸
            if np.max(np.abs(temp_input)) > 100:
                print(f"[WARN] Scaler output suspiciously large: {np.max(np.abs(temp_input)):.2f}. Check units!")

            temp_input_tensor = torch.from_numpy(temp_input[np.newaxis, :, :]).float().to(DEVICE)
            
            # CSP
            csp_feat = self.csp.transform(trial_data) 
            csp_feat_tensor = torch.from_numpy(csp_feat).float().to(DEVICE)
            
            with torch.no_grad():
                logits = self.model(temp_input_tensor, csp_feat_tensor)
                probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()[0]
            
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            return None

# ================= 数据加载 =================

def load_raw_stream(subject_id):
    gdf_path = os.path.join(DATA_DIR_GDF, f"A0{subject_id}E.gdf")
    mat_path = os.path.join(DATA_DIR_MAT, f"A0{subject_id}E.mat")
    
    print(f"\n[Data] Loading Raw Stream: {gdf_path}")
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    try: raw.pick_types(eeg=True, eog=True) 
    except: raw.pick(mne.pick_types(raw.info, eeg=True, eog=True, meg=False))
    
    # *** 关键：不乘以 1e6，保持 Volts ***
    print(f"[Data] Loaded {len(raw.ch_names)} channels. Units: Volts (Standard MNE)")

    mat_data = scipy.io.loadmat(mat_path)
    y_true = None
    if 'classlabel' in mat_data: y_true = mat_data['classlabel'].flatten()
    elif 'data' in mat_data:
        data_struct = mat_data['data']
        y_list = []
        for i in range(data_struct.shape[1]):
            run_data = data_struct[0, i]
            if 'y' in run_data.dtype.names:
                val = run_data['y']
                while val.dtype == np.object_ and val.size == 1: val = val.item()
                if isinstance(val, np.ndarray) and val.size > 0: y_list.append(val.flatten())
        if y_list: y_true = np.concatenate(y_list)
        
    if y_true is not None: y_true = y_true.astype(int) - 1 
        
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    target_codes = [v for k,v in event_id.items() if '783' in k or k in ['769','770','771','772']]
    if not target_codes: target_codes = [v for k,v in event_id.items() if '1023' not in k and '276' not in k]
    events_selected = [ev for ev in events if ev[2] in target_codes]
    events_selected = np.array(events_selected)
    
    n_min = min(len(events_selected), len(y_true))
    return raw, events_selected[:n_min], y_true[:n_min]

# ================= 主仿真循环 =================

def run_simulation(subject_id):
    model_path = os.path.join(ARTIFACTS_DIR, "best_model.pth")
    csp_path = os.path.join(ARTIFACTS_DIR, "best_csp.pkl")
    scaler_path = os.path.join(ARTIFACTS_DIR, "best_scalers.pkl")
    
    raw, events, y_true = load_raw_stream(subject_id)
    raw_data = raw.get_data() 
    n_channels_detected = raw_data.shape[0]
    
    sim = BCIStreamSimulator(model_path, csp_path, scaler_path, n_channels=n_channels_detected)
    
    print(f"\n[Sim] Starting Robust Simulation for Subject {subject_id}...")
    print(f"[Sim] Warm-up: 5.0 seconds (Smart Init enabled)")
    print("="*70)
    print(f"{'Trial':<6} | {'True':<8} | {'Pred':<8} | {'Conf':<6} | {'Result'}")
    print("-" * 70)
    
    correct = 0
    
    for i, (onset, _, _) in enumerate(events):
        true_lbl = LABEL_MAP[y_true[i]]
        
        # === 流程控制 ===
        # 1. 定义 Warm-up 起点 (5秒前)
        warmup_seconds = 5.0
        start_ptr = max(0, onset - int(warmup_seconds * FS))
        
        # 2. 取出该点的第一个样本，用于初始化滤波器
        first_sample = raw_data[:, start_ptr]
        sim.start_session(first_sample)
        
        # 3. 定义推流终点 (Cue + 1001 samples)
        end_ptr = onset + 1001
        
        curr_ptr = start_ptr
        
        # --- 模拟流式推送 ---
        while curr_ptr < end_ptr:
            chunk_len = min(CHUNK_SIZE, end_ptr - curr_ptr)
            packet = raw_data[:, curr_ptr : curr_ptr + chunk_len]
            sim.push_chunk(packet)
            curr_ptr += chunk_len
            
        # --- 触发推理 ---
        probs = sim.predict()
        
        if probs is not None:
            pred_idx = probs.argmax()
            conf = probs.max()
            pred_lbl = LABEL_MAP[pred_idx]
            
            is_corr = (pred_idx == y_true[i])
            if is_corr: correct += 1
            mark = "✅" if is_corr else "❌"
            
            print(f"#{i+1:<5} | {true_lbl:<8} | {pred_lbl:<8} | {conf:.2f}   | {mark}")
        else:
            print(f"#{i+1:<5} | {true_lbl:<8} | Error    | 0.00   | ⚠️")

    acc = correct / len(y_true)
    print("="*70)
    print(f"Final Robust Accuracy (A0{subject_id}E): {acc:.4f}")
    print("="*70)

if __name__ == "__main__":
    try:
        run_simulation(9)
    except Exception as e:
        traceback.print_exc()
