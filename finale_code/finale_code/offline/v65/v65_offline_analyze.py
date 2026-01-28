# v68_inference_final.py
# 最终版推理脚本：包含 Mat文件复杂结构修复 + TTA增强 + 逐样本置信度输出

import os, sys, json, joblib, traceback
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import mne

# ================= 配置区域 (根据你的报错日志已自动填好) =================

# 1. 模型文件夹路径 (确保 best_model.pth 在这里面)
# 根据你的日志，这个文件夹在当前脚本的同级目录
ARTIFACTS_DIR = "v65"

# 2. 数据文件路径
# 根据你的日志: D:\Backup\Downloads\bciciv_2a_gdf
DATA_DIR_GDF = r"D:\Backup\Downloads\bciciv_2a_gdf" 
# 根据你的日志: D:\Backup\Downloads\A0xE
DATA_DIR_MAT = r"D:\Backup\Downloads\A0xE"

# 3. 运行参数
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: 'left', 1: 'right', 2: 'foot', 3: 'tongue'}

# ================= 模型定义 (必须与训练时完全一致) =================

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
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.3):
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
        self.trm = SmallTransformer(d_model=128, nhead=4, num_layers=2, dropout=0.3)
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

# ================= 工具函数 =================

def load_artifacts():
    print(f"[INFO] Loading artifacts from {ARTIFACTS_DIR} ...")
    model_path = os.path.join(ARTIFACTS_DIR, "best_model.pth")
    csp_path = os.path.join(ARTIFACTS_DIR, "best_csp.pkl")
    scaler_path = os.path.join(ARTIFACTS_DIR, "best_scalers.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing: {model_path}\nPlease check ARTIFACTS_DIR path.")
    
    csp = joblib.load(csp_path)
    scalers = joblib.load(scaler_path)
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    print(f"   -> CSP components: {csp.n_components}")
    print(f"   -> Scalers count: {len(scalers)}")
    print("   -> Model weights loaded.")
    return csp, scalers, state_dict

def load_eval_data_robust(subject_id, tmin=0.0, tmax=4.0):
    """
    针对 Mat 文件结构混乱的终极解决方案。
    """
    gdf_name = f"A0{subject_id}E.gdf"
    mat_name = f"A0{subject_id}E.mat"
    gdf_path = os.path.join(DATA_DIR_GDF, gdf_name)
    mat_path = os.path.join(DATA_DIR_MAT, mat_name)
    
    print(f"\n[INFO] Loading Subject {subject_id}...")
    print(f"   -> GDF: {gdf_path}")
    print(f"   -> MAT: {mat_path}")
    
    # --- 1. Load GDF ---
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    try: raw.pick_types(eeg=True, eog=False)
    except: raw.pick(mne.pick_types(raw.info, eeg=True, eog=False, meg=False))
    
    raw.filter(0.5, 100., fir_design='firwin', verbose=False)
    try: raw.notch_filter(50, verbose=False)
    except: pass
    
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    # Identify cues
    target_codes = []
    for k, v in event_id.items():
        if '783' in k or k in ['769','770','771','772']:
            target_codes.append(v)
    if not target_codes:
        # Fallback heuristic
        target_codes = [v for k, v in event_id.items() 
                        if '1023' not in k and '276' not in k and '277' not in k and '32766' not in k]

    events_selected = [ev for ev in events if ev[2] in target_codes]
    events_selected = np.array(events_selected)
    
    # --- 2. Load MAT (The Fix) ---
    mat_data = scipy.io.loadmat(mat_path)
    y_true = None

    if 'classlabel' in mat_data:
        y_true = mat_data['classlabel'].flatten()
        print("   [INFO] Found simple 'classlabel'.")
        
    elif 'data' in mat_data:
        print("   [INFO] Found complex 'data' struct. Unwrapping...")
        data_struct = mat_data['data'] 
        y_list = []
        
        # Iterate through all runs (e.g., 9 runs)
        for i in range(data_struct.shape[1]):
            run_data = data_struct[0, i]
            
            if 'y' in run_data.dtype.names:
                val = run_data['y']
                
                # Recursive unwrap for nested objects (The key fix for TypeError)
                while val.dtype == np.object_ and val.size == 1:
                    val = val.item()
                
                # If we finally got a numpy array with data
                if isinstance(val, np.ndarray) and val.size > 0:
                    y_list.append(val.flatten())
                    
        if y_list:
            y_true = np.concatenate(y_list)
            print(f"   [INFO] Successfully extracted {len(y_true)} labels from {len(y_list)} runs.")
        else:
            raise ValueError("Parsed 'data' struct but found no valid labels inside.")
    else:
        raise KeyError(f"Unknown MAT format. Keys found: {list(mat_data.keys())}")

    # --- 3. Post-process Labels ---
    if y_true is not None:
        y_true = y_true.astype(int)
        # Adjust 1-4 to 0-3
        if np.min(y_true) == 1 and np.max(y_true) == 4:
            y_true = y_true - 1
            print("   [INFO] Labels adjusted: 1-4 -> 0-3")
            
    # --- 4. Alignment ---
    n_gdf = len(events_selected)
    n_mat = len(y_true)
    
    if n_gdf != n_mat:
        print(f"   [WARNING] Count mismatch! GDF: {n_gdf}, MAT: {n_mat}")
        n_min = min(n_gdf, n_mat)
        events_selected = events_selected[:n_min]
        y_true = y_true[:n_min]
        print(f"   [INFO] Truncated to {n_min} trials.")
        
    # --- 5. Create Epochs ---
    dummy_id = {str(events_selected[0,2]): events_selected[0,2]}
    epochs = mne.Epochs(raw, events_selected, event_id=dummy_id, tmin=tmin, tmax=tmax, 
                        baseline=None, preload=True, verbose=False)
    
    X = epochs.get_data()
    return X, y_true

class InferenceDataset(Dataset):
    def __init__(self, X, y, csp_feats, scalers):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.csp = csp_feats.astype(np.float32)
        self.scalers = scalers
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx].copy()
        for ch in range(x.shape[0]):
            x[ch] = self.scalers[ch].transform(x[ch].reshape(1, -1)).flatten()
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long), torch.from_numpy(self.csp[idx])

# ================= 主运行逻辑 =================

def evaluate_subject_detailed(subject_id):
    # 1. Load artifacts
    csp, scalers, state_dict = load_artifacts()
    
    # 2. Load Data
    try:
        X, y_true = load_eval_data_robust(subject_id)
    except Exception as e:
        print(f"[ERROR] Failed to load data for Subject {subject_id}")
        traceback.print_exc()
        return

    # 3. Preprocess (CSP Transform only)
    print("   -> Applying CSP transform...")
    X_csp = csp.transform(X)
    
    # 4. Dataloader
    ds = InferenceDataset(X, y_true, X_csp, scalers)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. Init Model
    n_ch = X.shape[1]
    n_times = X.shape[2]
    model = EEGNetLight(n_channels=n_ch, n_times=n_times, n_classes=4, csp_dim=6).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    print("\n" + "="*85)
    print(f"INFERENCE REPORT: Subject A0{subject_id}E")
    print("="*85)
    print(f"{'ID':<5} | {'True Label':<12} | {'Pred Label':<12} | {'Conf%':<8} | {'Result':<6} | {'Probs [L, R, F, T]'}")
    print("-" * 85)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        batch_idx_base = 0
        for xb, yb, cf in loader:
            xb = xb.to(DEVICE); cf = cf.to(DEVICE)
            
            # --- TTA Strategy (Test Time Augmentation) ---
            # 1. Original
            out1 = F.softmax(model(xb, cf), dim=1)
            # 2. Time Shift (+2 samples)
            out2 = F.softmax(model(torch.roll(xb, 2, dims=-1), cf), dim=1)
            # 3. Noise Injection
            out3 = F.softmax(model(xb + torch.randn_like(xb)*0.001, cf), dim=1)
            
            # Mean Probability
            avg_prob = (out1 + out2 + out3) / 3.0
            
            # Decisions
            conf, preds = torch.max(avg_prob, dim=1)
            preds = preds.cpu().numpy()
            truth = yb.numpy()
            conf = conf.cpu().numpy()
            probs = avg_prob.cpu().numpy()
            
            for i in range(len(preds)):
                idx = batch_idx_base + i
                p_lbl = LABEL_MAP[preds[i]]
                t_lbl = LABEL_MAP[truth[i]]
                is_corr = "PASS" if preds[i] == truth[i] else "FAIL"
                if preds[i] == truth[i]: correct += 1
                total += 1
                
                # Format probs for display
                p_str = ",".join([f"{p:.2f}" for p in probs[i]])
                
                print(f"{idx:<5} | {t_lbl:<12} | {p_lbl:<12} | {conf[i]*100:.1f}%   | {is_corr:<6} | [{p_str}]")
                
            batch_idx_base += len(preds)
            
    final_acc = correct / total if total > 0 else 0
    print("-" * 85)
    print(f"FINAL ACCURACY (A0{subject_id}E): {final_acc:.4f} ({correct}/{total})")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    # 你可以在这里指定要跑哪个受试者 (1-9)
    # 你的日志显示你在测试 Subject 1
    target_subject = 2
    
    evaluate_subject_detailed(target_subject)
