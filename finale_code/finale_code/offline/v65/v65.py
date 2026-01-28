# Final v50_optimized_T0_4_offline.py
# 基于 Final v50 (1).py 进行正则化增强修改
# **关键修改：将时间窗口改为 0.0s 到 4.0s (离线/非因果模式)**

import os, sys, json, shutil, hashlib, random, traceback
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from mne.decoding import CSP
import mne, joblib
from sklearn.metrics import confusion_matrix, classification_report

# ---------------- Config ----------------
# ****** 路径修正: 确保使用 Linux 绝对路径 ******
DATA_DIR = r"/root/autodl-tmp/bciciv_2a_gdf" 
# 更新 Artifacts 目录名以区分实验
ARTIFACTS_DIR = "artifacts_v50_optimized_T0_4_offline"
SEED = 42
# ****** 算力优化: 批量大小 128 ******
BATCH_SIZE = 128
EPOCHS = 120

# --- V41 蒸馏配置 ---
USE_DISTILLATION = True
DISTILL_ALPHA = 0.5       
DISTILL_TEMP = 3.0        
# -------------------

# NUM_WORKERS 和 PIN_MEMORY 将在 DataLoader 中被设置为 12 和 True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- V36: 全局停止配置 ---
GLOBAL_STOP_ACC = 0.866 
GLOBAL_STOP_FLAG = False 

# --- V35: Fold 调停配置 ---
HARD_STOP_EPOCH = 40
HARD_STOP_ACC = 0.70 
MAX_PATIENCE = 50
# --- V34: 方差损失 & 权重 ---
LAMBDA_VAR = 1e-5 
CLASS_WEIGHTS = torch.tensor([0.98, 1.01, 1.01, 1.0], dtype=torch.float32) 

GLOBAL_BEST_ACC = -1.0
GLOBAL_BEST_ARTIFACTS = {} 

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
mne.set_log_level("WARNING")

# ---------------- Utility: robust event mapping ----------------
def find_event_map(event_id_dict, events_array):
    target_codes = {'769':'left','770':'right','771':'foot','772':'tongue'}
    mapping = {}
    for k, v in event_id_dict.items():
        ks = str(k).strip()
        if ks in target_codes:
            mapping[target_codes[ks]] = int(v)
    if len(mapping) < 4:
        for k, v in event_id_dict.items():
            ks = str(k)
            for tcode, tlabel in target_codes.items():
                if tcode in ks and tlabel not in mapping:
                    mapping[tlabel] = int(v)
    present_codes = set(events_array[:,2]) if events_array is not None and len(events_array) else set()
    for num_str, tlabel in target_codes.items():
        num = int(num_str)
        if num in present_codes and tlabel not in mapping:
            mapping[tlabel] = num
    return mapping

# ---------------- Load one subject robustly (***OFFLINE, T=0.0s to 4.0s***) ----------------
# [FIX] 1. 更新了 tmin 和 tmax 的默认参数值
def load_bci_iv2a(path, tmin=0.0, tmax=4.0, pick_eeg=True): 
    print(f"Reading {path} ...")
    raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)
    try:
        raw.pick_types(eeg=True, eog=False)
    except Exception:
        picks = mne.pick_types(raw.info, eeg=True, eog=False, meg=False)
        raw.pick(picks)
    
    # *** 注意: MNE 的 raw.filter/notch_filter 默认使用 filtfilt (零相移/非因果) ***
    raw.filter(0.5, 100., fir_design='firwin', verbose=False)
    try: raw.notch_filter(50, verbose=False)
    except: pass

    events, event_id = mne.events_from_annotations(raw)
    if events is None or len(events) == 0: raise ValueError("No annotations")
    
    events = events[events[:, 2] != 1023] # 排除伪迹
    
    present_codes = set(events[:, 2]) if len(events) > 0 else set()
    filtered_event_id = {k: v for k, v in event_id.items() if int(v) in present_codes}
    mapping = find_event_map(filtered_event_id if filtered_event_id else event_id, events)
    
    wanted = {label: code for label, code in mapping.items()}
    # [FIX] 2. 更新了 epochs 的切片时间窗口
    epochs = mne.Epochs(raw, events, event_id=wanted, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
    
    X = epochs.get_data()
    label_order = ['left','right','foot','tongue']
    code2idx = {}
    for idx, lbl in enumerate(label_order):
        if lbl in wanted: code2idx[wanted[lbl]] = idx
    y = np.array([code2idx.get(code, -1) for code in epochs.events[:, -1]], dtype=np.int64)
    valid_idx = np.where(y >= 0)[0]
    if len(valid_idx) != len(y): X = X[valid_idx]; y = y[valid_idx]
    
    if len(np.unique(y)) < 4:
        raise ValueError(f"Subject data only contains {len(np.unique(y))} classes. Skipping.")
        
    return X, y

def load_all_subjects(data_dir):
    X_list, y_list = [], []
    print("\n[INFO] Starting data loading with OFFLINE filtering. Window T=0.0s to 4.0s (1001 points).")
    for sid in range(1, 10):
        fname = f"A0{sid}T.gdf"
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path): continue
        try:
            X, y = load_bci_iv2a(path)
            if X.size == 0: continue
            print(f"    loaded: {fname} -> trials={X.shape[0]}")
            X_list.append(X); y_list.append(y)
        except Exception as e:
            # ****** 关键修正：打印完整的错误堆栈 ******
            print(f"[FATAL ERROR] failed load {path}. Printing Traceback:")
            traceback.print_exc()
            
    # ****** 确保 X_list 不为空 ******
    if not X_list:
        raise RuntimeError("No data loaded from any subject.")
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    print(f"\n[INFO] Total data loaded: {X.shape}, labels: {y.shape}")
    return X, y

# ---------------- Augmentations ----------------
# [OPTIMIZATION] 修改点 3: 增大噪声强度 (0.005 -> 0.025) 仅在 pipeline 中生效
def eeg_jitter(x, sigma=0.003):
    return x + sigma * np.random.randn(*x.shape)

def eeg_shift(x, max_shift_samples=6):
    shift = np.random.randint(-max_shift_samples, max_shift_samples+1)
    if shift == 0: return x
    return np.roll(x, shift, axis=-1)

# ---------------- TEACHER MODEL: ShallowConvNet ----------------
class ShallowConvNetTeacher(nn.Module):
    def __init__(self, n_channels, n_classes, n_times):
        super().__init__()
        self.conv_time = nn.Conv2d(1, 40, kernel_size=(1, 25), stride=1)
        self.conv_spat = nn.Conv2d(40, 40, kernel_size=(n_channels, 1), stride=1, bias=False)
        self.bn = nn.BatchNorm2d(40, momentum=0.1, affine=True, eps=1e-5)
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(0.5)
        
        # n_times = 1001 (for 4.0s window)
        conv_out_time = n_times - 25 + 1
        out_width = int((conv_out_time - 75) / 15) + 1 # Should be 61 for n_times=1001
        
        self.clf = nn.Linear(40 * out_width, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        
        x = x * x 
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        
        x = self.dropout(x)
        x = x.flatten(1)
        return self.clf(x)

# ---------------- STUDENT MODEL: EEGNetLight (Your Original) ----------------
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
        # [OPTIMIZATION] 修改点 1: 增大 Transformer Dropout (0.1 -> 0.3)
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

# ---------------- Dataset and Loss ----------------
class EEGDataset(Dataset):
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

def worker_init_fn(worker_id):
    seed = SEED + worker_id
    np.random.seed(seed); random.seed(seed)

def distillation_loss(student_logits, teacher_logits, temperature=3.0):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_logprobs = F.log_softmax(student_logits / temperature, dim=1)
    # KLDivLoss expects log_probs as input
    loss_kd = F.kl_div(student_logprobs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return loss_kd

# ---------------- Training Pipeline ----------------
def save_global_artifacts():
    if not GLOBAL_BEST_ARTIFACTS: return
    torch.save(GLOBAL_BEST_ARTIFACTS['model_state'], os.path.join(ARTIFACTS_DIR, "best_model.pth"))
    joblib.dump(GLOBAL_BEST_ARTIFACTS['csp'], os.path.join(ARTIFACTS_DIR, "best_csp.pkl"))
    joblib.dump(GLOBAL_BEST_ARTIFACTS['scalers'], os.path.join(ARTIFACTS_DIR, "best_scalers.pkl"))
    
    config = {
        'n_channels': GLOBAL_BEST_ARTIFACTS['model_state']['temporal.0.weight'].size(1),
        'n_times': GLOBAL_BEST_ARTIFACTS['n_times'], # 存储实际 n_times=1001
        'csp_dim': GLOBAL_BEST_ARTIFACTS['csp'].n_components
    }
    with open(os.path.join(ARTIFACTS_DIR, "model_config.json"), 'w') as f:
        json.dump(config, f)
        
    print(f"\n[Artifacts Saved] Fold {GLOBAL_BEST_ARTIFACTS['fold_idx']+1}, Acc: {GLOBAL_BEST_ACC:.4f}")

def train_fold_pipeline(X_train, y_train, X_val, y_val, fold_idx, X_shape):
    global GLOBAL_BEST_ACC, GLOBAL_BEST_ARTIFACTS, GLOBAL_STOP_FLAG 
    
    if GLOBAL_STOP_FLAG:
        print(f"Fold {fold_idx + 1}: Global Stop Flag active. Skipping.")
        return 0.0, [], [], {} 
        
    print(f"\n--- Fold {fold_idx + 1} (Teacher: ShallowConvNet) ---")
    
    # 1. Balance
    unique_t, counts_t = np.unique(y_train, return_counts=True)
    min_c = counts_t.min()
    idxs_bal = []
    for c in unique_t:
        ids = np.where(y_train==c)[0]
        sel = np.random.choice(ids, min_c, replace=False)
        idxs_bal.extend(sel.tolist())
    idxs_bal = np.array(idxs_bal)
    X_train_bal = X_train[idxs_bal]; y_train_bal = y_train[idxs_bal]
    print(f"Fold {fold_idx+1} Training data balanced. Samples per class: {min_c}")

    # 2. CSP
    csp_dim = 6
    csp = CSP(n_components=csp_dim, reg=None, log=True, norm_trace=False)
    X_train_csp = csp.fit_transform(X_train_bal, y_train_bal)
    X_val_csp = csp.transform(X_val)

    # 3. Scalers
    n_ch = X_shape[1]
    scalers = [StandardScaler() for _ in range(n_ch)]
    for ch in range(n_ch):
        scalers[ch].fit(X_train_bal[:, ch, :])

    # 4. Loaders
    train_ds = EEGDataset(X_train_bal, y_train_bal, X_train_csp, scalers)
    val_ds = EEGDataset(X_val, y_val, X_val_csp, scalers)
    
    # ****** 算力优化: num_workers=12 和 pin_memory=True ******
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True, worker_init_fn=worker_init_fn)

    # 5. Initialize Models
    n_times = X_shape[2]
    # Student
    student = EEGNetLight(n_channels=n_ch, n_times=n_times, n_classes=4, csp_dim=csp_dim).to(DEVICE)
    
    # [OPTIMIZATION] 修改点 2: 增大 Weight Decay (1e-4 -> 1e-2) 以增强正则化
    opt_s = optim.AdamW(student.parameters(), lr=5e-4, weight_decay=1e-2)
    
    # Teacher 
    teacher = None
    opt_t = None
    if USE_DISTILLATION:
        teacher = ShallowConvNetTeacher(n_channels=n_ch, n_classes=4, n_times=n_times).to(DEVICE)
        opt_t = optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=1e-3)

    crit = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE)) 
    scheduler_s = optim.lr_scheduler.ReduceLROnPlateau(opt_s, mode='max', factor=0.5, patience=10)

    best_acc = 0.0; patience = 0
    best_preds = []; best_labels = [] 
    
    for epoch in range(EPOCHS):
        if GLOBAL_STOP_FLAG: break
            
        # --- TRAINING ---
        student.train()
        if teacher: teacher.train()
        
        running_loss = 0.0
        train_correct = 0; train_total = 0
        
        for xb, yb, cf in train_loader:
            xb_np = xb.cpu().numpy()
            if np.random.rand() < 0.6:
                # [OPTIMIZATION] 修改点 3: 增大噪声强度 (0.005 -> 0.025)
                xb_np = eeg_jitter(xb_np, sigma=0.025).astype(np.float32)
                xb_np = eeg_shift(xb_np, max_shift_samples=6).astype(np.float32)
            xb = torch.from_numpy(xb_np).to(DEVICE); yb = yb.to(DEVICE); cf = cf.to(DEVICE)

            # 1. Train Teacher (if enabled)
            teacher_logits = None
            if teacher:
                opt_t.zero_grad()
                teacher_logits = teacher(xb)
                loss_t = crit(teacher_logits, yb)
                loss_t.backward()
                opt_t.step()
                teacher_logits = teacher_logits.detach()

            # 2. Train Student
            opt_s.zero_grad()
            out_s = student(xb, cf)
            
            # Loss A: Hard Label
            loss_ce = crit(out_s, yb)
            probs = torch.softmax(out_s, dim=1) 
            var_loss = torch.var(probs, dim=1, unbiased=False).mean()
            loss_hard = loss_ce + LAMBDA_VAR * var_loss
            
            # Loss B: Distillation
            loss_final = loss_hard
            if teacher_logits is not None:
                loss_kd = distillation_loss(out_s, teacher_logits, temperature=DISTILL_TEMP)
                loss_final = (1.0 - DISTILL_ALPHA) * loss_hard + DISTILL_ALPHA * loss_kd

            loss_final.backward()
            opt_s.step()

            running_loss += loss_final.item()
            pred = out_s.argmax(1)
            train_correct += (pred == yb).sum().item()
            train_total += yb.size(0)

        # --- VALIDATION (Evaluate Student) ---
        student.eval()
        val_correct = 0; val_total = 0; running_val_loss = 0.0
        all_preds = []; all_labels = []
        
        with torch.no_grad():
            for xb, yb, cf in val_loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE); cf = cf.to(DEVICE)
                out = student(xb, cf)
                pred = out.argmax(1)
                all_preds.extend(pred.cpu().numpy()); all_labels.extend(yb.cpu().numpy())
                val_correct += (pred == yb).sum().item(); val_total += yb.size(0)
                running_val_loss += crit(out, yb).item()
        
        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)
        avg_train_loss = running_loss / max(1, len(train_loader))
        avg_val_loss = running_val_loss / max(1, len(val_loader))

        scheduler_s.step(val_acc)
        
        # Save Logic
        save_flag = ""
        if val_acc > best_acc:
            best_acc = val_acc; patience = 0
            best_preds = all_preds; best_labels = all_labels
            temp_best_model_state = student.state_dict()
            
            if val_acc > GLOBAL_BEST_ACC:
                GLOBAL_BEST_ACC = val_acc
                GLOBAL_BEST_ARTIFACTS['csp'] = csp
                GLOBAL_BEST_ARTIFACTS['scalers'] = scalers
                GLOBAL_BEST_ARTIFACTS['model_state'] = temp_best_model_state
                GLOBAL_BEST_ARTIFACTS['fold_idx'] = fold_idx
                GLOBAL_BEST_ARTIFACTS['n_times'] = n_times # 存储 n_times
                save_flag = " [NEW GLOBAL BEST!]"
                
                if val_acc >= GLOBAL_STOP_ACC:
                    GLOBAL_STOP_FLAG = True
                    print(f"\n!!! GLOBAL STOP TRIGGERED: {val_acc:.4f} !!!")
                    save_global_artifacts()
                    break 
        else:
            patience += 1
            if patience >= MAX_PATIENCE and EPOCHS > 1:
                print(f"Fold {fold_idx+1}: Soft early stopping.")
                break
        
        if (epoch + 1) >= HARD_STOP_EPOCH and best_acc < HARD_STOP_ACC:
            print(f"Fold {fold_idx+1}: Guardrail HARD STOP.")
            break

        print(f"Fold {fold_idx+1} Epoch {epoch+1}/{EPOCHS} | Train Loss={avg_train_loss:.4f} Acc={train_acc:.4f} | Val Loss={avg_val_loss:.4f} Acc={val_acc:.4f}{save_flag}")
    
    fold_report = classification_report(best_labels, best_preds, target_names=['left','right','foot','tongue'], output_dict=True, zero_division=0)
    print(f"--- Fold {fold_idx + 1} finished. Best val acc: {best_acc:.4f} ---")
    return best_acc, best_labels, best_preds, fold_report

# ---------------- CV Manager ----------------
def run_cross_validation(X, y, n_splits=5):
    global GLOBAL_BEST_ACC, GLOBAL_STOP_FLAG
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_accuracies = []
    
    for fold_idx, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        acc, _, _, _ = train_fold_pipeline(X_train, y_train, X_val, y_val, fold_idx, X.shape)
        if GLOBAL_STOP_FLAG: break
        if acc > 0.0: fold_accuracies.append(acc)

    if not GLOBAL_STOP_FLAG and GLOBAL_BEST_ARTIFACTS:
        save_global_artifacts()
        
    print("\n" + "="*80)
    print(f"Mean CV Accuracy: {np.mean(fold_accuracies):.4f}" if fold_accuracies else "No folds completed")
    print(f"Best Acc Found: {GLOBAL_BEST_ACC:.4f}")
    return GLOBAL_BEST_ACC

if __name__ == "__main__":
    try:
        X, y = load_all_subjects(DATA_DIR)
    except Exception as e:
        print("Data load failed:", e); traceback.print_exc(); sys.exit(1)
    
    print(f"\n[INFO] Executing CV with Teacher: ShallowConvNet (Online Distillation)...")
    run_cross_validation(X, y, n_splits=10)
    print("\n[END] Training complete. This is the new OFFINE PERFORMANCE CEILING benchmark.")
