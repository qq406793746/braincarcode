# v40_metrics_CNN_TRM_CSP_GLOBAL_STOP.py
# 核心改动 (基于 v40):
# 1. 引入详细的 History 记录 (Train/Val 的 Loss/Acc)
# 2. 训练结束保存 training_metrics.json 以便与 v50 对比
# 3. 自动绘制 Fold 1 的训练曲线

import os, sys, json, shutil, hashlib, random, traceback
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from mne.decoding import CSP
import mne, joblib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt # 用于画图

# ---------------- Config and Global Artifacts Tracking ----------------
DATA_DIR = r"D:\Backup\Downloads\bciciv_2a_gdf" # <-- 修改为你的数据路径
ARTIFACTS_DIR = "artifacts_v40_metrics"
SEED = 42
BATCH_SIZE = 32

EPOCHS = 60
# --------------------------------------------------
NUM_WORKERS = 0
PIN_MEMORY = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 全局高精度硬停止配置 ---
GLOBAL_STOP_ACC = 0.866 
GLOBAL_STOP_FLAG = False 

# --- Fold 调停/硬停止配置 (Guardrail) ---
HARD_STOP_EPOCH = 40
HARD_STOP_ACC = 0.74 
MAX_PATIENCE = 30 

# --- 方差损失超参数 ---
LAMBDA_VAR = 1e-5 

# --- 类别权重 ---
CLASS_WEIGHTS = torch.tensor([0.98, 1.01, 1.01, 1.0], dtype=torch.float32) 

# --- 全局变量用于追踪和保存最佳模型 ---
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

# ---------------- Load one subject robustly ----------------
def load_bci_iv2a(path, tmin=0.5, tmax=3.5, pick_eeg=True):
    print(f"Reading {path} ...")
    raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)
    try:
        raw.pick_types(eeg=True, eog=False)
    except Exception:
        picks = mne.pick_types(raw.info, eeg=True, eog=False, meg=False)
        raw.pick(picks)

    raw.filter(0.5, 100., fir_design='firwin', verbose=False)
    try:
        raw.notch_filter(50, verbose=False)
    except Exception:
        pass

    events, event_id = mne.events_from_annotations(raw)
    if events is None or len(events) == 0:
        raise ValueError("No annotations/events found")

    # 排除伪迹事件 1023
    events = events[events[:, 2] != 1023]
    
    present_codes = set(events[:, 2]) if len(events) > 0 else set()
    filtered_event_id = {k: v for k, v in event_id.items() if int(v) in present_codes}

    mapping = find_event_map(filtered_event_id if filtered_event_id else event_id, events)
    if not mapping:
        raise ValueError("No valid events found in file (mapping empty)")

    wanted = {label: code for label, code in mapping.items()}
    epochs = mne.Epochs(raw, events, event_id=wanted, tmin=0.5, tmax=3.5, baseline=None, preload=True, verbose=False)
    if len(epochs) == 0:
        raise ValueError("Epoching returned zero trials for this file")

    X = epochs.get_data()
    label_order = ['left','right','foot','tongue']
    code2idx = {}
    for idx, lbl in enumerate(label_order):
        if lbl in wanted:
            code2idx[wanted[lbl]] = idx
    y = np.array([code2idx.get(code, -1) for code in epochs.events[:, -1]], dtype=np.int64)
    valid_idx = np.where(y >= 0)[0]
    if len(valid_idx) != len(y):
        X = X[valid_idx]; y = y[valid_idx]
    return X, y

# ---------------- Load all (skip invalid files) ----------------
def load_all_subjects(data_dir):
    X_list, y_list = [], []
    for sid in range(1, 10):
        fname = f"A0{sid}T.gdf"
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"[WARN] not found: {path}")
            continue
        try:
            X, y = load_bci_iv2a(path)
            if X.size == 0 or len(y) == 0:
                continue
            print(f"    loaded: {fname} -> trials={X.shape[0]}")
            X_list.append(X); y_list.append(y)
        except Exception as e:
            print(f"[ERROR] failed load {path}: {e}")
            continue
    if len(X_list) == 0:
        raise FileNotFoundError(f"No valid files under {data_dir}")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    print(f"Total loaded: trials={X.shape[0]}, ch={X.shape[1]}, time={X.shape[2]}")
    return X, y

# ---------------- small augmentations ----------------
def eeg_jitter(x, sigma=0.003):
    return x + sigma * np.random.randn(*x.shape)

def eeg_shift(x, max_shift_samples=6):
    shift = np.random.randint(-max_shift_samples, max_shift_samples+1)
    if shift == 0: return x
    return np.roll(x, shift, axis=-1)

# ---------------- Model ----------------
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
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.5), 
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
        
        input_dim = 128 + 64 # Transformer features + CSP features
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

# ---------------- Dataset ----------------
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

# ---------------- DataLoader worker init ----------------
def worker_init_fn(worker_id):
    seed = SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)

# ---------------- Training pipeline for one fold ----------------
def train_fold_pipeline(X_train, y_train, X_val, y_val, fold_idx, X_shape):
    global GLOBAL_BEST_ACC, GLOBAL_BEST_ARTIFACTS, GLOBAL_STOP_FLAG 
    
    # 如果全局停止标志已激活，直接返回空结构
    if GLOBAL_STOP_FLAG:
        print(f"Fold {fold_idx + 1}: Global Stop Flag active. Skipping.")
        return 0.0, [], [], {}, {}
        
    print(f"\n--- Starting Fold {fold_idx + 1} (Target Global Acc: {GLOBAL_STOP_ACC*100:.1f}%) ---")
    
    # 1. Balance
    unique_t, counts_t = np.unique(y_train, return_counts=True)
    min_c = counts_t.min()
    idxs_bal = []
    for c in unique_t:
        ids = np.where(y_train==c)[0]
        sel = np.random.choice(ids, min_c, replace=False)
        idxs_bal.extend(sel.tolist())
    idxs_bal = np.array(idxs_bal)
    X_train_bal = X_train[idxs_bal]
    y_train_bal = y_train[idxs_bal]
    print(f"Fold {fold_idx+1} train trials (balanced): {len(y_train_bal)}")

    # 2. fit CSP
    csp_dim = 6
    csp = CSP(n_components=csp_dim, reg=None, log=True, norm_trace=False)
    X_train_csp = csp.fit_transform(X_train_bal, y_train_bal)
    X_val_csp = csp.transform(X_val)

    # 3. Scalers
    n_ch = X_shape[1]
    scalers = [StandardScaler() for _ in range(n_ch)]
    for ch in range(n_ch):
        scalers[ch].fit(X_train_bal[:, ch, :])

    # 4. Datasets
    train_ds = EEGDataset(X_train_bal, y_train_bal, X_train_csp, scalers)
    val_ds = EEGDataset(X_val, y_val, X_val_csp, scalers)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, worker_init_fn=worker_init_fn)

    # 5. Model, Optimizer, Loss
    model = EEGNetLight(n_channels=n_ch, n_times=X_shape[2], n_classes=4, csp_dim=csp_dim).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE)) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10)

    best_acc = 0.0; patience = 0
    best_preds = []; best_labels = [] 
    temp_best_model_state = None 
    
    # === [METRICS] 初始化历史记录字典 ===
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        if GLOBAL_STOP_FLAG:
            print(f"Fold {fold_idx + 1}: Global Stop Flag set during Epoch {epoch+1}. Exiting training loop.")
            break
            
        # --- TRAINING ---
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for xb, yb, cf in train_loader:
            # Augmentation
            xb_np = xb.cpu().numpy()
            if np.random.rand() < 0.6:
                xb_np = eeg_jitter(xb_np, sigma=0.005).astype(np.float32)
                xb_np = eeg_shift(xb_np, max_shift_samples=6).astype(np.float32)
            
            xb = torch.from_numpy(xb_np).to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            cf = cf.to(DEVICE, non_blocking=True)

            out = model(xb, cf)
            
            loss_ce = crit(out, yb)
            probs = torch.softmax(out, dim=1) 
            var_loss = torch.var(probs, dim=1, unbiased=False).mean()
            loss = loss_ce + LAMBDA_VAR * var_loss

            opt.zero_grad(); loss.backward(); opt.step()
            running_loss += loss.item()
            
            pred = out.argmax(1)
            train_correct += (pred == yb).sum().item()
            train_total += yb.size(0)

        # --- VALIDATION ---
        model.eval()
        val_correct = 0
        val_total = 0
        running_val_loss = 0.0
        all_preds_ep = []; all_labels_ep = []
        
        with torch.no_grad():
            for xb, yb, cf in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                cf = cf.to(DEVICE, non_blocking=True)
                out = model(xb, cf)
                pred = out.argmax(1)
                
                all_preds_ep.extend(pred.cpu().numpy())
                all_labels_ep.extend(yb.cpu().numpy())
                val_correct += (pred == yb).sum().item()
                val_total += yb.size(0)
                
                loss_ce = crit(out, yb)
                probs = torch.softmax(out, dim=1) 
                var_loss = torch.var(probs, dim=1, unbiased=False).mean()
                loss = loss_ce + LAMBDA_VAR * var_loss
                running_val_loss += loss.item()
        
        # --- Metrics Calculation ---
        avg_train_loss = running_loss / (len(train_loader) if len(train_loader) > 0 else 1)
        train_acc = train_correct / (train_total if train_total > 0 else 1)
        
        val_acc = 0.0
        if val_total > 0:
            val_acc = val_correct / val_total
        avg_val_loss = running_val_loss / (len(val_loader) if len(val_loader) > 0 else 1)

        scheduler.step(val_acc) 
        
        # === [METRICS] Append to history ===
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        save_flag = ""
        # 1. 检查是否达到 Fold 最佳
        if val_acc > best_acc:
            best_acc = val_acc; patience = 0
            best_preds = all_preds_ep
            best_labels = all_labels_ep
            temp_best_model_state = model.state_dict()
            
            # 2. 检查是否达到 GLOBAL 最佳
            if val_acc > GLOBAL_BEST_ACC:
                GLOBAL_BEST_ACC = val_acc
                GLOBAL_BEST_ARTIFACTS['csp'] = csp
                GLOBAL_BEST_ARTIFACTS['scalers'] = scalers
                GLOBAL_BEST_ARTIFACTS['model_state'] = temp_best_model_state
                GLOBAL_BEST_ARTIFACTS['fold_idx'] = fold_idx
                save_flag = " [NEW GLOBAL BEST!]"
                
                # 3. 检查是否达到 GLOBAL STOP 阈值
                if val_acc >= GLOBAL_STOP_ACC:
                    GLOBAL_STOP_FLAG = True
                    print("\n" + "#"*80)
                    print(f"!!! GLOBAL STOP TRIGGERED !!! ACCURACY {val_acc:.4f} >= {GLOBAL_STOP_ACC:.4f}")
                    print(f"Saving artifacts for Fold {fold_idx + 1} and immediately stopping CV.")
                    print("#"*80)
                    save_global_artifacts()
                    break 
            
        else:
            patience += 1
            if patience >= MAX_PATIENCE and EPOCHS > 1:
                print(f"Fold {fold_idx+1}: Soft early stopping (patience={MAX_PATIENCE}) at Epoch {epoch+1}.")
                break
        
        if (epoch + 1) >= HARD_STOP_EPOCH and best_acc < HARD_STOP_ACC:
            print(f"Fold {fold_idx+1}: HARD STOP Guardrail triggered at Epoch {epoch+1}! Best Acc {best_acc:.4f} < {HARD_STOP_ACC}.")
            break

        print(f"Fold {fold_idx+1} Epoch {epoch+1}/{EPOCHS} | Train Loss={avg_train_loss:.4f} Acc={train_acc:.4f} | Val Loss={avg_val_loss:.4f} Acc={val_acc:.4f}{save_flag}")
    
    target_names = ['left', 'right', 'foot', 'tongue'] 
    fold_report = classification_report(best_labels, best_preds, 
                                            target_names=target_names, output_dict=True, zero_division=0)
    
    print(f"--- Fold {fold_idx + 1} finished. Best val acc: {best_acc:.4f} ---")
    
    # 返回包含 history 的五元组
    return best_acc, best_labels, best_preds, fold_report, history

# ---------------- Artifact Saving Function ----------------
def save_global_artifacts():
    if not GLOBAL_BEST_ARTIFACTS:
        return
        
    best_fold = GLOBAL_BEST_ARTIFACTS['fold_idx'] + 1
    best_acc_val = GLOBAL_BEST_ACC
    
    model_path = os.path.join(ARTIFACTS_DIR, "best_model.pth")
    torch.save(GLOBAL_BEST_ARTIFACTS['model_state'], model_path)
    
    csp_path = os.path.join(ARTIFACTS_DIR, "best_csp.pkl")
    joblib.dump(GLOBAL_BEST_ARTIFACTS['csp'], csp_path)
    
    scalers_path = os.path.join(ARTIFACTS_DIR, "best_scalers.pkl")
    joblib.dump(GLOBAL_BEST_ARTIFACTS['scalers'], scalers_path)
    
    print(f"\n[Artifacts Saved] Fold {best_fold}, Acc: {best_acc_val:.4f}. Files saved to {ARTIFACTS_DIR}")
    
# ---------------- Cross Validation Manager ----------------
def run_cross_validation(X, y, n_splits=5):
    global GLOBAL_BEST_ACC, GLOBAL_BEST_ARTIFACTS, GLOBAL_STOP_FLAG
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    fold_accuracies = []
    all_true_labels = []
    all_pred_labels = []
    all_fold_reports = []
    
    # === [METRICS] 收集所有折的历史 ===
    all_histories = {}

    for fold_idx, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # 接收返回值（包含 history）
        best_acc, fold_labels, fold_preds, fold_report, fold_history = train_fold_pipeline(X_train, y_train, X_val, y_val, fold_idx, X.shape)
        
        if GLOBAL_STOP_FLAG:
            print(f"\n[CV Manager] Global Stop Flag detected. Terminating Cross-Validation.")
            # 即使停止，也保存当前这折的数据以便分析
            if best_acc > 0.0:
                all_histories[f"fold_{fold_idx+1}"] = fold_history
            break
            
        if best_acc > 0.0:
            fold_accuracies.append(best_acc)
            all_true_labels.extend(fold_labels)
            all_pred_labels.extend(fold_preds)
            all_fold_reports.append(fold_report)
            
            # 保存历史
            all_histories[f"fold_{fold_idx+1}"] = fold_history

    # --- Final Save Logic ---
    if not GLOBAL_STOP_FLAG and GLOBAL_BEST_ARTIFACTS:
        print("\n" + "#"*80)
        print(f"CV Finished (No Global Stop). Saving overall BEST Model (Acc: {GLOBAL_BEST_ACC:.4f}).")
        save_global_artifacts()
        print("#"*80)
    
    # === [METRICS] 保存训练过程数据到 JSON ===
    history_path = os.path.join(ARTIFACTS_DIR, "training_metrics.json")
    try:
        with open(history_path, "w") as f:
            json.dump(all_histories, f, indent=2)
        print(f"\n[INFO] Training metrics saved to {history_path}")
        
        # === [METRICS] 简单绘制第一折的曲线图以供检查 ===
        if "fold_1" in all_histories:
            h = all_histories["fold_1"]
            if h['train_loss']: # 确保非空
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(h['train_loss'], label='Train Loss')
                plt.plot(h['val_loss'], label='Val Loss')
                plt.title('Fold 1 Loss')
                plt.xlabel('Epochs'); plt.ylabel('Loss')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(h['train_acc'], label='Train Acc')
                plt.plot(h['val_acc'], label='Val Acc')
                plt.title('Fold 1 Accuracy')
                plt.xlabel('Epochs'); plt.ylabel('Accuracy')
                plt.legend()
                
                plt_path = os.path.join(ARTIFACTS_DIR, "fold_1_curves.png")
                plt.savefig(plt_path)
                print(f"[INFO] Plot saved to {plt_path}")
                plt.close()
            
    except Exception as e:
        print(f"[WARN] Failed to save history or plot: {e}")

    if not fold_accuracies:
        return GLOBAL_BEST_ACC
        
    print("\n" + "="*80)
    print(f"====== {len(fold_accuracies)}-FOLD CV SUMMARY ======")
    mean_acc = np.mean(fold_accuracies)
    print(f"Mean CV Accuracy: {mean_acc:.4f}")
    print(f"Absolute Best Acc Found: {GLOBAL_BEST_ACC:.4f}")
    
    target_names = ['left', 'right', 'foot', 'tongue']
    if len(all_true_labels) > 0:
        report = classification_report(all_true_labels, all_pred_labels, 
                                         target_names=target_names, output_dict=True, zero_division=0)
        print("\n" + "====== 汇总性能指标 (Combined) ======")
        print(f"{'类别':<10} {'召回率':<10} {'F1-Score':<10}")
        for name in target_names:
            print(f"{name:<10} {report[name]['recall']:.4f} {report[name]['f1-score']:.4f}")
            
    return GLOBAL_BEST_ACC

# ---------------- Main ----------------
if __name__ == "__main__":
    print("Device:", DEVICE)
    try:
        X, y = load_all_subjects(DATA_DIR)
    except Exception as e:
        print("Data load failed:", e)
        sys.exit(1)

    # Sanity check
    print("Loaded data shape:", X.shape, y.shape)
    
    TEST_N_SPLITS = 10 
    print(f"\n[INFO] Executing {TEST_N_SPLITS}-Fold CV (V40 Optimized Metrics)...")
    final_best_acc = run_cross_validation(X, y, n_splits=TEST_N_SPLITS)
    
    meta = {
        "saved_at": datetime.now().isoformat(),
        "pipeline_version": "V40_Metrics_Logging",
        "best_fold_accuracy": f"{final_best_acc:.4f}",
    }
    json.dump(meta, open(os.path.join(ARTIFACTS_DIR, "meta.json"), "w"), indent=2)
    print(f"\nArtifacts and Metrics saved to: {os.path.abspath(ARTIFACTS_DIR)}")
