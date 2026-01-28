# deploy_car_brain.py
# 杩愯鏂瑰紡锛?#   python deploy_car_brain.py
# 鏁版嵁鍗忚锛堝缓璁級锛氭瘡涓?JSON 涓€琛岋紝鏍煎紡锛?#   {"data": [[ch1...], [ch2...], ...]}
# 鍏朵腑 data 褰㈢姸涓?(n_channels, n_samples_per_packet)

import json
import os
import socket
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import serial
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

# ================= 鍩烘湰閰嶇疆 =================
HOST = "0.0.0.0"
PORT = 65432

FS = 250.0
WINDOW_SIZE = 1001
PLOT_WINDOW = 750
PLOT_CHANNELS = [7, 9, 11]  # C3, Cz, C4
PLOT_LABELS = ["C3 (uV)", "Cz (uV)", "C4 (uV)"]
PLOT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

PREDICT_STRIDE = 50  # 姣忛殧澶氬皯閲囨牱鐐瑰仛涓€娆℃帹鐞?WATCHDOG_SEC = 1.0

SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 妯″瀷鏂囦欢璺緞锛堢浉瀵硅剼鏈洰褰曪級
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "onlinev50pro")

# ================= 妯″瀷瀹氫箟 =================

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
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, activation="gelu"
        )
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
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.layers(x)


class EEGNetLight(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=4, csp_dim=6):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
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


# ================= 棰勫鐞嗕笌鎺ㄧ悊 =================

class OnlineFilter:
    def __init__(self, n_channels, fs=250.0):
        nyq = fs / 2
        self.b_bp, self.a_bp = butter(4, [0.5 / nyq, 100.0 / nyq], btype="bandpass")
        self.b_notch, self.a_notch = iirnotch(50.0, 30.0, fs)
        self.n_channels = n_channels
        self.zi_bp_unit = lfilter_zi(self.b_bp, self.a_bp)
        self.zi_notch_unit = lfilter_zi(self.b_notch, self.a_notch)
        self.initialized = False
        self.zi_bp = None
        self.zi_notch = None

    def init_state(self, first_sample):
        self.zi_bp = np.zeros((self.n_channels, len(self.zi_bp_unit)))
        self.zi_notch = np.zeros((self.n_channels, len(self.zi_notch_unit)))
        for i in range(self.n_channels):
            self.zi_bp[i] = self.zi_bp_unit * first_sample[i]
            self.zi_notch[i] = self.zi_notch_unit * first_sample[i]
        self.initialized = True

    def process(self, chunk):
        if not self.initialized:
            self.init_state(chunk[:, 0])
        n_ch = chunk.shape[0]
        filtered = np.zeros_like(chunk)
        for i in range(n_ch):
            out_bp, self.zi_bp[i] = lfilter(self.b_bp, self.a_bp, chunk[i], zi=self.zi_bp[i])
            out_notch, self.zi_notch[i] = lfilter(
                self.b_notch, self.a_notch, out_bp, zi=self.zi_notch[i]
            )
            filtered[i] = out_notch
        return filtered


class BCIProcessor:
    def __init__(self, n_channels):
        print("[System] Loading Artifacts & Models...")
        self.csp = joblib.load(os.path.join(ARTIFACTS_DIR, "best_csp.pkl"))
        self.scalers = joblib.load(os.path.join(ARTIFACTS_DIR, "best_scalers.pkl"))
        self.model = EEGNetLight(n_channels=n_channels, n_times=WINDOW_SIZE, n_classes=4, csp_dim=6).to(
            DEVICE
        )
        self.model.load_state_dict(
            torch.load(os.path.join(ARTIFACTS_DIR, "best_model.pth"), map_location=DEVICE)
        )
        self.model.eval()

        self.filter = OnlineFilter(n_channels)
        self.buffer = np.zeros((n_channels, WINDOW_SIZE), dtype=np.float32)
        self.vis_buffer = np.zeros((n_channels, PLOT_WINDOW), dtype=np.float32)
        self.n_channels = n_channels
        self.total_samples = 0
        self.last_pred_sample = 0
        self.last_result = None
        self.lock = threading.Lock()

    def process_chunk(self, raw_chunk):
        filtered = self.filter.process(raw_chunk)
        n_new = filtered.shape[1]

        with self.lock:
            self.buffer = np.roll(self.buffer, -n_new, axis=1)
            self.buffer[:, -n_new:] = filtered

            self.vis_buffer = np.roll(self.vis_buffer, -n_new, axis=1)
            self.vis_buffer[:, -n_new:] = filtered

            self.total_samples += n_new

        if self.total_samples >= WINDOW_SIZE and (self.total_samples - self.last_pred_sample) >= PREDICT_STRIDE:
            self.last_pred_sample = self.total_samples
            return self.run_inference()
        return None

    def run_inference(self):
        try:
            with self.lock:
                temp_input = self.buffer.copy()
                trial_data = self.buffer[np.newaxis, :, :]

            for ch in range(self.n_channels):
                s_idx = ch if ch < len(self.scalers) else -1
                temp_input[ch] = self.scalers[s_idx].transform(temp_input[ch].reshape(1, -1)).flatten()

            t_in = torch.from_numpy(temp_input[np.newaxis, :, :]).float().to(DEVICE)
            c_feat = self.csp.transform(trial_data)
            c_in = torch.from_numpy(c_feat).float().to(DEVICE)

            with torch.no_grad():
                logits = self.model(t_in, c_in)
                probs = F.softmax(logits, dim=1)
            p_idx = int(probs.argmax().item())
            conf = float(probs.max().item())
            self.last_result = (p_idx, conf)
            return p_idx, conf
        except Exception as e:
            print(f"[Inference Error] {e}")
            return None

    def get_vis_snapshot(self):
        with self.lock:
            return self.vis_buffer.copy(), self.total_samples, self.last_result


# ================= 鐢垫満鎺у埗 =================

class MotorController:
    def __init__(self, port, baud):
        self.ser = None
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            print(f"[Motor] Serial open: {port} @ {baud}")
        except Exception as e:
            print(f"[Motor] Serial open failed: {e}")
        self.lock = threading.Lock()

    def send_cmd(self, cmd):
        if not self.ser:
            return
        try:
            with self.lock:
                self.ser.write(cmd.encode("utf-8"))
        except Exception as e:
            print(f"[Motor] Send failed: {e}")

    def stop(self):
        self.send_cmd("$spd:0,0,0,0#")

    def send_by_class(self, class_id):
        if class_id == 0:
            self.send_cmd("$spd:0,0,0,0#")
        elif class_id == 1:
            self.send_cmd("$spd:-200,-200,200,200#")
        elif class_id == 2:
            self.send_cmd("$spd:200,200,-200,-200#")
        elif class_id == 3:
            self.send_cmd("$spd:200,200,200,200#")


# ================= TCP 鎺ユ敹绾跨▼ =================

def parse_json_lines(buffer):
    lines = buffer.split(b"\n")
    return lines[:-1], lines[-1]


def to_chunk(data, default_channels=22):
    arr = np.array(data, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1 and arr.size % default_channels == 0:
        return arr.reshape(default_channels, -1)
    raise ValueError("Invalid data shape. Expect 2D [channels][samples] or flat array divisible by channels.")


def tcp_server_loop(shared):
    motor = shared["motor"]
    processor = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, PORT))
    sock.listen(1)
    print(f"[TCP] Listening on {HOST}:{PORT}")

    conn, addr = sock.accept()
    print(f"[TCP] Client connected: {addr}")
    conn.settimeout(0.2)

    buffer = b""
    last_recv = time.monotonic()
    shared["signal_lost"] = False

    while not shared["stop"]:
        now = time.monotonic()
        if (now - last_recv) > WATCHDOG_SEC and not shared["signal_lost"]:
            motor.stop()
            shared["signal_lost"] = True

        try:
            chunk = conn.recv(4096)
            if not chunk:
                time.sleep(0.01)
                continue
            buffer += chunk
            lines, buffer = parse_json_lines(buffer)
            for line in lines:
                if not line.strip():
                    continue
                payload = json.loads(line.decode("utf-8"))
                if "data" not in payload:
                    continue
                raw_chunk = to_chunk(payload["data"])

                if processor is None:
                    processor = BCIProcessor(raw_chunk.shape[0])
                    shared["processor"] = processor

                last_recv = time.monotonic()
                shared["signal_lost"] = False
                result = processor.process_chunk(raw_chunk)
                if result:
                    class_id, conf = result
                    motor.send_by_class(class_id)
                    shared["last_result"] = (class_id, conf)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[TCP] Error: {e}")
            time.sleep(0.05)

    try:
        conn.close()
    except Exception:
        pass
    try:
        sock.close()
    except Exception:
        pass


# ================= GUI 涓荤嚎绋?=================

def run_gui(shared):
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 2, 2, 1])
    axes_wave = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
    ax_status = fig.add_subplot(gs[3])

    lines = []
    for i, ax in enumerate(axes_wave):
        line, = ax.plot(np.zeros(PLOT_WINDOW), color=PLOT_COLORS[i], lw=1.5)
        ax.set_ylabel(PLOT_LABELS[i])
        ax.set_ylim(-50, 50)
        ax.grid(True, alpha=0.3)
        lines.append(line)

    axes_wave[0].set_title("Real-time EEG Stream (Filtered)")

    ax_status.set_xlim(0, 1)
    ax_status.set_ylim(0, 1)
    ax_status.set_xticks([])
    ax_status.set_yticks([])
    status_text = ax_status.text(
        0.02, 0.6, "Status: Waiting for data...", transform=ax_status.transAxes, fontsize=10, fontweight="bold"
    )
    result_text = ax_status.text(0.02, 0.2, "Result: -", transform=ax_status.transAxes, fontsize=10)

    plt.tight_layout()

    while not shared["stop"]:
        processor = shared.get("processor")
        if processor:
            vis, total_samples, last_result = processor.get_vis_snapshot()
            vis_uV = vis * 1e6
            for i, ch_idx in enumerate(PLOT_CHANNELS):
                if ch_idx < vis_uV.shape[0]:
                    lines[i].set_ydata(vis_uV[ch_idx, :])
            if shared["signal_lost"]:
                status_text.set_text("Status: Signal Lost")
            else:
                sec = total_samples / FS
                status_text.set_text(f"Status: Streaming  |  Time: {sec:0.1f}s")
            if last_result:
                class_id, conf = last_result
                result_text.set_text(f"Result: Class {class_id}  Conf {conf:.2f}")
        else:
            status_text.set_text("Status: Waiting for data...")
            result_text.set_text("Result: -")

        plt.draw()
        plt.pause(0.02)


def main():
    shared = {
        "processor": None,
        "last_result": None,
        "signal_lost": False,
        "stop": False,
        "motor": MotorController(SERIAL_PORT, SERIAL_BAUD),
    }

    t = threading.Thread(target=tcp_server_loop, args=(shared,), daemon=True)
    t.start()

    try:
        run_gui(shared)
    except KeyboardInterrupt:
        pass
    finally:
        shared["stop"] = True
        shared["motor"].stop()
        time.sleep(0.2)


if __name__ == "__main__":
    main()

