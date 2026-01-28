# v77_sender_standby.py
# 模拟脑电采集设备：服务器模式
# 逻辑：启动 -> 加载数据 -> 待命(Standby) -> 检测到接收端 -> 开始推流

import socket
import time
import struct
import numpy as np
import mne
import os

# ================= 配置区域 =================
# 数据路径 (请确认路径正确)
DATA_DIR_GDF = r"D:\Backup\Downloads\bciciv_2a_gdf"
SUBJECT_ID = 5

# 网络配置
HOST = '127.0.0.1' # 本地地址
PORT = 65432       # 通信端口

# 仿真参数 (严格 1:1 时间流逝)
CHUNK_SIZE = 10    # 每次发送 10 个点 (40ms)
FS = 250.0         # 采样率

# ================= 数据加载 =================
def load_data():
    gdf_path = os.path.join(DATA_DIR_GDF, f"a0{SUBJECT_ID}e.gdf")
    print(f"[Sender] Loading GDF file: {gdf_path}...")
    
    # 读取 GDF
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    
    # 确保加载 25 通道 (22 EEG + 3 EOG)
    try: 
        raw.pick_types(eeg=True, eog=True)
    except: 
        # 如果 pick 失败，尝试 pick all non-stim channels
        raw.pick(mne.pick_types(raw.info, eeg=True, eog=True, meg=False))
    
    # 获取数据 (单位: Volts)
    data = raw.get_data() 
    
    # 获取 Events (用于发送 Marker)
    events, _ = mne.events_from_annotations(raw, verbose=False)
    
    print(f"[Sender] Data loaded successfully.")
    print(f"         Shape: {data.shape} (Channels x Samples)")
    print(f"         Duration: {data.shape[1]/FS/60:.1f} minutes")
    
    return data, events

# ================= 主逻辑 =================
def run_server():
    # 1. 先把数据加载进内存，准备好
    data, events = load_data()
    n_channels = data.shape[0]
    n_samples = data.shape[1]
    
    # 构建 Event 索引字典，方便快速查找
    event_map = {ev[0]: ev[2] for ev in events}
    
    # 2. 建立 Socket 服务器
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 允许端口复用 (防止报错 Address already in use)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        s.bind((HOST, PORT))
        s.listen()
        
        print("\n" + "="*60)
        print(f"★ [Sender] SERVER STARTED on {HOST}:{PORT}")
        print(f"★ [Sender] STATUS: STANDBY (Waiting for Receiver...)")
        print("="*60 + "\n")
        
        # *** 关键点：程序会在这里卡住，直到 receiver.py 运行并连接 ***
        conn, addr = s.accept()
        
        with conn:
            print(f"★ [Sender] RECEIVER DETECTED! Connected by {addr}")
            print(f"★ [Sender] Stream starting in 3 seconds...")
            time.sleep(1)
            print(f"           2...")
            time.sleep(1)
            print(f"           1...")
            time.sleep(1)
            print(f"★ [Sender] GO! Streaming real-time data...\n")
            
            # --- 握手阶段 ---
            # 发送通道数给接收端，让它知道怎么初始化
            conn.sendall(struct.pack('!I', n_channels))
            
            # --- 推流循环 ---
            current_ptr = 0
            
            try:
                while current_ptr < n_samples:
                    # 记录循环开始时间 (用于控制发送速度)
                    loop_start = time.time()
                    
                    chunk_end = min(current_ptr + CHUNK_SIZE, n_samples)
                    
                    # A. 发送 Marker (如果有)
                    # 我们检查当前 chunk 的起始点是否有 event
                    # (实际 LSL 会带时间戳，这里简化为“随数据包发送”)
                    if current_ptr in event_map:
                        eid = event_map[current_ptr]
                        # 协议头: 'M' (1 byte) + EventID (4 bytes)
                        msg = struct.pack('!cI', b'M', eid)
                        conn.sendall(msg)
                        print(f"   -> [Event] Sent Marker {eid} at sample {current_ptr}")

                    # B. 发送数据块 (Raw Chunk)
                    # 取出数据
                    chunk = data[:, current_ptr : chunk_end].astype(np.float32)
                    # 转为二进制流
                    chunk_bytes = chunk.tobytes()
                    
                    # 协议头: 'D' (1 byte) + 数据长度 (4 bytes) + 数据体
                    header = struct.pack('!cI', b'D', len(chunk_bytes))
                    conn.sendall(header + chunk_bytes)
                    
                    # 指针后移
                    current_ptr = chunk_end
                    
                    # C. 速度控制 (这一步是模拟"在线"的关键)
                    # 我们发送了 40ms 的数据，那么我们就应该休息 40ms
                    # 减去代码执行消耗的时间，剩下的就是 sleep 时间
                    elapsed = time.time() - loop_start
                    target_interval = CHUNK_SIZE / FS # 10/250 = 0.04s
                    
                    sleep_time = target_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
            except BrokenPipeError:
                print("\n[Sender] Connection lost (Receiver disconnected).")
            except KeyboardInterrupt:
                print("\n[Sender] Stopped by user.")
            except Exception as e:
                print(f"\n[Sender] Error: {e}")
                
    print("[Sender] Session finished.")

if __name__ == "__main__":
    run_server()
