# generate_cue_timeline.py
# 从 a01e.gdf 中提取 783 事件的精确采样点位置，生成时间表
import os
import mne
import numpy as np

# 修改为你的路径
DATA_DIR = r"D:\Backup\Downloads\bciciv_2a_gdf"
SUBJECT_ID = 5

def generate():
    gdf_path = os.path.join(DATA_DIR, f"A0{SUBJECT_ID}E.gdf")
    print(f"Reading {gdf_path}...")
    
    # 读取数据
    raw = mne.io.read_raw_gdf(gdf_path, preload=False, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    # 寻找 783 (Unknown Cue) 的 Event ID
    target_id = None
    for k, v in event_id.items():
        if '783' in k:
            target_id = v
            break
            
    if target_id is None:
        print("Error: No 783 event found in GDF.")
        return

    # 提取所有 783 事件的采样点索引
    # events 结构: [sample_index, 0, event_id]
    cues = events[events[:, 2] == target_id][:, 0]
    
    # 保存到 txt
    output_file = "cue_timeline5.txt"
    np.savetxt(output_file, cues, fmt='%d')
    
    print(f"Successfully saved {len(cues)} cue points to {output_file}")
    print(f"First 5 points: {cues[:5]}")

if __name__ == "__main__":
    generate()
