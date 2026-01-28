# BrainCarCode

基于 BCI Competition IV 2a 脑电数据集的运动想象分类项目，包含离线训练/评估与在线实时推理（socket 模拟数据流）。

## 目录结构

```
BrainCarCode/
├─ bciciv_2a_gdf/                 # BCI IV 2a GDF 数据（a01e.gdf 等）
└─ finale_code/
   └─ finale_code/
      ├─ offline/                 # 离线训练/推理
      │  ├─ v42/                  # 版本 v42 训练
      │  └─ v65/                  # 版本 v65 训练与离线分析
      └─ online/                  # 在线推理与模拟
         ├─ onlinev50pro.py       # 在线版本主脚本
         ├─ receiver.py           # 接收端：socket 接收 + 实时推理 + 可视化
         ├─ sender.py             # 发送端：用 GDF 数据模拟采集流
         ├─ acc_analyze/          # 在线准确率分析
         └─ cue_timeline/         # 线索时间轴生成
```

## 环境依赖

主要依赖（Python）：
- numpy, scipy, scikit-learn
- mne
- torch
- matplotlib
- joblib

建议使用 Python 3.8+。

## 离线训练与评估

1) 训练（示例）
```
python finale_code/finale_code/offline/v42/v42.py
python finale_code/finale_code/offline/v65/v65.py
```

2) 离线推理/评估
```
python finale_code/finale_code/offline/v65/v65_offline_analyze.py
```

训练产物会保存在对应 `*_model` 目录（`best_model.pth`, `best_csp.pkl`, `best_scalers.pkl`）。

## 在线推理（模拟实时）

1) 启动发送端（模拟采集设备）
```
python finale_code/finale_code/online/sender.py
```

2) 启动接收端（实时推理 + 可视化）
```
python finale_code/finale_code/online/receiver.py
```

可选：在线仿真分析
```
python finale_code/finale_code/online/onlinev50pro_online_analyze.py
```

## 重要路径说明

脚本中存在绝对路径（如 `DATA_DIR`, `ARTIFACTS_DIR`, `TIMELINE_FILE`），需要根据本机路径修改。
如果你已把数据放在仓库的 `bciciv_2a_gdf/` 下，建议改为相对路径。

示例（Windows）：
```
DATA_DIR_GDF = r"D:\Backup\Downloads\bciciv_2a_gdf"
```

## 数据集说明

`bciciv_2a_gdf/` 中包含 `a01e.gdf` ～ `a09e.gdf`（训练/实验）与 `a01t.gdf` ～ `a09t.gdf`（测试）等文件。

## 备注

- 文件/目录已统一为全小写 + 下划线命名。
- 如果需要将绝对路径全部改为相对路径，可继续告知。

