# Codex Log

## 2026-01-28
- 统一命名规范：全小写 + 下划线，去空格（例如 `finale code` -> `finale_code`，`acc analyze` -> `acc_analyze`）。
- 数据集目录改名为 `bciciv_2a_gdf`，并将所有 GDF 文件改为小写（例如 `a01e.gdf`）。
- 更新脚本中的路径引用与文件名引用以匹配新命名。
- 生成 `README.md` 项目说明文档。
- 新增 `deploy_car_brain.py`：TCP Server 接收 EEG JSON 数据、实时可视化、推理与串口控制、看门狗断联停车。
- 更新 `.gitignore`，忽略 `bciciv_2a_gdf/` 与 `.vs/`。
- 在 `test-infinite-network` 分支完成备份推送（不含数据集目录）。
- 修复 `codex_log.md` 编码为 UTF-8，避免中文乱码。
- 新增 `deploy_car_brain_mobaxterm.py` 副本：HOST 改为 `0.0.0.0`，并指定 Matplotlib 后端 `TkAgg` 以适配 MobaXterm。
