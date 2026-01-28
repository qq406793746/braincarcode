import re
import os
import sys

# ================= 配置区 =================
FILE_EE3 = r"C:\Users\Administrator\Desktop\finale_code\online\acc_analyze\ee9.txt"  # 包含正确答案的表格文件
FILE_E3 = r"C:\Users\Administrator\Desktop\finale_code\online\acc_analyze\e9.txt"  # 包含实时日志的文件
# =========================================

def parse_ee3_table(filepath):
    """
    解析 ee3.txt (表格格式)
    返回字典: {id: {'true': 'Left', 'pred': 'Left', 'is_correct': True}}
    """
    data = {}
    if not os.path.exists(filepath):
        print(f"[错误] 找不到文件: {filepath}")
        return {}
    
    print(f"[1/3] 正在读取 {filepath} ...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 匹配格式: #1 | Left | Left | 0.93 | ✅
            # 正则解释: #数字 | 单词 | 单词 | 数字 | 符号
            match = re.search(r'#(\d+)\s+\|\s+(\w+)\s+\|\s+(\w+)\s+\|\s+([\d\.]+)\s+\|\s+([✅❌])', line)
            if match:
                trial_id = int(match.group(1))
                data[trial_id] = {
                    'true_label': match.group(2),
                    'pred_label': match.group(3),
                    'is_correct': match.group(5) == '✅'
                }
    print(f"      -> 解析到 {len(data)} 条表格数据")
    return data

def parse_e3_log(filepath):
    """
    解析 e3.txt (日志格式)
    返回字典: {id: {'pred': 'Left', 'conf': 0.94}}
    """
    data = {}
    if not os.path.exists(filepath):
        print(f"[错误] 找不到文件: {filepath}")
        return {}
    
    print(f"[2/3] 正在读取 {filepath} ...")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        # 匹配格式: Cue #1 triggered... RESULT: Left (0.94)
        # 使用 re.DOTALL 让 . 能够匹配换行符
        matches = re.findall(r'Cue #(\d+) triggered.*?RESULT: (\w+) \(([\d\.]+)\)', content, re.DOTALL)
        for m in matches:
            trial_id = int(m[0])
            data[trial_id] = {
                'pred_label': m[1],
                'conf': float(m[2])
            }
    print(f"      -> 解析到 {len(data)} 条日志数据")
    return data

def main():
    ee3_data = parse_ee3_table(FILE_EE3)
    e3_data = parse_e3_log(FILE_E3)

    if not ee3_data or not e3_data:
        print("\n无法完成对比，请检查文件路径是否正确。")
        input("按回车键退出...")
        return

    print("[3/3] 开始比对...\n")
    
    # 获取所有 ID 的并集，确保不漏掉任何一条
    all_ids = sorted(list(set(ee3_data.keys()) | set(e3_data.keys())))
    
    discrepancies = []  # 存储两个模型预测不一致的情况
    ee3_correct = 0
    e3_correct = 0
    total_valid = 0

    print("-" * 85)
    print(f"{'ID':<5} | {'正确答案':<8} | {'Ee3预测':<8} | {'e3预测':<8} | {'Ee3对错':<6} | {'e3对错':<6} | {'说明'}")
    print("-" * 85)

    for tid in all_ids:
        ee3 = ee3_data.get(tid)
        e3 = e3_data.get(tid)

        if not ee3 or not e3:
            # 数据缺失的情况
            continue
        
        total_valid += 1
        
        # 获取各类标签
        true_lbl = ee3['true_label']
        ee3_pred = ee3['pred_label']
        e3_pred = e3['pred_label']
        
        # 判断正确性
        # Ee3 的正确性直接读取文件里的 ✅❌ 即可，也可以重新算
        is_ee3_right = (ee3_pred == true_lbl)
        if is_ee3_right: ee3_correct += 1
        
        # e3 的正确性需要拿它的预测去和 Ee3 里的正确答案比
        is_e3_right = (e3_pred == true_lbl)
        if is_e3_right: e3_correct += 1
        
        # 判断两者预测是否一致
        is_consistent = (ee3_pred == e3_pred)
        
        note = ""
        if not is_consistent:
            note = "<<<< 预测打架 (不一致)"
            discrepancies.append({
                'id': tid,
                'true': true_lbl,
                'ee3': ee3_pred,
                'e3': e3_pred
            })

        # 图标
        icon_ee3 = "✅" if is_ee3_right else "❌"
        icon_e3 = "✅" if is_e3_right else "❌"
        
        # 只打印有问题的行（或者你可以注释掉这个 if 来打印所有行）
        if not is_consistent or not is_ee3_right or not is_e3_right:
            # 高亮不一致的行
            prefix = "★ " if not is_consistent else "  "
            print(f"{prefix}{tid:<3} | {true_lbl:<8} | {ee3_pred:<8} | {e3_pred:<8} | {icon_ee3:<6} | {icon_e3:<6} | {note}")

    print("-" * 85)
    print("\n=== 最终统计结果 ===")
    print(f"总样本数: {total_valid}")
    print(f"Ee3 准确率: {ee3_correct}/{total_valid} ({ee3_correct/total_valid*100:.2f}%)")
    print(f"e3  准确率: {e3_correct}/{total_valid} ({e3_correct/total_valid*100:.2f}%)")
    
    print(f"\n=== 差异汇总 (共 {len(discrepancies)} 处) ===")
    if discrepancies:
        print("以下题目中，Ee3 和 e3 给出了不同的预测：")
        for item in discrepancies:
            winner = "e3" if item['e3'] == item['true'] else ("Ee3" if item['ee3'] == item['true'] else "都不是")
            print(f"  - Trial #{item['id']}: 正确是[{item['true']}] -> Ee3猜[{item['ee3']}] vs e3猜[{item['e3']}] -> 胜者: {winner}")
    else:
        print("神奇！Ee3 和 e3 的预测结果完全一致。")

if __name__ == "__main__":
    main()
