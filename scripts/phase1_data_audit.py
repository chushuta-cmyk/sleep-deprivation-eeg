#!/usr/bin/env python3
"""
Phase 1: Data Audit
检查 BIDS 结构完整性，创建变量字典，审计 pilot 数据
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# 项目路径
PROJECT_ROOT = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/sleep_deprivation_eeg")
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DERIVATIVES_DIR = PROJECT_ROOT / "data" / "derivatives"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
RESULTS_DIR = PROJECT_ROOT / "results"

def check_bids_structure():
    """检查 BIDS 结构完整性"""
    print("=" * 80)
    print("Phase 1.1: BIDS 结构检查")
    print("=" * 80)
    
    # 检查根目录文件
    required_files = ["dataset_description.json", "participants.tsv", "participants.json"]
    print("\n根目录文件检查:")
    for f in required_files:
        path = RAW_DATA_DIR / f
        exists = path.exists()
        print(f"  {f}: {'✓' if exists else '✗'}")
    
    # 获取所有被试
    participants_tsv = pd.read_csv(RAW_DATA_DIR / "participants.tsv", sep="\t")
    n_participants = len(participants_tsv)
    print(f"\n被试总数: {n_participants}")
    
    # 检查每个被试的目录结构
    print("\n被试目录结构检查:")
    bids_issues = []
    subjects_with_data = []
    
    for idx, row in participants_tsv.iterrows():
        sub_id = row["participant_id"]
        sub_dir = RAW_DATA_DIR / sub_id
        
        if not sub_dir.exists():
            bids_issues.append(f"{sub_id}: 目录不存在")
            continue
        
        # 检查 session 目录
        for ses in ["ses-1", "ses-2"]:
            ses_dir = sub_dir / ses
            if not ses_dir.exists():
                bids_issues.append(f"{sub_id}/{ses}: 目录不存在")
                continue
            
            # 检查 EEG 目录
            eeg_dir = ses_dir / "eeg"
            if not eeg_dir.exists():
                bids_issues.append(f"{sub_id}/{ses}/eeg: 目录不存在")
                continue
            
            # 检查 EEG 文件
            eeg_files = list(eeg_dir.glob("*.set"))
            fdt_files = list(eeg_dir.glob("*.fdt"))
            
            if len(eeg_files) == 0:
                bids_issues.append(f"{sub_id}/{ses}/eeg: 缺少 .set 文件")
            else:
                subjects_with_data.append({
                    "subject": sub_id,
                    "session": ses,
                    "eeg_file": eeg_files[0].name,
                    "has_fdt": len(fdt_files) > 0
                })
    
    # 输出检查结果
    print(f"\n✓ 有 EEG 数据的被试-session 组合: {len(subjects_with_data)}")
    if bids_issues:
        print(f"\n✗ 发现 {len(bids_issues)} 个 BIDS 结构问题:")
        for issue in bids_issues[:10]:  # 只显示前 10 个
            print(f"  - {issue}")
        if len(bids_issues) > 10:
            print(f"  ... 还有 {len(bids_issues) - 10} 个问题")
    
    # 保存检查结果
    audit_report = {
        "timestamp": datetime.now().isoformat(),
        "n_participants": n_participants,
        "n_subjects_with_data": len(subjects_with_data),
        "bids_issues": bids_issues,
        "subjects_with_data": subjects_with_data
    }
    
    with open(METADATA_DIR / "bids_audit_report.json", "w") as f:
        json.dump(audit_report, f, indent=2)
    
    print(f"\n✓ BIDS 审计报告已保存: {METADATA_DIR / 'bids_audit_report.json'}")
    
    return participants_tsv, subjects_with_data


def create_variable_dictionary(participants_tsv):
    """创建变量字典"""
    print("\n" + "=" * 80)
    print("Phase 1.2: 创建变量字典")
    print("=" * 80)
    
    # 读取 participants.json 获取变量描述
    participants_json_path = RAW_DATA_DIR / "participants.json"
    if participants_json_path.exists():
        with open(participants_json_path) as f:
            var_desc = json.load(f)
    else:
        var_desc = {}
    
    # 创建变量字典
    variables = []
    
    # 人口统计学变量
    variables.append({
        "name": "participant_id",
        "type": "identifier",
        "description": "被试唯一标识符",
        "values": participants_tsv["participant_id"].tolist()
    })
    
    variables.append({
        "name": "Gender",
        "type": "categorical",
        "description": "性别",
        "levels": {"M": "男", "F": "女"},
        "n_missing": participants_tsv["Gender"].isna().sum()
    })
    
    variables.append({
        "name": "Age",
        "type": "continuous",
        "description": "年龄（岁）",
        "mean": participants_tsv["Age"].mean(),
        "std": participants_tsv["Age"].std(),
        "range": [participants_tsv["Age"].min(), participants_tsv["Age"].max()],
        "n_missing": participants_tsv["Age"].isna().sum()
    })
    
    variables.append({
        "name": "SessionOrder",
        "type": "categorical",
        "description": "实验顺序（平衡设计）",
        "levels": {"NS->SD": "正常睡眠 -> 睡眠剥夺", "SD->NS": "睡眠剥夺 -> 正常睡眠"},
        "counts": participants_tsv["SessionOrder"].value_counts().to_dict()
    })
    
    # 行为变量
    behavior_vars = [
        ("PVT_item1_NS", "PVT 项目1 - 正常睡眠"),
        ("PVT_item2_NS", "PVT 项目2 - 正常睡眠"),
        ("PVT_item3_NS", "PVT 项目3 - 正常睡眠"),
        ("PVT_item1_SD", "PVT 项目1 - 睡眠剥夺"),
        ("PVT_item2_SD", "PVT 项目2 - 睡眠剥夺"),
        ("PVT_item3_SD", "PVT 项目3 - 睡眠剥夺"),
    ]
    
    for var_name, desc in behavior_vars:
        if var_name in participants_tsv.columns:
            values = pd.to_numeric(participants_tsv[var_name], errors="coerce")
            variables.append({
                "name": var_name,
                "type": "continuous",
                "description": desc,
                "mean": values.mean(),
                "std": values.std(),
                "n_missing": values.isna().sum(),
                "n_valid": values.notna().sum()
            })
    
    # 情绪变量
    emotion_vars = [
        ("PANAS_P_NS", "PANAS 正性情绪 - 正常睡眠"),
        ("PANAS_P_SD", "PANAS 正性情绪 - 睡眠剥夺"),
        ("PANAS_N_NS", "PANAS 负性情绪 - 正常睡眠"),
        ("PANAS_N_SD", "PANAS 负性情绪 - 睡眠剥夺"),
    ]
    
    for var_name, desc in emotion_vars:
        if var_name in participants_tsv.columns:
            values = pd.to_numeric(participants_tsv[var_name], errors="coerce")
            variables.append({
                "name": var_name,
                "type": "continuous",
                "description": desc,
                "mean": values.mean(),
                "std": values.std(),
                "n_missing": values.isna().sum(),
                "n_valid": values.notna().sum()
            })
    
    # 焦虑变量
    anxiety_vars = [
        ("ATQ_NS", "焦虑思维问卷 - 正常睡眠"),
        ("ATQ_SD", "焦虑思维问卷 - 睡眠剥夺"),
        ("SAI_NS", "状态焦虑量表 - 正常睡眠"),
        ("SAI_SD", "状态焦虑量表 - 睡眠剥夺"),
    ]
    
    for var_name, desc in anxiety_vars:
        if var_name in participants_tsv.columns:
            values = pd.to_numeric(participants_tsv[var_name], errors="coerce")
            variables.append({
                "name": var_name,
                "type": "continuous",
                "description": desc,
                "mean": values.mean(),
                "std": values.std(),
                "n_missing": values.isna().sum(),
                "n_valid": values.notna().sum()
            })
    
    # 嗜睡变量
    sleepiness_vars = [
        ("SSS_NS", "Stanford 嗜睡量表 - 正常睡眠"),
        ("SSS_SD", "Stanford 嗜睡量表 - 睡眠剥夺"),
        ("KSS_NS", "Karolinska 嗜睡量表 - 正常睡眠"),
        ("KSS_SD", "Karolinska 嗜睡量表 - 睡眠剥夺"),
    ]
    
    for var_name, desc in sleepiness_vars:
        if var_name in participants_tsv.columns:
            values = pd.to_numeric(participants_tsv[var_name], errors="coerce")
            variables.append({
                "name": var_name,
                "type": "continuous",
                "description": desc,
                "mean": values.mean(),
                "std": values.std(),
                "n_missing": values.isna().sum(),
                "n_valid": values.notna().sum()
            })
    
    # 睡眠质量变量
    sleep_quality_vars = [
        ("PSQI_GlobalScore", "PSQI 总分 - 睡眠质量"),
        ("PSQI_item1", "PSQI 项目1 - 主观睡眠质量"),
        ("PSQI_item2", "PSQI 项目2 - 睡眠潜伏期"),
        ("PSQI_item3", "PSQI 项目3 - 睡眠持续时间"),
        ("PSQI_item4", "PSQI 项目4 - 习惯性睡眠效率"),
        ("PSQI_item5", "PSQI 项目5 - 睡眠障碍"),
        ("PSQI_item6", "PSQI 项目6 - 使用睡眠药物"),
        ("PSQI_item7", "PSQI 项目7 - 日间功能障碍"),
    ]
    
    for var_name, desc in sleep_quality_vars:
        if var_name in participants_tsv.columns:
            values = pd.to_numeric(participants_tsv[var_name], errors="coerce")
            variables.append({
                "name": var_name,
                "type": "continuous",
                "description": desc,
                "mean": values.mean(),
                "std": values.std(),
                "n_missing": values.isna().sum(),
                "n_valid": values.notna().sum()
            })
    
    # 保存变量字典
    variable_dict_df = pd.DataFrame(variables)
    variable_dict_df.to_csv(METADATA_DIR / "variable_dictionary.csv", index=False)
    
    print(f"\n✓ 变量字典已保存: {METADATA_DIR / 'variable_dictionary.csv'}")
    print(f"  总变量数: {len(variables)}")
    
    # 打印关键变量统计
    print("\n关键变量统计:")
    print(f"  被试数: {len(participants_tsv)}")
    print(f"  平均年龄: {participants_tsv['Age'].mean():.1f} ± {participants_tsv['Age'].std():.1f} 岁")
    print(f"  性别分布: {participants_tsv['Gender'].value_counts().to_dict()}")
    print(f"  实验顺序: {participants_tsv['SessionOrder'].value_counts().to_dict()}")
    
    return variables


def audit_pilot_data(subjects_with_data, n_pilot=10):
    """审计 pilot 数据"""
    print("\n" + "=" * 80)
    print("Phase 1.3: Pilot 数据审计")
    print("=" * 80)
    
    # 选择前 n_pilot 个被试作为 pilot
    pilot_subjects = subjects_with_data[:n_pilot]
    
    print(f"\n选择 {len(pilot_subjects)} 个被试-session 组合作为 pilot 数据:")
    for sub in pilot_subjects:
        print(f"  - {sub['subject']} / {sub['session']}")
    
    # 检查 EEG 文件大小
    print("\nEEG 文件大小检查:")
    file_sizes = []
    for sub in pilot_subjects:
        sub_id = sub["subject"]
        ses = sub["session"]
        eeg_file = RAW_DATA_DIR / sub_id / ses / "eeg" / sub["eeg_file"]
        
        if eeg_file.exists():
            size_mb = eeg_file.stat().st_size / (1024 * 1024)
            file_sizes.append({
                "subject": sub_id,
                "session": ses,
                "file": sub["eeg_file"],
                "size_mb": size_mb
            })
            print(f"  {sub_id}/{ses}: {size_mb:.1f} MB")
        else:
            print(f"  {sub_id}/{ses}: 文件不存在")
    
    # 保存 pilot 数据清单
    pilot_df = pd.DataFrame(pilot_subjects)
    pilot_df.to_csv(METADATA_DIR / "pilot_subjects.csv", index=False)
    
    file_sizes_df = pd.DataFrame(file_sizes)
    file_sizes_df.to_csv(METADATA_DIR / "pilot_file_sizes.csv", index=False)
    
    print(f"\n✓ Pilot 数据清单已保存: {METADATA_DIR / 'pilot_subjects.csv'}")
    print(f"✓ 文件大小统计已保存: {METADATA_DIR / 'pilot_file_sizes.csv'}")
    
    return pilot_subjects


def generate_data_flow_diagram():
    """生成数据流程图"""
    print("\n" + "=" * 80)
    print("Phase 1.4: 生成数据流程图")
    print("=" * 80)
    
    flow_diagram = """
数据流程图 (Data Flow Diagram)
================================

原始数据 (Raw Data)
├── participants.tsv          # 被试人口统计学和问卷数据
├── sub-XX/
│   ├── ses-1/
│   │   ├── eeg/
│   │   │   ├── *_eeg.set     # EEGLAB 文件（正常睡眠/睡眠剥夺）
│   │   │   └── *_eeg.fdt     # 数据文件
│   │   └── beh/
│   │       └── *_beh.tsv     # 行为数据（PVT）
│   └── ses-2/
│       ├── eeg/
│       │   ├── *_eeg.set
│       │   └── *_eeg.fdt
│       └── beh/
│           └── *_beh.tsv

预处理 (Preprocessing)
├── 1. 导入 EEGLAB 数据
├── 2. 重参考（平均参考）
├── 3. 滤波（0.5-45 Hz）
├── 4. 降采样（500 Hz → 250 Hz）
├── 5. 坏道插值
├── 6. ICA 去伪迹（眼电、肌电）
└── 7. 分段（连续数据 → 2s epochs）

特征提取 (Feature Extraction)
├── 频谱功率（delta, theta, alpha, beta, gamma）
├── 频谱指数（spectral exponent）
├── 频谱斜率（spectral slope）
├── 功能连接（相干性、相位锁值）
└── 微状态（microstate analysis）

统计分析 (Statistical Analysis)
├── Aim 1: 睡眠剥夺对 EEG 频谱的影响（LMM）
├── Aim 2: 个体差异的稳定性（ICC）
├── Aim 3: 多维表型结构（PLS）
├── Aim 4: 预测模型（嵌套 CV）
└── Aim 5: 中介分析（情绪、焦虑、嗜睡）

输出 (Outputs)
├── results/tables/           # 统计结果表格
├── results/figures/          # 图表
├── results/models/           # 预测模型
└── manuscript/               # 论文稿件
"""
    
    print(flow_diagram)
    
    # 保存流程图
    with open(RESULTS_DIR / "data_flow_diagram.txt", "w") as f:
        f.write(flow_diagram)
    
    print(f"✓ 数据流程图已保存: {RESULTS_DIR / 'data_flow_diagram.txt'}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Phase 1: 数据审计 (Data Audit)")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"原始数据目录: {RAW_DATA_DIR}")
    
    # 创建必要的目录
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 检查 BIDS 结构
    participants_tsv, subjects_with_data = check_bids_structure()
    
    # 2. 创建变量字典
    variables = create_variable_dictionary(participants_tsv)
    
    # 3. 审计 pilot 数据
    pilot_subjects = audit_pilot_data(subjects_with_data, n_pilot=10)
    
    # 4. 生成数据流程图
    generate_data_flow_diagram()
    
    print("\n" + "=" * 80)
    print("Phase 1 完成!")
    print("=" * 80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return participants_tsv, subjects_with_data, pilot_subjects


if __name__ == "__main__":
    participants_tsv, subjects_with_data, pilot_subjects = main()
