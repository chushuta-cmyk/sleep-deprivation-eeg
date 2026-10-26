#!/usr/bin/env python3
"""
One-Click End-to-End Pipeline Runner
按顺序执行完整分析流程：
1. 02a_preprocess_v3_conservative.py (鲁棒预处理、球面样条坏道修复、ICA 眼电剔除)
2. 02b_qc_review.py                  (QC 质量审计与分布可视化)
3. 02c_extract_features.py           (参数化 1/f 非周期与纯周期功率、Hjorth 特征提取)
4. phase3_aim1.py                    (Aim 1: 配对效应 dz、Wilcoxon、LMM 与 FDR 校正)
5. phase4_aim2.py                    (Aim 2: 跨条件特质稳定性 ICC(A,1) & ICC(C,1))
6. phase5_aim3.py                    (Aim 3: Procrustes 对齐 Bootstrap PLS 脑-行为耦合)
7. phase6_aim4.py                    (Aim 4: LOSO 状态分类基准 + Nested-CV 易感性回归)
8. phase7_aim5.py                    (Aim 5: 增量效度 ΔR² F检验 + Bootstrap 中介分析)
9. 09_normative_csd_19ch_validation.py (Phase 9: A+B 19导 CAR+CSD 双流 + MST 构图 + 常模 Z 校准验证)
10. phase8_generate_report.py        (自动汇总所有真实指标生成总报告)
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

PIPELINE_STEPS = [
    "02a_preprocess_v3_conservative.py",
    "02b_qc_review.py",
    "02c_extract_features.py",
    "phase3_aim1.py",
    "phase4_aim2.py",
    "phase5_aim3.py",
    "phase6_aim4.py",
    "phase7_aim5.py",
    "09_normative_csd_19ch_validation.py",
    "phase8_generate_report.py",
]


def main():
    for step in PIPELINE_STEPS:
        script_path = SCRIPTS_DIR / step
        print(f"\n>>> Running {step} ...")
        res = subprocess.run([sys.executable, str(script_path)], check=True)
    print("\n✓ All pipeline stages completed successfully!")


if __name__ == "__main__":
    main()
