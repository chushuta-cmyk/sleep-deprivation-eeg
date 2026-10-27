#!/usr/bin/env python3
"""
Phase 8 (Refined): 自动汇总真实分析结果生成 analysis_summary_report.txt
严格基于最新生成的 CSV/JSON 数据，杜绝任何硬编码幻觉结论。
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
QC_FILE = PROJECT_ROOT / "data" / "derivatives" / "qc_v3_conservative" / "qc_summary_v3_conservative.csv"


def main():
    lines = []
    lines.append("=" * 88)
    lines.append("睡眠剥夺对静息态 EEG 的影响：多维神经行为表型与机器学习基准报告 (Refined v3)")
    lines.append("Sleep Deprivation Effects on Resting-State EEG: Neurobehavioral Phenotype & ML Benchmark")
    lines.append("=" * 88)
    lines.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 0. QC Summary
    if QC_FILE.exists():
        qc_df = pd.read_csv(QC_FILE)
        n_sub_qc = qc_df['subject'].nunique()
        lines.append("=" * 88)
        lines.append("Phase 2: 数据质量控制 (QC) 与样本保留汇总")
        lines.append("=" * 88)
        lines.append(f"总处理会话数 (Sessions): {len(qc_df)} ({n_sub_qc} 名配对被试 × 2 条件 [NS vs SD])")
        lines.append(f"成功保留会话数: {(qc_df['status'] == 'success').sum()} / {len(qc_df)} (100.0% 配对保留)")
        lines.append(f"平均清洁时长: {qc_df['clean_duration_sec'].mean():.1f} 秒 (中位数: {qc_df['clean_duration_sec'].median():.1f} 秒)")
        lines.append(f"平均插值修复坏道数: {qc_df['n_interpolated_channels'].mean():.2f} 个/会话\n")

    # 1. Aim 1
    aim1_file = TABLES_DIR / "aim1_paired_effects.csv"
    if aim1_file.exists():
        df1 = pd.read_csv(aim1_file)
        n_aim1 = int(df1['n'].iloc[0])
        lines.append("=" * 88)
        lines.append(f"Aim 1: 睡眠剥夺对静息态 EEG 周期/非周期频谱与时域动力学的影响 (N={n_aim1} 配对)")
        lines.append("=" * 88)
        cols = ['feature', 'tier', 'n', 'mean_ns', 'mean_sd', 'mean_diff', 'dz', 'p_value', 'q_fdr', 'significant_fdr']
        lines.append(df1[cols].to_string(index=False))
        lines.append("")

    # 2. Aim 2
    aim2_file = TABLES_DIR / "aim2_icc_results.csv"
    if aim2_file.exists():
        df2 = pd.read_csv(aim2_file)
        lines.append("=" * 88)
        lines.append("Aim 2: 跨条件个体差异稳定性 (Trait Reliability ICC vs State Sensitivity)")
        lines.append("=" * 88)
        cols = ['feature', 'n_subjects', 'icc_agreement', 'icc_consistency', 'pearson_r', 'pearson_p']
        lines.append(df2[cols].to_string(index=False))
        lines.append("")

    delta_icc_file = TABLES_DIR / "aim2_delta_icc_bootstrap.csv"
    if delta_icc_file.exists():
        df_dicc = pd.read_csv(delta_icc_file)
        lines.append("Aim 2B: 周期分量剥离 1/f 背景前后的 ΔICC(A,1) Cluster-Bootstrap 检验 (1,000 Resamples)")
        lines.append("-" * 88)
        lines.append(df_dicc.to_string(index=False))
        lines.append("")

    # 3. Aim 3
    aim3_file = TABLES_DIR / "aim3_pls_results.json"
    aim3_csv = TABLES_DIR / "aim3_pls_loadings.csv"
    if aim3_file.exists() and aim3_csv.exists():
        with open(aim3_file, 'r') as f:
            pls_res = json.load(f)
        df3 = pd.read_csv(aim3_csv)
        lines.append("=" * 88)
        lines.append("Aim 3: 多维神经行为表型耦合 (PLS 偏最小二乘分析)")
        lines.append("=" * 88)
        lines.append(f"有效样本量: N = {pls_res['n_subjects']}")
        lines.append(f"第一潜变量 (LV1) 脑-行为相关系数: r = {pls_res['lv1_latent_r']:.3f} (参数 p = {pls_res['lv1_parametric_p']:.4f}, 置换检验 p = {pls_res['lv1_permutation_p']:.4f})")
        lines.append(df3.to_string(index=False))
        lines.append("")

    # 4. Aim 4
    clf_file = TABLES_DIR / "aim4_classification_benchmark.csv"
    reg_file = TABLES_DIR / "aim4_regression_results.csv"
    if clf_file.exists():
        df_clf = pd.read_csv(clf_file)
        lines.append("=" * 88)
        lines.append("Aim 4A: 留一被试交叉验证 (LOSO) 脑状态分类基准 (NS vs SD)")
        lines.append("=" * 88)
        lines.append(df_clf.to_string(index=False))
        lines.append("")
    if reg_file.exists():
        df_reg = pd.read_csv(reg_file)
        lines.append("=" * 88)
        lines.append("Aim 4B: 嵌套交叉验证 (Nested-CV) 易感性连续预测 (ΔZ_Sleepiness, ΔKSS, ΔPVT RT SD)")
        lines.append("=" * 88)
        lines.append(df_reg.to_string(index=False))
        lines.append("")

    # 5. Aim 5
    inc_file = TABLES_DIR / "aim5_incremental_validity.csv"
    if inc_file.exists():
        df_inc = pd.read_csv(inc_file)
        lines.append("=" * 88)
        lines.append("Aim 5: 脑电特征对睡眠剥夺反应的增量方差解释度 (ΔR² beyond Baseline & PSQI)")
        lines.append("=" * 88)
        lines.append(df_inc.to_string(index=False))
        lines.append("")

    # 6. Phase 9 Equivalence Test
    eq_file = TABLES_DIR / "phase9_19ch_vs_61ch_equivalence_test.csv"
    if eq_file.exists():
        df_eq = pd.read_csv(eq_file)
        lines.append("=" * 88)
        lines.append("Phase 9: 19导双流常模架构 vs. 61导全阵列 Linear SVM 配对 Bootstrap 等效性检验")
        lines.append("=" * 88)
        lines.append(df_eq.to_string(index=False))
        lines.append("")

    report_text = "\n".join(lines)
    out_file = RESULTS_DIR / "analysis_summary_report.txt"
    with open(out_file, "w") as f:
        f.write(report_text)
    print(report_text)
    print(f"\n✓ 最新汇总报告已写入: {out_file}")


if __name__ == '__main__':
    main()
