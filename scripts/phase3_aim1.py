#!/usr/bin/env python3
"""
Phase 3 (Refined): Aim 1 - 睡眠剥夺对静息态 EEG 周期性振荡、非周期 1/f 背景及时域动态的影响
修复与升级项：
1. 修正原脚本中 session_order 编码错误 ('SD-NS' -> 'SD->NS')。
2. 联合输出配对 t 检验、Wilcoxon 符号秩检验、Cohen's dz 效应量及线性混合效应模型 (LMM, 控制顺序与清洁时长协变量)。
3. 同时评估主特征（5项）与扩展机理特征（剥离 1/f 后的纯周期功率、1/f 截距、Theta/Alpha 比值、Hjorth 复杂度），分别进行 Benjamini-Hochberg FDR 校正。
"""

import os
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

CONFIG_PATH = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/sleep_deprivation_eeg/config/analysis_config.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

PROJECT_ROOT = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/sleep_deprivation_eeg")
FEATURES_DIR = PROJECT_ROOT / config['paths']['features'] / 'v3_conservative'
RESULTS_DIR = PROJECT_ROOT / config['paths']['results']
(RESULTS_DIR / 'tables').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'figures').mkdir(parents=True, exist_ok=True)

STATS = config['statistics']


def load_session_features():
    session_file = FEATURES_DIR / 'eeg_session_features.parquet'
    df = pd.read_parquet(session_file)

    participants_file = PROJECT_ROOT / 'data' / 'raw' / 'participants.tsv'
    if participants_file.exists():
        participants = pd.read_csv(participants_file, sep='\t')
        cols_to_merge = ['participant_id', 'Gender', 'Age', 'SessionOrder']
        cols_to_merge = [c for c in cols_to_merge if c in participants.columns]
        df = df.merge(participants[cols_to_merge], left_on='subject', right_on='participant_id', how='left')
        if 'session_order' not in df.columns and 'SessionOrder' in df.columns:
            df['session_order'] = df['SessionOrder']
    return df


def compute_paired_stats(ns_data, sd_data):
    diff = sd_data - ns_data
    n = len(diff)
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    dz = mean_diff / sd_diff if sd_diff > 0 else 0.0
    se_dz = float(np.sqrt(1.0 / n + (dz ** 2) / (2.0 * n)))

    t_stat, p_ttest = stats.ttest_rel(sd_data, ns_data)
    try:
        w_stat, p_wilcox = stats.wilcoxon(sd_data, ns_data)
    except Exception:
        w_stat, p_wilcox = np.nan, np.nan

    return {
        'n': int(n),
        'mean_ns': float(np.mean(ns_data)),
        'sd_ns': float(np.std(ns_data, ddof=1)),
        'mean_sd': float(np.mean(sd_data)),
        'sd_sd': float(np.std(sd_data, ddof=1)),
        'mean_diff': mean_diff,
        'sd_diff': sd_diff,
        't_stat': float(t_stat),
        'p_value': float(p_ttest),
        'p_wilcoxon': float(p_wilcox),
        'dz': float(dz),
        'se_dz': se_dz,
        'ci_lower': float(dz - 1.96 * se_dz),
        'ci_upper': float(dz + 1.96 * se_dz),
    }


def run_lmm_analysis(session_df, feature_col):
    df = session_df[['subject', 'condition', 'session_order', 'clean_duration_sec', feature_col]].dropna().copy()
    df['condition_num'] = (df['condition'] == 'SD').astype(int)
    df['session_order_num'] = (df['session_order'].astype(str).str.contains('SD->NS')).astype(int)
    df['clean_duration_z'] = (df['clean_duration_sec'] - df['clean_duration_sec'].mean()) / (df['clean_duration_sec'].std() + 1e-8)

    try:
        model = smf.mixedlm(
            f"{feature_col} ~ condition_num + session_order_num + clean_duration_z",
            df,
            groups=df['subject']
        )
        result = model.fit(reml=True, verbose=False)
        params = result.params
        pvalues = result.pvalues
        conf_int = result.conf_int()

        return {
            'feature': feature_col,
            'condition_coef': float(params['condition_num']),
            'condition_se': float(result.bse['condition_num']),
            'condition_p': float(pvalues['condition_num']),
            'condition_ci_lower': float(conf_int.loc['condition_num', 0]),
            'condition_ci_upper': float(conf_int.loc['condition_num', 1]),
            'session_order_coef': float(params['session_order_num']),
            'session_order_p': float(pvalues['session_order_num']),
            'clean_duration_coef': float(params['clean_duration_z']),
            'clean_duration_p': float(pvalues['clean_duration_z']),
            'n_subjects': int(df['subject'].nunique()),
            'n_observations': int(len(df)),
        }
    except Exception as e:
        print(f"  ⚠ LMM 拟合回退 ({feature_col}): {e}")
        return None


def plot_paired_effects(session_df, features_to_plot, paired_df):
    n_features = len(features_to_plot)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.8 * n_rows))
    axes = np.array(axes).flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]
        pivot_df = session_df.pivot(index='subject', columns='condition', values=feature).dropna()
        ns_data = pivot_df['NS'].values
        sd_data = pivot_df['SD'].values

        parts = ax.violinplot([ns_data, sd_data], positions=[0, 1], widths=0.55, showmeans=False, showmedians=False)
        for i_pc, pc in enumerate(parts['bodies']):
            pc.set_facecolor('#3498db' if i_pc == 0 else '#e74c3c')
            pc.set_alpha(0.35)
            pc.set_edgecolor('black')

        for i in range(len(ns_data)):
            ax.plot([0, 1], [ns_data[i], sd_data[i]], color='gray', alpha=0.35, linewidth=1.0)
            ax.scatter([0, 1], [ns_data[i], sd_data[i]], color=['#2980b9', '#c0392b'], s=22, alpha=0.7, zorder=3)

        mean_ns, mean_sd = np.mean(ns_data), np.mean(sd_data)
        se_ns, se_sd = stats.sem(ns_data), stats.sem(sd_data)
        ax.errorbar(0, mean_ns, yerr=1.96 * se_ns, fmt='o', color='#1b4f72', markersize=9, capsize=5, capthick=2, zorder=4)
        ax.errorbar(1, mean_sd, yerr=1.96 * se_sd, fmt='o', color='#78281f', markersize=9, capsize=5, capthick=2, zorder=4)

        row_stat = paired_df[paired_df['feature'] == feature].iloc[0]
        q_str = f"q={row_stat['q_fdr']:.3f}" if 'q_fdr' in row_stat else ""
        ax.text(
            0.5, 0.96,
            f"n={int(row_stat['n'])} | dz={row_stat['dz']:.2f}\np={row_stat['p_value']:.4f} ({q_str})",
            ha='center', va='top', transform=ax.transAxes, fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fcf3cf', alpha=0.85, edgecolor='#d4ac0d')
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Normal Sleep (NS)', 'Sleep Deprivation (SD)'], fontweight='bold')
        ax.set_ylabel(feature)
        ax.set_title(feature.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    for idx in range(len(features_to_plot), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig_path = RESULTS_DIR / 'figures' / 'aim1_paired_effects.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ 配对效应图已保存: {fig_path}")


def main():
    print("=" * 80)
    print("Phase 3: Aim 1 - 睡眠剥夺对静息态 EEG 特征的影响 (N=全配对样本)")
    print("=" * 80)

    session_df = load_session_features()
    primary_features = config['analysis']['primary_features']
    extended_features = [
        'posterior_alpha_periodic_power',
        'frontal_theta_periodic_power',
        'global_theta_logpower',
        'global_alpha_logpower',
        'global_beta_logpower',
        'frontal_theta_posterior_alpha_ratio',
        'global_hjorth_mobility',
        'global_hjorth_complexity',
    ]
    all_target_features = [f for f in (primary_features + extended_features) if f in session_df.columns]

    paired_records = []
    lmm_records = []

    for feat in all_target_features:
        pivot_df = session_df.pivot(index='subject', columns='condition', values=feat).dropna()
        if len(pivot_df) >= 5:
            p_stat = compute_paired_stats(pivot_df['NS'].values, pivot_df['SD'].values)
            p_stat['feature'] = feat
            p_stat['tier'] = 'Primary' if feat in primary_features else 'Extended'
            paired_records.append(p_stat)

        lmm_res = run_lmm_analysis(session_df, feat)
        if lmm_res:
            lmm_res['tier'] = 'Primary' if feat in primary_features else 'Extended'
            lmm_records.append(lmm_res)

    paired_df = pd.DataFrame(paired_records)
    # 分别在 Primary (5项) 和全集上进行 BH-FDR 校正
    prim_mask = paired_df['tier'] == 'Primary'
    paired_df['q_fdr'] = np.nan
    paired_df['significant_fdr'] = False
    if prim_mask.sum() > 0:
        rej, qvals, _, _ = multipletests(paired_df.loc[prim_mask, 'p_value'], alpha=0.05, method='fdr_bh')
        paired_df.loc[prim_mask, 'q_fdr'] = qvals
        paired_df.loc[prim_mask, 'significant_fdr'] = rej
    ext_mask = ~prim_mask
    if ext_mask.sum() > 0:
        rej_e, qvals_e, _, _ = multipletests(paired_df.loc[ext_mask, 'p_value'], alpha=0.05, method='fdr_bh')
        paired_df.loc[ext_mask, 'q_fdr'] = qvals_e
        paired_df.loc[ext_mask, 'significant_fdr'] = rej_e

    cols_order = [
        'feature', 'tier', 'n', 'mean_ns', 'sd_ns', 'mean_sd', 'sd_sd',
        'mean_diff', 'sd_diff', 'dz', 'se_dz', 'ci_lower', 'ci_upper',
        't_stat', 'p_value', 'q_fdr', 'p_wilcoxon', 'significant_fdr'
    ]
    paired_df = paired_df[[c for c in cols_order if c in paired_df.columns]]
    paired_file = RESULTS_DIR / 'tables' / 'aim1_paired_effects.csv'
    paired_df.to_csv(paired_file, index=False)
    print(f"\n✓ 配对统计表已保存: {paired_file}")
    print(paired_df[['feature', 'tier', 'n', 'mean_diff', 'dz', 'p_value', 'q_fdr']].to_string(index=False))

    lmm_df = pd.DataFrame(lmm_records)
    if len(lmm_df) > 0:
        p_mask = lmm_df['tier'] == 'Primary'
        lmm_df['condition_p_fdr'] = np.nan
        lmm_df['significant_fdr'] = False
        if p_mask.sum() > 0:
            rej, qvals, _, _ = multipletests(lmm_df.loc[p_mask, 'condition_p'], alpha=0.05, method='fdr_bh')
            lmm_df.loc[p_mask, 'condition_p_fdr'] = qvals
            lmm_df.loc[p_mask, 'significant_fdr'] = rej
        if (~p_mask).sum() > 0:
            rej_e, qvals_e, _, _ = multipletests(lmm_df.loc[~p_mask, 'condition_p'], alpha=0.05, method='fdr_bh')
            lmm_df.loc[~p_mask, 'condition_p_fdr'] = qvals_e
            lmm_df.loc[~p_mask, 'significant_fdr'] = rej_e

        lmm_file = RESULTS_DIR / 'tables' / 'aim1_lmm_results.csv'
        lmm_df.to_csv(lmm_file, index=False)
        print(f"\n✓ LMM 统计表已保存: {lmm_file}")

    # 绘图
    plot_paired_effects(session_df, [f for f in primary_features if f in session_df.columns], paired_df)

    # 森林图 (Cohen's dz)
    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(paired_df))[::-1]
    colors = ['#c0392b' if sig else '#2c3e50' for sig in paired_df['significant_fdr']]
    ax.errorbar(
        paired_df['dz'], y_pos,
        xerr=[paired_df['dz'] - paired_df['ci_lower'], paired_df['ci_upper'] - paired_df['dz']],
        fmt='o', color='#2c3e50', ecolor='#7f8c8d', elinewidth=2, capsize=4
    )
    ax.scatter(paired_df['dz'], y_pos, color=colors, s=70, zorder=3)
    ax.axvline(0, color='black', linestyle='--', alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(paired_df['feature'])
    ax.set_xlabel("Paired Effect Size (Cohen's dz, SD - NS)", fontweight='bold')
    ax.set_title("Aim 1: Standardized Effects of Sleep Deprivation on Resting-State EEG", fontweight='bold')
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figures' / 'aim1_forest_plot.png', dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
