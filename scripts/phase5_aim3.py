#!/usr/bin/env python3
"""
Phase 5 (Refined Full-Cohort): Aim 3 - 多维神经行为表型结构 (PLS 偏最小二乘分析)
升级项：
1. 全面支持 ds004902 全样本 (sub-01 ~ sub-71)：
   - Cohort-Wide PLS (全样本统一主观困倦 Z 变化 ΔSleepiness_Z + 基线 PSQI)
   - Sub-study 1 Objective Vigilance PLS (sub-01~38, 70~71: 客观警觉度反应时 ΔPVT_RT, 反应时变异性 ΔPVT_RTSD, ΔSSS, ΔPANAS)
   - Sub-study 2 Affective/Sleepiness PLS (sub-39~71: ΔKSS 与状态焦虑 ΔSAI)
2. 识别并规避 OpenNeuro ds004902 原始 participants.tsv 中 sub-39~69 的 PANAS_SD 列被错误复制为 SAI 的元数据污染，
   仅在无污染的 sub-01~38 & sub-70~71 中使用 PANAS_SD。
3. 在 Bootstrap 重抽样中加入 Procrustes 符号对齐 (Sign Alignment)，解决潜变量方向翻转导致的伪方差膨胀问题。
"""

import os
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings('ignore')

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "analysis_config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / config['paths']['features'] / 'v3_conservative'
RESULTS_DIR = PROJECT_ROOT / config['paths']['results']
(RESULTS_DIR / 'tables').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'figures').mkdir(parents=True, exist_ok=True)

STATS = config['statistics']


def prepare_behavioral_dataframe():
    participants_file = PROJECT_ROOT / 'data' / 'raw' / 'participants.tsv'
    part_df = pd.read_csv(participants_file, sep='\t')

    num_cols = [
        'KSS_NS', 'KSS_SD', 'SSS_NS', 'SSS_SD',
        'SAI_NS', 'SAI_SD',
        'PVT_item1_NS', 'PVT_item1_SD',
        'PVT_item2_NS', 'PVT_item2_SD',
        'PVT_item3_NS', 'PVT_item3_SD',
        'PANAS_P_NS', 'PANAS_P_SD', 'PANAS_N_NS', 'PANAS_N_SD',
        'ATQ_NS', 'ATQ_SD',
        'PSQI_GlobalScore', 'EQ', 'Buss_Perry', 'Age'
    ]
    for c in num_cols:
        if c in part_df.columns:
            part_df[c] = pd.to_numeric(part_df[c], errors='coerce')

    # 1. 主观嗜睡变化量 (KSS 与 SSS) 及全样本标准化主观困倦变化量 (Sleepiness_Z_change)
    part_df['KSS_change'] = part_df['KSS_SD'] - part_df['KSS_NS']
    part_df['SSS_change'] = part_df['SSS_SD'] - part_df['SSS_NS']

    kss_z_ns = (part_df['KSS_NS'] - part_df['KSS_NS'].mean()) / (part_df['KSS_NS'].std() + 1e-8)
    sss_z_ns = (part_df['SSS_NS'] - part_df['SSS_NS'].mean()) / (part_df['SSS_NS'].std() + 1e-8)
    part_df['Sleepiness_Z_NS'] = kss_z_ns.combine_first(sss_z_ns)

    kss_z_chg = (part_df['KSS_change'] - part_df['KSS_change'].mean()) / (part_df['KSS_change'].std() + 1e-8)
    sss_z_chg = (part_df['SSS_change'] - part_df['SSS_change'].mean()) / (part_df['SSS_change'].std() + 1e-8)
    part_df['Sleepiness_Z_change'] = kss_z_chg.combine_first(sss_z_chg)

    # 2. 客观警觉度 PVT 变化量 (sub-01~38, sub-70~71)
    part_df['PVT_Lapses_change'] = part_df['PVT_item1_SD'] - part_df['PVT_item1_NS']
    part_df['PVT_RT_change'] = part_df['PVT_item2_SD'] - part_df['PVT_item2_NS']
    part_df['PVT_RTSD_change'] = part_df['PVT_item3_SD'] - part_df['PVT_item3_NS']

    # 3. 状态焦虑变化量 (SAI, sub-39~69)
    part_df['SAI_change'] = part_df['SAI_SD'] - part_df['SAI_NS']

    # 4. 无污染 PANAS 变化量 (仅保留 SAI_NS 为 NaN 的真实 PANAS 子集: sub-01~38, sub-70~71)
    valid_panas_mask = part_df['SAI_NS'].isna()
    part_df['PANAS_P_change_clean'] = np.where(valid_panas_mask, part_df['PANAS_P_SD'] - part_df['PANAS_P_NS'], np.nan)
    part_df['PANAS_N_change_clean'] = np.where(valid_panas_mask, part_df['PANAS_N_SD'] - part_df['PANAS_N_NS'], np.nan)

    return part_df


def run_pls_analysis(X, Y, x_features, y_features, n_permutations=5000, n_bootstrap=2000):
    np.random.seed(42)
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = StandardScaler().fit_transform(Y)

    n_components = min(2, X.shape[1], Y.shape[1])
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, Y_scaled)

    x_scores = pls.x_scores_[:, 0]
    y_scores = pls.y_scores_[:, 0]
    if np.corrcoef(x_scores, y_scores)[0, 1] < 0:
        pls.x_weights_[:, 0] *= -1
        pls.x_scores_[:, 0] *= -1
        x_scores = pls.x_scores_[:, 0]

    observed_cov = float(np.abs(np.sum(x_scores * y_scores)))
    observed_r, observed_r_p = stats.pearsonr(x_scores, y_scores)

    # 1. 置换检验
    perm_covs = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_idx = np.random.permutation(len(Y_scaled))
        pls_perm = PLSRegression(n_components=1)
        pls_perm.fit(X_scaled, Y_scaled[perm_idx])
        xs_p = pls_perm.x_scores_[:, 0]
        ys_p = pls_perm.y_scores_[:, 0]
        perm_covs[i] = np.abs(np.sum(xs_p * ys_p))

    p_value_cov = float((np.sum(perm_covs >= observed_cov) + 1) / (n_permutations + 1))

    # 2. 带 Procrustes 符号对齐的 Bootstrap 稳定性估计
    orig_xw = pls.x_weights_[:, 0].copy()
    orig_yw = pls.y_weights_[:, 0].copy()

    X_boot_w = np.zeros((n_bootstrap, X.shape[1]))
    Y_boot_w = np.zeros((n_bootstrap, Y.shape[1]))

    for i in range(n_bootstrap):
        boot_idx = np.random.choice(len(X_scaled), size=len(X_scaled), replace=True)
        if len(np.unique(boot_idx)) < 5:
            continue
        pls_b = PLSRegression(n_components=1)
        pls_b.fit(X_scaled[boot_idx], Y_scaled[boot_idx])
        xw = pls_b.x_weights_[:, 0]
        yw = pls_b.y_weights_[:, 0]
        if np.dot(xw, orig_xw) < 0:
            xw = -xw
            yw = -yw
        X_boot_w[i, :] = xw
        Y_boot_w[i, :] = yw

    X_boot_se = np.std(X_boot_w, axis=0, ddof=1) + 1e-8
    Y_boot_se = np.std(Y_boot_w, axis=0, ddof=1) + 1e-8
    X_boot_ratio = orig_xw / X_boot_se
    Y_boot_ratio = orig_yw / Y_boot_se

    stable_X = np.abs(X_boot_ratio) >= 1.96
    stable_Y = np.abs(Y_boot_ratio) >= 1.96

    results = {
        'n_subjects': int(len(X)),
        'n_components': int(n_components),
        'lv1_covariance': observed_cov,
        'lv1_latent_r': float(observed_r),
        'lv1_parametric_p': float(observed_r_p),
        'lv1_permutation_p': p_value_cov,
        'x_features': x_features,
        'y_features': y_features,
        'x_weights': orig_xw.tolist(),
        'y_weights': orig_yw.tolist(),
        'x_bootstrap_ratio': X_boot_ratio.tolist(),
        'y_bootstrap_ratio': Y_boot_ratio.tolist(),
        'stable_x_features': [x_features[i] for i in range(len(x_features)) if stable_X[i]],
        'stable_y_features': [y_features[i] for i in range(len(y_features)) if stable_Y[i]],
    }
    return pls, results


def plot_pls_results(pls, results):
    x_features = results['x_features']
    y_features = results['y_features']
    x_br = np.array(results['x_bootstrap_ratio'])
    y_br = np.array(results['y_bootstrap_ratio'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    y_pos = np.arange(len(x_features))
    colors_x = ['#c0392b' if abs(r) >= 1.96 else '#7f8c8d' for r in x_br]
    ax.barh(y_pos, x_br, color=colors_x, alpha=0.8, edgecolor='black')
    ax.axvline(1.96, color='#c0392b', linestyle='--', alpha=0.6, label='|BSR| = 1.96 (p<0.05)')
    ax.axvline(-1.96, color='#c0392b', linestyle='--', alpha=0.6)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace('_change', ' Δ') for f in x_features], fontsize=9.5)
    ax.set_xlabel('Bootstrap Ratio (Weight / SE)', fontweight='bold')
    ax.set_title('EEG Salience (LV1 Bootstrap Ratios)', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8.5)

    ax = axes[1]
    y_pos_b = np.arange(len(y_features))
    colors_y = ['#2980b9' if abs(r) >= 1.96 else '#7f8c8d' for r in y_br]
    ax.barh(y_pos_b, y_br, color=colors_y, alpha=0.8, edgecolor='black')
    ax.axvline(1.96, color='#2980b9', linestyle='--', alpha=0.6, label='|BSR| = 1.96 (p<0.05)')
    ax.axvline(-1.96, color='#2980b9', linestyle='--', alpha=0.6)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y_pos_b)
    ax.set_yticklabels(y_features, fontsize=9.5)
    ax.set_xlabel('Bootstrap Ratio (Weight / SE)', fontweight='bold')
    ax.set_title('Neurobehavioral Salience (LV1 Bootstrap Ratios)', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figures' / 'aim3_pls_weights.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Latent Scores Scatter Plot
    fig, ax = plt.subplots(figsize=(7, 5.5))
    xs = pls.x_scores_[:, 0]
    ys = pls.y_scores_[:, 0]
    ax.scatter(xs, ys, s=65, color='#2c3e50', alpha=0.75, edgecolors='black')
    slope, intercept, r_val, p_val, _ = stats.linregress(xs, ys)
    x_line = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color='#c0392b', linewidth=2.2,
            label=f"Latent r = {r_val:.2f} (Perm p = {results['lv1_permutation_p']:.4f})")
    ax.set_xlabel('EEG Latent Score (LV1)', fontweight='bold')
    ax.set_ylabel('Behavioral/Vigilance Latent Score (LV1)', fontweight='bold')
    ax.set_title(f"Aim 3: Multivariate Brain-Behavior Coupling (n={results['n_subjects']})", fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figures' / 'aim3_pls_scores.png', dpi=200, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 80)
    print("Phase 5: Aim 3 - 多维神经行为表型结构 (PLS, Full Cohort + Sub-cohorts)")
    print("=" * 80)

    change_df = pd.read_parquet(FEATURES_DIR / 'eeg_change_features.parquet')
    part_df = prepare_behavioral_dataframe()
    merged_df = change_df.merge(part_df, left_on='subject', right_on='participant_id', how='left')

    eeg_features = [
        'posterior_alpha_periodic_power_change',
        'frontal_theta_logpower_change',
        'global_delta_logpower_change',
        'global_spectral_exponent_change',
        'global_hjorth_complexity_change'
    ]
    eeg_features = [f for f in eeg_features if f in merged_df.columns]

    n_perm = int(STATS.get('n_permutations', 5000))
    n_boot = int(STATS.get('n_bootstrap', 2000))

    # 1. 全样本综合 PLS (Unified Sleepiness_Z_change + Baseline Sleepiness_Z_NS)
    beh_full = ['Sleepiness_Z_change', 'Sleepiness_Z_NS']
    df_full = merged_df[['subject'] + eeg_features + beh_full].dropna().copy()
    pls_full, res_full = run_pls_analysis(
        df_full[eeg_features].values, df_full[beh_full].values,
        eeg_features, beh_full, n_permutations=n_perm, n_bootstrap=n_boot
    )
    print(f"[Full-Cohort PLS, n={res_full['n_subjects']}] LV1 r = {res_full['lv1_latent_r']:.3f}, Perm p = {res_full['lv1_permutation_p']:.4f}")

    # 2. Sub-cohort 1: PVT 客观警觉度与主观困倦 PLS (PVT_RT_change, PVT_RTSD_change, SSS_change)
    beh_pvt = ['PVT_RT_change', 'PVT_RTSD_change', 'SSS_change']
    df_pvt = merged_df[['subject'] + eeg_features + beh_pvt].dropna().copy()
    res_pvt = None
    if len(df_pvt) >= 10:
        pls_pvt, res_pvt = run_pls_analysis(
            df_pvt[eeg_features].values, df_pvt[beh_pvt].values,
            eeg_features, beh_pvt, n_permutations=n_perm, n_bootstrap=n_boot
        )
        print(f"[PVT Vigilance Sub-cohort PLS, n={res_pvt['n_subjects']}] LV1 r = {res_pvt['lv1_latent_r']:.3f}, Perm p = {res_pvt['lv1_permutation_p']:.4f}")

    # 3. Sub-cohort 2: KSS 主观嗜睡与 SAI 状态焦虑 PLS (KSS_change, SAI_change)
    beh_kss = ['KSS_change', 'SAI_change']
    df_kss = merged_df[['subject'] + eeg_features + beh_kss].dropna().copy()
    pls_kss, res_kss = run_pls_analysis(
        df_kss[eeg_features].values, df_kss[beh_kss].values,
        eeg_features, beh_kss, n_permutations=n_perm, n_bootstrap=n_boot
    )
    print(f"[KSS + SAI Sub-cohort PLS, n={res_kss['n_subjects']}] LV1 r = {res_kss['lv1_latent_r']:.3f}, Perm p = {res_kss['lv1_permutation_p']:.4f}")

    # 汇总保存 (主结果采用 Full-Cohort PLS，同时附带 PVT 与 KSS+SAI 子队列 PLS)
    res_full['subcohort_pvt_pls'] = res_pvt
    res_full['subcohort_kss_sai_pls'] = res_kss

    with open(RESULTS_DIR / 'tables' / 'aim3_pls_results.json', 'w') as f:
        json.dump(res_full, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / 'aim3_pls_results.json', 'w') as f:
        json.dump(res_full, f, indent=2, ensure_ascii=False)

    loadings_rows = []
    for cohort_label, res_obj in [('Full_Cohort_Sleepiness', res_full), ('Subcohort_PVT_Vigilance', res_pvt), ('Subcohort_KSS_SAI', res_kss)]:
        if res_obj is None:
            continue
        for f_name, w, bsr in zip(res_obj['x_features'], res_obj['x_weights'], res_obj['x_bootstrap_ratio']):
            loadings_rows.append({'cohort': cohort_label, 'domain': 'EEG', 'feature': f_name, 'weight': w, 'bootstrap_ratio': bsr, 'stable_95ci': abs(bsr) >= 1.96})
        for f_name, w, bsr in zip(res_obj['y_features'], res_obj['y_weights'], res_obj['y_bootstrap_ratio']):
            loadings_rows.append({'cohort': cohort_label, 'domain': 'Behavioral', 'feature': f_name, 'weight': w, 'bootstrap_ratio': bsr, 'stable_95ci': abs(bsr) >= 1.96})

    loadings_df = pd.DataFrame(loadings_rows)
    loadings_df.to_csv(RESULTS_DIR / 'tables' / 'aim3_pls_loadings.csv', index=False)

    plot_pls_results(pls_full, res_full)
    print("\n✓ Aim 3 PLS 载荷汇总表已保存。")


if __name__ == '__main__':
    main()
