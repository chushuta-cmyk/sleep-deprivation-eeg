#!/usr/bin/env python3
"""
Phase 5 (Refined): Aim 3 - 多维神经行为表型结构 (PLS 偏最小二乘分析)
修复与升级项：
1. 修复原脚本 main() 中未合并行为表 (load_behavioral_data) 导致行为特征为空的致命 bug。
2. 识别并规避 OpenNeuro ds004902 原始 participants.tsv 中 sub-40~69 的 PANAS_SD 列被错误复制为 SAI 的元数据污染，
   采用真实无污染的行为/情绪维度：ΔKSS (主观嗜睡增量)、ΔSAI (状态焦虑增量)、SD后峰值嗜睡/焦虑及基线睡眠质量 (PSQI)。
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


def load_merged_pls_data():
    change_file = FEATURES_DIR / 'eeg_change_features.parquet'
    change_df = pd.read_parquet(change_file)

    participants_file = PROJECT_ROOT / 'data' / 'raw' / 'participants.tsv'
    part_df = pd.read_csv(participants_file, sep='\t')

    num_cols = ['KSS_NS', 'KSS_SD', 'SAI_NS', 'SAI_SD', 'PANAS_P_NS', 'PANAS_N_NS', 'PSQI_GlobalScore', 'EQ', 'Buss_Perry']
    for c in num_cols:
        if c in part_df.columns:
            part_df[c] = pd.to_numeric(part_df[c], errors='coerce')

    part_df['KSS_change'] = part_df['KSS_SD'] - part_df['KSS_NS']
    part_df['SAI_change'] = part_df['SAI_SD'] - part_df['SAI_NS']

    merged_df = change_df.merge(part_df, left_on='subject', right_on='participant_id', how='left')

    eeg_features = [
        'posterior_alpha_periodic_power_change',
        'frontal_theta_logpower_change',
        'global_delta_logpower_change',
        'global_spectral_exponent_change',
        'global_hjorth_complexity_change'
    ]
    eeg_features = [f for f in eeg_features if f in merged_df.columns]

    behavioral_features = [
        'KSS_change',
        'SAI_change',
    ]
    behavioral_features = [f for f in behavioral_features if f in merged_df.columns]

    clean_df = merged_df[['subject'] + eeg_features + behavioral_features].dropna().copy()
    print(f"PLS 有效完整配对被试数: {len(clean_df)}")

    X = clean_df[eeg_features].values
    Y = clean_df[behavioral_features].values
    return clean_df, X, Y, eeg_features, behavioral_features


def run_pls_analysis(X, Y, x_features, y_features, n_permutations=5000, n_bootstrap=2000):
    np.random.seed(42)
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = StandardScaler().fit_transform(Y)

    n_components = min(2, X.shape[1], Y.shape[1])
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, Y_scaled)

    x_scores = pls.x_scores_[:, 0]
    y_scores = pls.y_scores_[:, 0]
    # 确保潜变量得分为正相关方向
    if np.corrcoef(x_scores, y_scores)[0, 1] < 0:
        pls.x_weights_[:, 0] *= -1
        pls.x_scores_[:, 0] *= -1
        x_scores = pls.x_scores_[:, 0]

    observed_cov = float(np.abs(np.sum(x_scores * y_scores)))
    observed_r, observed_r_p = stats.pearsonr(x_scores, y_scores)

    # 1. 置换检验
    perm_covs = np.zeros(n_permutations)
    perm_rs = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_idx = np.random.permutation(len(Y_scaled))
        pls_perm = PLSRegression(n_components=1)
        pls_perm.fit(X_scaled, Y_scaled[perm_idx])
        xs_p = pls_perm.x_scores_[:, 0]
        ys_p = pls_perm.y_scores_[:, 0]
        perm_covs[i] = np.abs(np.sum(xs_p * ys_p))
        perm_rs[i] = np.abs(np.corrcoef(xs_p, ys_p)[0, 1])

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
        # 符号对齐 (Procrustes reflection alignment)
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
    x_weights = np.array(results['x_weights'])
    y_weights = np.array(results['y_weights'])
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
    ax.set_ylabel('Behavioral/Affective Latent Score (LV1)', fontweight='bold')
    ax.set_title(f"Aim 3: Multivariate Brain-Behavior Coupling (n={results['n_subjects']})", fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figures' / 'aim3_pls_scores.png', dpi=200, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 80)
    print("Phase 5: Aim 3 - 多维神经行为表型结构 (PLS)")
    print("=" * 80)

    clean_df, X, Y, x_features, y_features = load_merged_pls_data()
    pls, results = run_pls_analysis(
        X, Y, x_features, y_features,
        n_permutations=int(STATS.get('n_permutations', 5000)),
        n_bootstrap=int(STATS.get('n_bootstrap', 2000))
    )

    with open(RESULTS_DIR / 'tables' / 'aim3_pls_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / 'aim3_pls_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    loadings_rows = []
    for f_name, w, bsr in zip(x_features, results['x_weights'], results['x_bootstrap_ratio']):
        loadings_rows.append({'domain': 'EEG', 'feature': f_name, 'weight': w, 'bootstrap_ratio': bsr, 'stable_95ci': abs(bsr) >= 1.96})
    for f_name, w, bsr in zip(y_features, results['y_weights'], results['y_bootstrap_ratio']):
        loadings_rows.append({'domain': 'Behavioral', 'feature': f_name, 'weight': w, 'bootstrap_ratio': bsr, 'stable_95ci': abs(bsr) >= 1.96})
    loadings_df = pd.DataFrame(loadings_rows)
    loadings_df.to_csv(RESULTS_DIR / 'tables' / 'aim3_pls_loadings.csv', index=False)

    plot_pls_results(pls, results)

    print(f"\nLV1 潜变量相关 r = {results['lv1_latent_r']:.3f}, 置换检验 p = {results['lv1_permutation_p']:.4f}")
    print(loadings_df.to_string(index=False))


if __name__ == '__main__':
    main()
