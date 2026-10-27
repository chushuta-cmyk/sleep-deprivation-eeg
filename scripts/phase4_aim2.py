#!/usr/bin/env python3
"""
Phase 4 (Refined): Aim 2 - 跨条件个体排序稳定性 (Trait-like Stability vs State Sensitivity)
修复项：
1. 修复原脚本中 pingouin Type=='ICC(A,1)' 名称不匹配导致返回空的 bug（pingouin 标准名称为 'ICC2' 绝对一致度 与 'ICC3' 一致性）。
2. 同时报告 ICC(2,1) [Absolute Agreement]、ICC(3,1) [Consistency]、Pearson r 与 Spearman ρ，全面刻画特质稳定性。
"""

import os
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import pingouin as pg
from scipy import stats

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "analysis_config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

FEATURES_DIR = PROJECT_ROOT / config['paths']['features'] / 'v3_conservative'
RESULTS_DIR = PROJECT_ROOT / config['paths']['results']
(RESULTS_DIR / 'tables').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'figures').mkdir(parents=True, exist_ok=True)


def load_session_features():
    session_file = FEATURES_DIR / 'eeg_session_features.parquet'
    return pd.read_parquet(session_file)


def compute_icc(session_df, feature_col):
    df = session_df[['subject', 'condition', feature_col]].dropna().copy()
    pivot_df = df.pivot(index='subject', columns='condition', values=feature_col).dropna()
    if len(pivot_df) < 5:
        return None

    valid_subjects = pivot_df.index.tolist()
    df = df[df['subject'].isin(valid_subjects)]

    icc_res = pg.intraclass_corr(data=df, targets='subject', raters='condition', ratings=feature_col)
    icc2_row = icc_res[icc_res['Type'] == 'ICC2'].iloc[0]  # ICC(A,1) Absolute agreement
    icc3_row = icc_res[icc_res['Type'] == 'ICC3'].iloc[0]  # ICC(C,1) Consistency

    ns_data = pivot_df['NS'].values
    sd_data = pivot_df['SD'].values
    r_pearson, p_pearson = stats.pearsonr(ns_data, sd_data)
    r_spearman, p_spearman = stats.spearmanr(ns_data, sd_data)

    return {
        'feature': feature_col,
        'n_subjects': int(len(pivot_df)),
        'icc_agreement': float(icc2_row['ICC']),
        'icc_ci_lower': float(icc2_row['CI95%'][0]),
        'icc_ci_upper': float(icc2_row['CI95%'][1]),
        'icc_p': float(icc2_row['pval']),
        'icc_consistency': float(icc3_row['ICC']),
        'icc_c_ci_lower': float(icc3_row['CI95%'][0]),
        'icc_c_ci_upper': float(icc3_row['CI95%'][1]),
        'pearson_r': float(r_pearson),
        'pearson_p': float(p_pearson),
        'spearman_r': float(r_spearman),
        'spearman_p': float(p_spearman),
        'mean_ns': float(np.mean(ns_data)),
        'mean_sd': float(np.mean(sd_data)),
    }


def icc2_fast(Y):
    """
    Analytical two-way random-effects absolute agreement single-measure ICC(2,1) / ICC(A,1).
    Mathematically identical to pingouin ICC2 on balanced (n_subjects, k_conditions) matrix Y.
    """
    n, k = Y.shape
    grand_mean = np.mean(Y)
    row_means = np.mean(Y, axis=1)
    col_means = np.mean(Y, axis=0)
    SSR = k * np.sum((row_means - grand_mean) ** 2)
    SSC = n * np.sum((col_means - grand_mean) ** 2)
    SST = np.sum((Y - grand_mean) ** 2)
    SSE = SST - SSR - SSC
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    return float((MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n))


def compute_delta_icc_bootstrap(session_df, feat_periodic, feat_raw, n_boot=1000, seed=42):
    """
    Cluster-bootstrap test (resampling subjects with replacement) for the difference in
    cross-condition trait stability ΔICC(A,1) = ICC(periodic) - ICC(unadjusted).
    """
    piv_per = session_df.pivot(index='subject', columns='condition', values=feat_periodic).dropna()
    piv_raw = session_df.pivot(index='subject', columns='condition', values=feat_raw).dropna()
    common_subjs = piv_per.index.intersection(piv_raw.index)
    Y_per = piv_per.loc[common_subjs, ['NS', 'SD']].values
    Y_raw = piv_raw.loc[common_subjs, ['NS', 'SD']].values
    n = len(common_subjs)

    icc_per = icc2_fast(Y_per)
    icc_raw = icc2_fast(Y_raw)
    obs_delta = icc_per - icc_raw

    rng = np.random.RandomState(seed)
    boot_deltas = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_deltas[b] = icc2_fast(Y_per[idx]) - icc2_fast(Y_raw[idx])

    ci_low, ci_high = np.percentile(boot_deltas, [2.5, 97.5])
    p_one_sided = float(np.mean(boot_deltas <= 0) if obs_delta > 0 else np.mean(boot_deltas >= 0))
    p_two_sided = float(max(1.0 / n_boot, 2.0 * min(np.mean(boot_deltas <= 0), np.mean(boot_deltas >= 0))))

    return {
        'periodic_feature': feat_periodic,
        'unadjusted_feature': feat_raw,
        'n_subjects': int(n),
        'icc_periodic': float(icc_per),
        'icc_unadjusted': float(icc_raw),
        'delta_icc': float(obs_delta),
        'ci95_lower': float(ci_low),
        'ci95_upper': float(ci_high),
        'p_boot_one_sided': p_one_sided,
        'p_boot_two_sided': p_two_sided,
    }


def plot_cross_condition_stability(session_df, features_to_plot, icc_df):
    n_features = len(features_to_plot)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5.0 * n_rows))
    axes = np.array(axes).flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]
        pivot_df = session_df.pivot(index='subject', columns='condition', values=feature).dropna()
        ns_data = pivot_df['NS'].values
        sd_data = pivot_df['SD'].values

        ax.scatter(ns_data, sd_data, alpha=0.75, s=55, color='#2980b9', edgecolors='black', linewidth=0.6)
        min_val = min(ns_data.min(), sd_data.min())
        max_val = max(ns_data.max(), sd_data.max())
        pad = 0.05 * (max_val - min_val + 1e-6)
        ax.plot([min_val - pad, max_val + pad], [min_val - pad, max_val + pad], 'k--', alpha=0.5, label='Identity (y=x)')

        slope, intercept, r_val, _, _ = stats.linregress(ns_data, sd_data)
        x_line = np.linspace(min_val, max_val, 100)
        ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2, label=f'Fit (r={r_val:.2f})')

        row = icc_df[icc_df['feature'] == feature].iloc[0]
        ax.text(
            0.05, 0.95,
            f"n={int(row['n_subjects'])}\n"
            f"ICC(A,1)={row['icc_agreement']:.2f} [{row['icc_ci_lower']:.2f}, {row['icc_ci_upper']:.2f}]\n"
            f"ICC(C,1)={row['icc_consistency']:.2f} | r={row['pearson_r']:.2f} (p={row['pearson_p']:.4f})",
            ha='left', va='top', transform=ax.transAxes, fontsize=8.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fcf3cf', alpha=0.85)
        )
        ax.set_xlabel(f'{feature} (NS)', fontweight='bold')
        ax.set_ylabel(f'{feature} (SD)', fontweight='bold')
        ax.set_title(feature.replace('_', ' ').title(), fontweight='bold')
        ax.legend(loc='lower right', fontsize=8.5)
        ax.grid(True, linestyle=':', alpha=0.3)

    for idx in range(len(features_to_plot), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig_path = RESULTS_DIR / 'figures' / 'aim2_cross_condition_stability.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ 跨条件稳定性图已保存: {fig_path}")


def plot_bland_altman(session_df, features_to_plot):
    n_features = len(features_to_plot)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.6 * n_rows))
    axes = np.array(axes).flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]
        pivot_df = session_df.pivot(index='subject', columns='condition', values=feature).dropna()
        ns_data = pivot_df['NS'].values
        sd_data = pivot_df['SD'].values

        mean_vals = (ns_data + sd_data) / 2.0
        diff_vals = sd_data - ns_data
        mean_diff = np.mean(diff_vals)
        sd_diff = np.std(diff_vals, ddof=1)

        ax.scatter(mean_vals, diff_vals, alpha=0.75, s=50, color='#16a085', edgecolors='black', linewidth=0.6)
        ax.axhline(mean_diff, color='#2980b9', linestyle='-', linewidth=2, label=f'Mean Δ: {mean_diff:.3f}')
        ax.axhline(mean_diff + 1.96 * sd_diff, color='#c0392b', linestyle='--', label=f'+1.96 SD: {mean_diff + 1.96 * sd_diff:.3f}')
        ax.axhline(mean_diff - 1.96 * sd_diff, color='#c0392b', linestyle='--', label=f'-1.96 SD: {mean_diff - 1.96 * sd_diff:.3f}')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
        ax.set_xlabel('Mean of NS and SD', fontweight='bold')
        ax.set_ylabel('Difference (SD - NS)', fontweight='bold')
        ax.set_title(f'{feature} Bland-Altman', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.3)

    for idx in range(len(features_to_plot), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig_path = RESULTS_DIR / 'figures' / 'aim2_bland_altman_qc.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Bland-Altman QC 图已保存: {fig_path}")


def main():
    print("=" * 80)
    print("Phase 4: Aim 2 - 跨条件个体差异稳定性 (ICC & Rank Correlation)")
    print("=" * 80)

    session_df = load_session_features()
    primary_features = config['analysis']['primary_features']
    extra_features = [
        'posterior_alpha_periodic_power',
        'frontal_theta_periodic_power',
        'global_theta_logpower',
        'global_alpha_logpower',
        'global_beta_logpower',
        'global_hjorth_mobility',
        'global_hjorth_complexity',
    ]
    all_features = [f for f in (primary_features + extra_features) if f in session_df.columns]

    icc_records = []
    for feat in all_features:
        res = compute_icc(session_df, feat)
        if res:
            icc_records.append(res)

    icc_df = pd.DataFrame(icc_records)
    results_file = RESULTS_DIR / 'tables' / 'aim2_icc_results.csv'
    icc_df.to_csv(results_file, index=False)
    print(f"\n✓ ICC 统计表已保存: {results_file}")
    print(icc_df[['feature', 'n_subjects', 'icc_agreement', 'icc_consistency', 'pearson_r', 'pearson_p']].to_string(index=False))

    # Cluster-bootstrap test for ΔICC (periodic vs unadjusted log-power)
    delta_icc_pairs = [
        ('frontal_theta_periodic_power', 'frontal_theta_logpower'),
        ('posterior_alpha_periodic_power', 'posterior_alpha_logpower'),
    ]
    delta_records = []
    print("\n--- ΔICC(A,1) Cluster-Bootstrap 检验 (Periodic vs. Unadjusted, 1,000 resamples) ---")
    for f_per, f_raw in delta_icc_pairs:
        if f_per in session_df.columns and f_raw in session_df.columns:
            d_res = compute_delta_icc_bootstrap(session_df, f_per, f_raw, n_boot=1000, seed=42)
            delta_records.append(d_res)
            print(
                f"  - {f_per} vs. {f_raw} (N={d_res['n_subjects']}): "
                f"ICC_per={d_res['icc_periodic']:.3f}, ICC_raw={d_res['icc_unadjusted']:.3f} -> "
                f"ΔICC = {d_res['delta_icc']:+.3f} (95% CI [{d_res['ci95_lower']:+.3f}, {d_res['ci95_upper']:+.3f}], "
                f"p_one_sided={d_res['p_boot_one_sided']:.4f}, p_two_sided={d_res['p_boot_two_sided']:.4f})"
            )

    delta_df = pd.DataFrame(delta_records)
    delta_file = RESULTS_DIR / 'tables' / 'aim2_delta_icc_bootstrap.csv'
    delta_df.to_csv(delta_file, index=False)
    print(f"✓ ΔICC Bootstrap 统计表已保存: {delta_file}")

    plot_feats = [f for f in primary_features if f in session_df.columns]
    plot_cross_condition_stability(session_df, plot_feats, icc_df)
    plot_bland_altman(session_df, plot_feats)


if __name__ == '__main__':
    main()
