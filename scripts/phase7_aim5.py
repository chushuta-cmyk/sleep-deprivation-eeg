#!/usr/bin/env python3
"""
Phase 7 (Refined): Aim 5 - 脑-行为关联、增量效度 (Incremental Validity ΔR²) 与 Bootstrap 中介分析
修复与升级项：
1. 修复原脚本路径缺失 ('v3_conservative') 与未计算行为变化量 (KSS_change, SAI_change) 的 bug。
2. 联合输出三套统计检验：
   - Model 1: 基线校正后的脑电变化与主观困倦 (ΔKSS) / 焦虑反应 (ΔSAI) 回归关联；
   - Model 2: 控制基线睡眠质量 (PSQI) 与基线状态后的增量方差解释度 (ΔR² F-test)；
   - Model 3: 非参数 Bootstrap (2000次) 路径与中介效应检验。
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

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "analysis_config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / config['paths']['features'] / 'v3_conservative'
RESULTS_DIR = PROJECT_ROOT / config['paths']['results']
(RESULTS_DIR / 'tables').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'figures').mkdir(parents=True, exist_ok=True)

STATS = config['statistics']


def load_association_data():
    change_file = FEATURES_DIR / 'eeg_change_features.parquet'
    change_df = pd.read_parquet(change_file)

    participants_file = PROJECT_ROOT / 'data' / 'raw' / 'participants.tsv'
    part_df = pd.read_csv(participants_file, sep='\t')

    for col in ['KSS_NS', 'KSS_SD', 'SAI_NS', 'SAI_SD', 'PSQI_GlobalScore', 'Age']:
        if col in part_df.columns:
            part_df[col] = pd.to_numeric(part_df[col], errors='coerce')

    part_df['KSS_change'] = part_df['KSS_SD'] - part_df['KSS_NS']
    part_df['SAI_change'] = part_df['SAI_SD'] - part_df['SAI_NS']

    df = change_df.merge(part_df, left_on='subject', right_on='participant_id', how='left')
    return df


def run_association_models(df, eeg_features, outcomes):
    records = []
    for outcome, base_col in outcomes:
        for eeg_feat in eeg_features:
            cols = [eeg_feat, outcome, base_col]
            sub = df[cols].dropna().copy()
            if len(sub) < 10:
                continue

            # 标准化以便比较标准化回归系数 β
            sub_z = (sub - sub.mean()) / (sub.std() + 1e-8)
            r_biv, p_biv = stats.pearsonr(sub[eeg_feat], sub[outcome])

            model = smf.ols(f"{outcome} ~ {eeg_feat} + {base_col}", data=sub_z).fit()
            ci = model.conf_int()

            records.append({
                'outcome': outcome,
                'baseline_covariate': base_col,
                'eeg_feature': eeg_feat,
                'n_samples': int(len(sub)),
                'bivariate_r': float(r_biv),
                'bivariate_p': float(p_biv),
                'std_beta_adjusted': float(model.params[eeg_feat]),
                'se_adjusted': float(model.bse[eeg_feat]),
                'p_adjusted': float(model.pvalues[eeg_feat]),
                'ci_lower': float(ci.loc[eeg_feat, 0]),
                'ci_upper': float(ci.loc[eeg_feat, 1]),
                'model_r2': float(model.rsquared),
            })

    assoc_df = pd.DataFrame(records)
    if len(assoc_df) > 0:
        _, qvals, _, _ = multipletests(assoc_df['p_adjusted'], alpha=0.05, method='fdr_bh')
        assoc_df['q_fdr'] = qvals
    return assoc_df


def run_incremental_models(df, eeg_features, outcomes):
    records = []
    for outcome, base_col in outcomes:
        for eeg_feat in eeg_features:
            cols = [eeg_feat, outcome, base_col, 'PSQI_GlobalScore']
            sub = df[cols].dropna().copy()
            if len(sub) < 10:
                continue

            m1 = smf.ols(f"{outcome} ~ {base_col} + PSQI_GlobalScore", data=sub).fit()
            m2 = smf.ols(f"{outcome} ~ {base_col} + PSQI_GlobalScore + {eeg_feat}", data=sub).fit()

            r2_change = float(m2.rsquared - m1.rsquared)
            df_resid = len(sub) - 4
            f_change = (r2_change / 1.0) / ((1.0 - m2.rsquared) / df_resid) if m2.rsquared < 1.0 else 0.0
            p_change = float(1.0 - stats.f.cdf(max(0.0, f_change), 1, df_resid))

            records.append({
                'outcome': outcome,
                'eeg_feature': eeg_feat,
                'n_samples': int(len(sub)),
                'r2_baseline_psqi': float(m1.rsquared),
                'r2_with_eeg': float(m2.rsquared),
                'delta_r2': r2_change,
                'f_change': float(f_change),
                'p_change': p_change,
            })
    return pd.DataFrame(records)


def run_bootstrap_mediation(df, eeg_features, n_boot=2000):
    """
    Bootstrap 路径分析 (X: EEG 变化 -> M: 主观困倦 KSS_change / 基线调节 -> Y: 状态焦虑 SAI_change)
    同时检验反向耦合路径 (X: EEG 变化 -> M: SAI_change -> Y: KSS_change)
    """
    np.random.seed(42)
    records = []

    triads = [
        ('KSS_change', 'SAI_change'),
        ('SAI_change', 'KSS_change'),
    ]

    for eeg_feat in eeg_features:
        for med_col, out_col in triads:
            sub = df[[eeg_feat, med_col, out_col]].dropna().copy()
            if len(sub) < 10:
                continue
            sub_z = (sub - sub.mean()) / (sub.std() + 1e-8)
            x = sub_z[eeg_feat].values
            m = sub_z[med_col].values
            y = sub_z[out_col].values

            # 原始路径系数
            path_a, _, _, p_a, _ = stats.linregress(x, m)
            c_tot, _, _, p_tot, _ = stats.linregress(x, y)
            X_mat = np.column_stack([np.ones(len(x)), x, m])
            betas = np.linalg.lstsq(X_mat, y, rcond=None)[0]
            c_prime, path_b = float(betas[1]), float(betas[2])
            indirect = float(path_a * path_b)

            # Bootstrap 95% CI
            boot_ind = np.zeros(n_boot)
            for b in range(n_boot):
                idx = np.random.choice(len(x), size=len(x), replace=True)
                xb, mb, yb = x[idx], m[idx], y[idx]
                if np.std(xb) < 1e-8 or np.std(mb) < 1e-8:
                    continue
                a_b = np.cov(xb, mb)[0, 1] / (np.var(xb) + 1e-8)
                Xb_mat = np.column_stack([np.ones(len(xb)), xb, mb])
                b_b = np.linalg.lstsq(Xb_mat, yb, rcond=None)[0][2]
                boot_ind[b] = a_b * b_b

            ci_low, ci_high = np.percentile(boot_ind, [2.5, 97.5])
            records.append({
                'eeg_feature': eeg_feat,
                'mediator': med_col,
                'outcome': out_col,
                'n_samples': int(len(sub)),
                'path_a': float(path_a),
                'path_a_p': float(p_a),
                'path_b': float(path_b),
                'direct_c_prime': c_prime,
                'total_effect': float(c_tot),
                'total_effect_p': float(p_tot),
                'indirect_effect': indirect,
                'boot_ci_lower': float(ci_low),
                'boot_ci_upper': float(ci_high),
                'significant_boot_95ci': bool((ci_low > 0) or (ci_high < 0))
            })

    return pd.DataFrame(records)


def plot_aim5_figures(df, eeg_features):
    cols = [c for c in eeg_features + ['KSS_change', 'SAI_change', 'PSQI_GlobalScore'] if c in df.columns]
    corr_df = df[cols].dropna().corr()

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        corr_df, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-0.8, vmax=0.8,
        square=True, linewidths=0.5, ax=ax
    )
    ax.set_title('Aim 5: Cross-Domain Correlation Matrix (EEG Changes & Behavioral Shifts)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figures' / 'aim5_correlation_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()

    # 关键散点图矩阵
    outcomes = ['KSS_change', 'SAI_change']
    top_eeg = eeg_features[:4]
    fig, axes = plt.subplots(len(outcomes), len(top_eeg), figsize=(4.5 * len(top_eeg), 4.2 * len(outcomes)))
    for i, out_c in enumerate(outcomes):
        for j, eeg_c in enumerate(top_eeg):
            ax = axes[i, j]
            sub = df[[eeg_c, out_c]].dropna()
            x, y = sub[eeg_c].values, sub[out_c].values
            ax.scatter(x, y, s=50, alpha=0.75, color='#2980b9', edgecolors='black')
            slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color='#c0392b', linewidth=2)
            ax.set_xlabel(eeg_c.replace('_change', ' Δ'), fontsize=9)
            ax.set_ylabel(out_c, fontsize=9)
            ax.set_title(f"r = {r_val:.2f} (p = {p_val:.3f})", fontsize=10, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figures' / 'aim5_associations.png', dpi=200, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 80)
    print("Phase 7: Aim 5 - 脑-行为关联、增量效度与 Bootstrap 中介分析")
    print("=" * 80)

    df = load_association_data()
    eeg_features = [
        'posterior_alpha_logpower_change',
        'frontal_theta_logpower_change',
        'global_delta_logpower_change',
        'global_spectral_exponent_change',
        'posterior_alpha_periodic_power_change',
        'frontal_theta_periodic_power_change',
        'global_hjorth_complexity_change'
    ]
    eeg_features = [f for f in eeg_features if f in df.columns]
    outcomes = [('KSS_change', 'KSS_NS'), ('SAI_change', 'SAI_NS')]

    assoc_df = run_association_models(df, eeg_features, outcomes)
    assoc_df.to_csv(RESULTS_DIR / 'tables' / 'aim5_model1_results.csv', index=False)
    print("\n--- Model 1: 基线校正后关联结果 ---")
    print(assoc_df[['outcome', 'eeg_feature', 'n_samples', 'bivariate_r', 'std_beta_adjusted', 'p_adjusted']].to_string(index=False))

    inc_df = run_incremental_models(df, eeg_features, outcomes)
    inc_df.to_csv(RESULTS_DIR / 'tables' / 'aim5_incremental_validity.csv', index=False)
    print("\n--- Model 2: 增量方差解释度 (ΔR² beyond Baseline & PSQI) ---")
    print(inc_df[['outcome', 'eeg_feature', 'r2_baseline_psqi', 'r2_with_eeg', 'delta_r2', 'p_change']].to_string(index=False))

    med_df = run_bootstrap_mediation(df, eeg_features[:4], n_boot=2000)
    med_df.to_csv(RESULTS_DIR / 'tables' / 'aim5_mediation_results.csv', index=False)
    print("\n--- Model 3: Bootstrap 路径与中介检验 ---")
    print(med_df[['eeg_feature', 'mediator', 'outcome', 'indirect_effect', 'boot_ci_lower', 'boot_ci_upper']].to_string(index=False))

    plot_aim5_figures(df, eeg_features)


if __name__ == '__main__':
    main()
