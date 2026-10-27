#!/usr/bin/env python3
"""
Phase 7 (Refined Full-Cohort): Aim 5 - 脑-行为关联、增量效度 (Incremental Validity ΔR²) 与 Bootstrap 中介分析
升级项：
1. 全面涵盖 ds004902 全样本 (sub-01 ~ sub-71) 的多维神经行为结局：
   - Sleepiness_Z_change (全样本统一主观困倦标准化增幅)
   - KSS_change (Karolinska 主观嗜睡量表增幅)
   - SSS_change (Stanford 主观嗜睡量表增幅)
   - PVT_RT_change (客观警觉度任务中位反应时增幅, ms)
   - PVT_RTSD_change (客观警觉度任务反应时变异性增幅, ms)
   - SAI_change (状态焦虑量表增幅)
   - PANAS_P_change_clean / PANAS_N_change_clean (无元数据污染的正负性情绪变化)
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

    num_cols = [
        'KSS_NS', 'KSS_SD', 'SSS_NS', 'SSS_SD',
        'SAI_NS', 'SAI_SD',
        'PVT_item1_NS', 'PVT_item1_SD',
        'PVT_item2_NS', 'PVT_item2_SD',
        'PVT_item3_NS', 'PVT_item3_SD',
        'PANAS_P_NS', 'PANAS_P_SD', 'PANAS_N_NS', 'PANAS_N_SD',
        'PSQI_GlobalScore', 'Age'
    ]
    for col in num_cols:
        if col in part_df.columns:
            part_df[col] = pd.to_numeric(part_df[col], errors='coerce')

    part_df['KSS_change'] = part_df['KSS_SD'] - part_df['KSS_NS']
    part_df['SSS_change'] = part_df['SSS_SD'] - part_df['SSS_NS']
    part_df['SAI_change'] = part_df['SAI_SD'] - part_df['SAI_NS']
    part_df['PVT_Lapses_change'] = part_df['PVT_item1_SD'] - part_df['PVT_item1_NS']
    part_df['PVT_RT_change'] = part_df['PVT_item2_SD'] - part_df['PVT_item2_NS']
    part_df['PVT_RTSD_change'] = part_df['PVT_item3_SD'] - part_df['PVT_item3_NS']

    kss_z_ns = (part_df['KSS_NS'] - part_df['KSS_NS'].mean()) / (part_df['KSS_NS'].std() + 1e-8)
    sss_z_ns = (part_df['SSS_NS'] - part_df['SSS_NS'].mean()) / (part_df['SSS_NS'].std() + 1e-8)
    part_df['Sleepiness_Z_NS'] = kss_z_ns.combine_first(sss_z_ns)

    kss_z_chg = (part_df['KSS_change'] - part_df['KSS_change'].mean()) / (part_df['KSS_change'].std() + 1e-8)
    sss_z_chg = (part_df['SSS_change'] - part_df['SSS_change'].mean()) / (part_df['SSS_change'].std() + 1e-8)
    part_df['Sleepiness_Z_change'] = kss_z_chg.combine_first(sss_z_chg)

    valid_panas_mask = part_df['SAI_NS'].isna()
    part_df['PANAS_P_change_clean'] = np.where(valid_panas_mask, part_df['PANAS_P_SD'] - part_df['PANAS_P_NS'], np.nan)
    part_df['PANAS_N_change_clean'] = np.where(valid_panas_mask, part_df['PANAS_N_SD'] - part_df['PANAS_N_NS'], np.nan)

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
        assoc_df['q_fdr'] = np.nan
        for out_name, grp in assoc_df.groupby('outcome'):
            _, qvals, _, _ = multipletests(grp['p_adjusted'], alpha=0.05, method='fdr_bh')
            assoc_df.loc[grp.index, 'q_fdr'] = qvals
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
    np.random.seed(42)
    records = []

    triads = [
        ('KSS_change', 'SAI_change'),
        ('SAI_change', 'KSS_change'),
        ('SSS_change', 'PVT_RT_change'),
        ('PVT_RT_change', 'SSS_change'),
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

            path_a, _, _, p_a, _ = stats.linregress(x, m)
            c_tot, _, _, p_tot, _ = stats.linregress(x, y)
            X_mat = np.column_stack([np.ones(len(x)), x, m])
            betas = np.linalg.lstsq(X_mat, y, rcond=None)[0]
            c_prime, path_b = float(betas[1]), float(betas[2])
            indirect = float(path_a * path_b)

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
    cols = [c for c in eeg_features + ['Sleepiness_Z_change', 'KSS_change', 'PVT_RT_change', 'SAI_change', 'PSQI_GlobalScore'] if c in df.columns]
    corr_df = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(
        corr_df, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-0.8, vmax=0.8,
        square=True, linewidths=0.5, ax=ax
    )
    ax.set_title('Aim 5: Cross-Domain Correlation Matrix (Pairwise Available N)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figures' / 'aim5_correlation_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()

    outcomes = ['KSS_change', 'PVT_RT_change', 'Sleepiness_Z_change']
    outcomes = [o for o in outcomes if o in df.columns and df[o].notna().sum() >= 10]
    top_eeg = eeg_features[:4]
    fig, axes = plt.subplots(len(outcomes), len(top_eeg), figsize=(4.5 * len(top_eeg), 4.0 * len(outcomes)))
    if len(outcomes) == 1:
        axes = np.expand_dims(axes, axis=0)
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
            ax.set_title(f"r = {r_val:.2f} (p = {p_val:.3f}, n={len(sub)})", fontsize=9.5, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figures' / 'aim5_associations.png', dpi=200, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 80)
    print("Phase 7: Aim 5 - 脑-行为关联、增量效度与 Bootstrap 中介分析 (Full Cohort)")
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
    outcomes = [
        ('Sleepiness_Z_change', 'Sleepiness_Z_NS'),
        ('KSS_change', 'KSS_NS'),
        ('SSS_change', 'SSS_NS'),
        ('PVT_RT_change', 'PVT_item2_NS'),
        ('PVT_RTSD_change', 'PVT_item3_NS'),
        ('SAI_change', 'SAI_NS'),
        ('PANAS_P_change_clean', 'PANAS_P_NS'),
        ('PANAS_N_change_clean', 'PANAS_N_NS'),
    ]

    assoc_df = run_association_models(df, eeg_features, outcomes)
    assoc_df.to_csv(RESULTS_DIR / 'tables' / 'aim5_model1_results.csv', index=False)
    print("\n--- Model 1: 基线校正后关联结果 (Top Associations) ---")
    print(assoc_df.sort_values('p_adjusted').head(15)[['outcome', 'eeg_feature', 'n_samples', 'bivariate_r', 'std_beta_adjusted', 'p_adjusted']].to_string(index=False))

    inc_df = run_incremental_models(df, eeg_features, outcomes)
    inc_df.to_csv(RESULTS_DIR / 'tables' / 'aim5_incremental_validity.csv', index=False)
    print("\n--- Model 2: 增量方差解释度 (ΔR² beyond Baseline & PSQI) ---")
    print(inc_df.sort_values('p_change').head(15)[['outcome', 'eeg_feature', 'n_samples', 'r2_baseline_psqi', 'r2_with_eeg', 'delta_r2', 'p_change']].to_string(index=False))

    med_df = run_bootstrap_mediation(df, eeg_features[:5], n_boot=2000)
    med_df.to_csv(RESULTS_DIR / 'tables' / 'aim5_mediation_results.csv', index=False)
    print("\n--- Model 3: Bootstrap 路径与中介检验 ---")
    print(med_df[['eeg_feature', 'mediator', 'outcome', 'n_samples', 'indirect_effect', 'boot_ci_lower', 'boot_ci_upper']].to_string(index=False))

    plot_aim5_figures(df, eeg_features)


if __name__ == '__main__':
    main()
