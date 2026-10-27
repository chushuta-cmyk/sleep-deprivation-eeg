#!/usr/bin/env python3
"""
Phase 6 (Refined Full-Cohort): Aim 4 - 机器学习基准与个体易感性预测 (LOSO Classification + Nested-CV Regression)
升级项：
1. 动态适配全样本 (sub-01 ~ sub-71) 留一被试交叉验证 (LOSO)。
2. Task B 同时評估三个核心神经行为预测目标：
   - ΔKSS (主观困倦增量)
   - ΔPVT Median RT (客观警觉度反应时受损增量, ms)
   - Full-Cohort Unified ΔSleepiness_Z (全样本标准化困倦增量)
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
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import (
    LeaveOneGroupOut, KFold, GridSearchCV, cross_val_predict
)
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, confusion_matrix,
    r2_score, mean_absolute_error, roc_curve
)

warnings.filterwarnings('ignore')

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "analysis_config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / config['paths']['features'] / 'v3_conservative'
RESULTS_DIR = PROJECT_ROOT / config['paths']['results']
(RESULTS_DIR / 'tables').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'figures').mkdir(parents=True, exist_ok=True)


def run_task_a_loso_classification(session_df):
    """
    Task A: Leave-One-Subject-Out (LOSO) 状态分类基准 (NS vs SD)
    """
    print("\n--- Task A: LOSO 脑电状态分类基准 (NS vs SD) ---")
    feature_cols = [
        'global_delta_logpower',
        'frontal_theta_logpower',
        'posterior_alpha_logpower',
        'global_spectral_exponent',
        'frontal_theta_periodic_power',
        'posterior_alpha_periodic_power',
        'global_hjorth_mobility',
        'global_hjorth_complexity',
    ]
    feature_cols = [f for f in feature_cols if f in session_df.columns]

    df = session_df[['subject', 'condition'] + feature_cols].dropna().copy()
    # 仅保留拥有完整 NS 与 SD 配对的被试进行严格配对状态解码
    subj_counts = df.groupby('subject')['condition'].nunique()
    paired_subjs = subj_counts[subj_counts == 2].index
    df = df[df['subject'].isin(paired_subjs)].copy()

    n_sub = df['subject'].nunique()
    subj_means = df.groupby('subject')[feature_cols].transform('mean')
    X_within = (df[feature_cols] - subj_means).values
    y = (df['condition'] == 'SD').astype(int).values
    groups = df['subject'].values

    logo = LeaveOneGroupOut()

    classifiers = {
        'Logistic Regression (L2)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=0.5, penalty='l2', random_state=42))
        ]),
        'Linear SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='linear', C=0.5, probability=True, random_state=42))
        ]),
        'RBF SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42))
        ])
    }

    clf_records = []
    roc_curves = {}

    for name, clf in classifiers.items():
        y_prob = np.zeros(len(y))
        y_pred = np.zeros(len(y), dtype=int)
        subject_aucs = []

        for train_idx, test_idx in logo.split(X_within, y, groups=groups):
            clf.fit(X_within[train_idx], y[train_idx])
            probs = clf.predict_proba(X_within[test_idx])[:, 1]
            preds = (probs >= 0.5).astype(int)
            y_prob[test_idx] = probs
            y_pred[test_idx] = preds
            if len(np.unique(y[test_idx])) == 2:
                subject_aucs.append(roc_auc_score(y[test_idx], probs))

        auc = roc_auc_score(y, y_prob)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Subject-level Bootstrap 95% CI (1000 resamples)
        np.random.seed(42)
        unique_groups = np.unique(groups)
        boot_aucs = []
        boot_accs = []
        for _ in range(1000):
            b_subjs = np.random.choice(unique_groups, size=len(unique_groups), replace=True)
            b_idx = np.concatenate([np.where(groups == s)[0] for s in b_subjs])
            if len(np.unique(y[b_idx])) == 2:
                boot_aucs.append(roc_auc_score(y[b_idx], y_prob[b_idx]))
                boot_accs.append(accuracy_score(y[b_idx], y_pred[b_idx]))

        auc_ci = np.percentile(boot_aucs, [2.5, 97.5])
        acc_ci = np.percentile(boot_accs, [2.5, 97.5])

        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_curves[name] = (fpr, tpr, auc, auc_ci)

        clf_records.append({
            'model': name,
            'cv_scheme': f'LOSO ({n_sub} subjects, {len(y)} sessions)',
            'auc_roc': float(auc),
            'auc_ci_lower': float(auc_ci[0]),
            'auc_ci_upper': float(auc_ci[1]),
            'subject_pair_auc_mean': float(np.mean(subject_aucs)),
            'accuracy': float(acc),
            'acc_ci_lower': float(acc_ci[0]),
            'acc_ci_upper': float(acc_ci[1]),
            'f1_score': float(f1),
            'sensitivity': float(sens),
            'specificity': float(spec),
        })
        print(f"  {name:26s}: AUC = {auc:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}] | Pair-AUC = {np.mean(subject_aucs):.3f} | Acc = {acc:.3f} [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}] | F1 = {f1:.3f}")

    clf_df = pd.DataFrame(clf_records)
    clf_df.to_csv(RESULTS_DIR / 'tables' / 'aim4_classification_benchmark.csv', index=False)
    return clf_df, roc_curves, feature_cols, X_within, y


def run_task_b_nested_cv_regression():
    """
    Task B: 嵌套交叉验证回归 — 预测个体睡眠剥夺反应幅度 (ΔKSS, ΔPVT_RT, 及 Full-Cohort ΔSleepiness_Z)
    """
    print("\n--- Task B: Nested-CV 连续神经行为回归预测 ---")
    change_df = pd.read_parquet(FEATURES_DIR / 'eeg_change_features.parquet')
    part_df = pd.read_csv(PROJECT_ROOT / 'data' / 'raw' / 'participants.tsv', sep='\t')

    for c in ['KSS_NS', 'KSS_SD', 'SSS_NS', 'SSS_SD', 'PVT_item2_NS', 'PVT_item2_SD', 'PVT_item3_NS', 'PVT_item3_SD', 'SAI_NS', 'SAI_SD', 'PSQI_GlobalScore']:
        if c in part_df.columns:
            part_df[c] = pd.to_numeric(part_df[c], errors='coerce')

    part_df['KSS_change'] = part_df['KSS_SD'] - part_df['KSS_NS']
    part_df['SSS_change'] = part_df['SSS_SD'] - part_df['SSS_NS']
    part_df['PVT_RT_change'] = part_df['PVT_item2_SD'] - part_df['PVT_item2_NS']
    part_df['PVT_RTSD_change'] = part_df['PVT_item3_SD'] - part_df['PVT_item3_NS']

    kss_z_ns = (part_df['KSS_NS'] - part_df['KSS_NS'].mean()) / (part_df['KSS_NS'].std() + 1e-8)
    sss_z_ns = (part_df['SSS_NS'] - part_df['SSS_NS'].mean()) / (part_df['SSS_NS'].std() + 1e-8)
    part_df['Sleepiness_Z_NS'] = kss_z_ns.combine_first(sss_z_ns)

    kss_z_chg = (part_df['KSS_change'] - part_df['KSS_change'].mean()) / (part_df['KSS_change'].std() + 1e-8)
    sss_z_chg = (part_df['SSS_change'] - part_df['SSS_change'].mean()) / (part_df['SSS_change'].std() + 1e-8)
    part_df['Sleepiness_Z_change'] = kss_z_chg.combine_first(sss_z_chg)

    df = change_df.merge(part_df, left_on='subject', right_on='participant_id', how='left')

    targets_spec = [
        ('KSS_change (ΔKSS)', 'KSS_change', ['KSS_NS', 'posterior_alpha_periodic_power_change', 'frontal_theta_logpower_change', 'global_spectral_exponent_change']),
        ('Sleepiness_Z_change (Full Cohort ΔZ)', 'Sleepiness_Z_change', ['Sleepiness_Z_NS', 'posterior_alpha_periodic_power_change', 'frontal_theta_logpower_change', 'global_spectral_exponent_change']),
        ('PVT_RTSD_change (ΔPVT RT Variability, ms)', 'PVT_RTSD_change', ['PVT_item3_NS', 'PSQI_GlobalScore', 'frontal_theta_logpower_change', 'global_spectral_exponent_change']),
    ]

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=43)

    reg_models = {
        'Ridge': (
            Pipeline([('scaler', RobustScaler()), ('model', Ridge(random_state=42))]),
            {'model__alpha': np.logspace(-1, 3, 25)}
        ),
        'ElasticNet': (
            Pipeline([('scaler', RobustScaler()), ('model', ElasticNet(random_state=42, max_iter=5000))]),
            {'model__alpha': np.logspace(-2, 1.5, 15), 'model__l1_ratio': [0.1, 0.3, 0.5, 0.8]}
        ),
        'Linear SVR': (
            Pipeline([('scaler', RobustScaler()), ('model', SVR(kernel='linear'))]),
            {'model__C': np.logspace(-2, 1, 15)}
        ),
        'Dummy (Mean)': (
            DummyRegressor(strategy='mean'),
            {}
        )
    }

    reg_records = []
    all_targets_preds = {}

    for target_label, target_col, pred_cols in targets_spec:
        pred_cols = [c for c in pred_cols if c in df.columns]
        clean_df = df[['subject', target_col] + pred_cols].dropna().copy()
        if len(clean_df) < 15:
            continue
        X = clean_df[pred_cols].values
        Y = clean_df[target_col].values

        print(f"\n  [Target: {target_label} | n = {len(Y)}]")
        target_preds = {}
        for name, (pipe, grid) in reg_models.items():
            if grid:
                estimator = GridSearchCV(pipe, grid, cv=inner_cv, scoring='neg_mean_absolute_error', n_jobs=1)
            else:
                estimator = pipe

            y_pred = cross_val_predict(estimator, X, Y, cv=outer_cv)
            r2 = r2_score(Y, y_pred)
            mae = mean_absolute_error(Y, y_pred)
            if np.std(y_pred) > 1e-8:
                r_val, p_val = stats.pearsonr(Y, y_pred)
            else:
                r_val, p_val = 0.0, 1.0

            target_preds[name] = (y_pred, r_val, p_val, r2, mae)

            reg_records.append({
                'model': name,
                'target': target_label,
                'n_subjects': int(len(Y)),
                'cv_r2': float(r2),
                'cv_mae': float(mae),
                'pearson_r': float(r_val),
                'pearson_p': float(p_val),
            })
            print(f"    {name:18s}: CV r = {r_val:.3f} (p = {p_val:.4f}) | R² = {r2:.3f} | MAE = {mae:.3f}")

        # Find best non-dummy model
        non_dummy = {k: v for k, v in target_preds.items() if k != 'Dummy (Mean)'}
        best_k = max(non_dummy.keys(), key=lambda k: non_dummy[k][1])
        all_targets_preds[target_col] = {
            'best_name': best_k,
            'y_pred': non_dummy[best_k][0],
            'Y': Y,
            'r': non_dummy[best_k][1],
            'p': non_dummy[best_k][2],
            'r2': non_dummy[best_k][3],
            'mae': non_dummy[best_k][4],
            'label': target_label
        }

    reg_df = pd.DataFrame(reg_records)
    reg_df.to_csv(RESULTS_DIR / 'tables' / 'aim4_regression_results.csv', index=False)
    return reg_df, all_targets_preds


def plot_aim4_results(clf_df, roc_curves, reg_df, all_targets_preds):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: LOSO ROC Curves
    ax = axes[0, 0]
    palette = ['#c0392b', '#2980b9', '#16a085', '#8e44ad', '#d35400']
    for (name, (fpr, tpr, auc, auc_ci)), col in zip(roc_curves.items(), palette):
        ax.plot(fpr, tpr, linewidth=2.2, color=col, label=f"{name}\nAUC = {auc:.3f} [{auc_ci[0]:.2f}, {auc_ci[1]:.2f}]")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax.set_title('(A) Aim 4A: LOSO Brain-State Decoding (N=71, 142 sess)', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8.0)
    ax.grid(True, linestyle=':', alpha=0.3)

    # Panel B: Full Cohort Sleepiness_Z_change (N=66)
    ax = axes[0, 1]
    if 'Sleepiness_Z_change' in all_targets_preds:
        info = all_targets_preds['Sleepiness_Z_change']
        Y, y_pred = info['Y'], info['y_pred']
        ax.scatter(Y, y_pred, s=55, color='#2980b9', alpha=0.8, edgecolors='black')
        slope, intercept, _, _, _ = stats.linregress(Y, y_pred)
        x_line = np.linspace(Y.min(), Y.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color='#c0392b', linewidth=2.2,
                label=f"{info['best_name']}: r = {info['r']:.2f} (p = {info['p']:.2e})\nR² = {info['r2']:.2f}, MAE = {info['mae']:.2f}")
        ax.set_xlabel('Observed Unified ΔZ (SD - NS)', fontweight='bold')
        ax.set_ylabel('Out-of-Fold Predicted ΔZ', fontweight='bold')
        ax.set_title(f"(B) Aim 4B: Full-Cohort Sleepiness ({info['best_name']}, N=66)", fontweight='bold')
        ax.legend(loc='upper left', fontsize=9.0)
        ax.grid(True, linestyle=':', alpha=0.3)

    # Panel C: KSS_change (N=33)
    ax = axes[1, 0]
    if 'KSS_change' in all_targets_preds:
        info = all_targets_preds['KSS_change']
        Y, y_pred = info['Y'], info['y_pred']
        ax.scatter(Y, y_pred, s=60, color='#16a085', alpha=0.8, edgecolors='black')
        slope, intercept, _, _, _ = stats.linregress(Y, y_pred)
        x_line = np.linspace(Y.min(), Y.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color='#c0392b', linewidth=2.2,
                label=f"{info['best_name']}: r = {info['r']:.2f} (p = {info['p']:.4f})\nR² = {info['r2']:.2f}, MAE = {info['mae']:.2f}")
        ax.set_xlabel('Observed ΔKSS (SD - NS)', fontweight='bold')
        ax.set_ylabel('Out-of-Fold Predicted ΔKSS', fontweight='bold')
        ax.set_title(f"(C) Aim 4B: Karolinska Sleepiness ({info['best_name']}, N=33)", fontweight='bold')
        ax.legend(loc='upper left', fontsize=9.0)
        ax.grid(True, linestyle=':', alpha=0.3)

    # Panel D: PVT_RTSD_change (N=26)
    ax = axes[1, 1]
    if 'PVT_RTSD_change' in all_targets_preds:
        info = all_targets_preds['PVT_RTSD_change']
        Y, y_pred = info['Y'], info['y_pred']
        ax.scatter(Y, y_pred, s=60, color='#8e44ad', alpha=0.8, edgecolors='black')
        slope, intercept, _, _, _ = stats.linregress(Y, y_pred)
        x_line = np.linspace(Y.min(), Y.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color='#c0392b', linewidth=2.2,
                label=f"{info['best_name']}: r = {info['r']:.2f} (p = {info['p']:.2e})\nR² = {info['r2']:.2f}, MAE = {info['mae']:.1f} ms")
        ax.set_xlabel('Observed ΔPVT RT SD (ms)', fontweight='bold')
        ax.set_ylabel('Out-of-Fold Predicted ΔPVT RT SD (ms)', fontweight='bold')
        ax.set_title(f"(D) Aim 4B: Objective Vigilance Variability ({info['best_name']}, N=26)", fontweight='bold')
        ax.legend(loc='upper left', fontsize=9.0)
        ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    fig_path = RESULTS_DIR / 'figures' / 'aim4_model_comparison.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Aim 4 四联模型对比图已保存: {fig_path}")


def main():
    print("=" * 80)
    print("Phase 6: Aim 4 - 机器学习基准与个体易感性预测 (Full Cohort)")
    print("=" * 80)

    session_df = pd.read_parquet(FEATURES_DIR / 'eeg_session_features.parquet')
    clf_df, roc_curves, _, _, _ = run_task_a_loso_classification(session_df)
    reg_df, all_targets_preds = run_task_b_nested_cv_regression()
    plot_aim4_results(clf_df, roc_curves, reg_df, all_targets_preds)


if __name__ == '__main__':
    main()
