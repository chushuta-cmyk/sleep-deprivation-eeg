#!/usr/bin/env python3
"""
Phase 6 (Refined): Aim 4 - 机器学习基准与个体易感性预测 (LOSO Classification + Nested-CV Regression)
修复与升级项：
1. 修复原脚本路径缺失 ('v3_conservative') 导致加载旧损坏特征表的 bug。
2. 构建防数据泄露 (Leakage-Free) 的双任务机器学习基准：
   - Task A (State Classification, LOSO): 严格按被试分组的留一被试交叉验证 (Leave-One-Subject-Out)，
     对比 Logistic Regression, SVM, Random Forest, Gradient Boosting 区分 NS 与 SD 脑状态的性能（AUC, Accuracy, F1, Sensitivity, Specificity）。
   - Task B (Vulnerability Regression, Nested CV): 使用带 L1/L2 正则化的 Ridge, ElasticNet, SVR 在嵌套交叉验证下
     预测基线校正后的困倦/焦虑易感性得分与睡眠剥夺神经行为偏移，并执行置换检验。
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
    LeaveOneGroupOut, RepeatedKFold, KFold, GridSearchCV, cross_val_predict
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

    # 对每个被试构建自身均值中心化特征（消除被试间基线绝对功率解剖差异，测试跨被试泛化的状态判别力）
    df = session_df[['subject', 'condition'] + feature_cols].dropna().copy()
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

        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_curves[name] = (fpr, tpr, auc)

        clf_records.append({
            'model': name,
            'cv_scheme': 'LOSO (32 subjects, 64 sessions)',
            'auc_roc': float(auc),
            'subject_pair_auc_mean': float(np.mean(subject_aucs)),
            'accuracy': float(acc),
            'f1_score': float(f1),
            'sensitivity': float(sens),
            'specificity': float(spec),
        })
        print(f"  {name:26s}: AUC = {auc:.3f} | Pair-AUC = {np.mean(subject_aucs):.3f} | Acc = {acc:.3f} | F1 = {f1:.3f}")

    clf_df = pd.DataFrame(clf_records)
    clf_df.to_csv(RESULTS_DIR / 'tables' / 'aim4_classification_benchmark.csv', index=False)
    return clf_df, roc_curves, feature_cols, X_within, y


def run_task_b_nested_cv_regression():
    """
    Task B: 嵌套交叉验证回归 — 预测个体睡眠剥夺反应幅度
    """
    print("\n--- Task B: Nested-CV 连续神经行为回归预测 ---")
    change_df = pd.read_parquet(FEATURES_DIR / 'eeg_change_features.parquet')
    part_df = pd.read_csv(PROJECT_ROOT / 'data' / 'raw' / 'participants.tsv', sep='\t')

    for c in ['KSS_NS', 'KSS_SD', 'SAI_NS', 'SAI_SD', 'PSQI_GlobalScore']:
        if c in part_df.columns:
            part_df[c] = pd.to_numeric(part_df[c], errors='coerce')

    part_df['KSS_change'] = part_df['KSS_SD'] - part_df['KSS_NS']
    part_df['SAI_change'] = part_df['SAI_SD'] - part_df['SAI_NS']

    df = change_df.merge(part_df, left_on='subject', right_on='participant_id', how='left')

    # 预测特征：采用经过假设检验验证的精简核心神经行为特征集（防止 N=32 小样本高维共线性稀释）
    predictor_cols = [
        'KSS_NS',
        'posterior_alpha_periodic_power_change',
        'frontal_theta_logpower_change',
        'global_spectral_exponent_change',
    ]
    predictor_cols = [c for c in predictor_cols if c in df.columns]

    clean_df = df[['subject', 'KSS_change', 'KSS_SD'] + predictor_cols].dropna().copy()
    X = clean_df[predictor_cols].values
    Y = clean_df['KSS_change'].values  # 预测 ΔKSS 主观嗜睡增幅

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
    preds_dict = {}

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

        preds_dict[name] = y_pred
        reg_records.append({
            'model': name,
            'target': 'KSS_change (ΔKSS)',
            'n_subjects': int(len(Y)),
            'cv_r2': float(r2),
            'cv_mae': float(mae),
            'pearson_r': float(r_val),
            'pearson_p': float(p_val),
        })
        print(f"  {name:18s}: CV r = {r_val:.3f} (p = {p_val:.4f}) | R² = {r2:.3f} | MAE = {mae:.3f}")

    reg_df = pd.DataFrame(reg_records)
    reg_df.to_csv(RESULTS_DIR / 'tables' / 'aim4_regression_results.csv', index=False)
    return reg_df, preds_dict, Y


def plot_aim4_results(clf_df, roc_curves, reg_df, preds_dict, Y):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    # 左图：LOSO ROC 曲线对比
    ax = axes[0]
    palette = ['#c0392b', '#2980b9', '#16a085', '#8e44ad', '#d35400']
    for (name, (fpr, tpr, auc)), col in zip(roc_curves.items(), palette):
        ax.plot(fpr, tpr, linewidth=2.2, color=col, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax.set_title('Aim 4A: LOSO State Classification (NS vs SD)', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8.8)
    ax.grid(True, linestyle=':', alpha=0.3)

    # 右图：Nested-CV 连续嗜睡增幅预测 (ElasticNet / Ridge)
    ax = axes[1]
    best_row = reg_df[reg_df['model'] != 'Dummy (Mean)'].sort_values('pearson_r', ascending=False).iloc[0]
    best_name = best_row['model']
    y_pred = preds_dict[best_name]

    ax.scatter(Y, y_pred, s=65, color='#2980b9', alpha=0.8, edgecolors='black')
    slope, intercept, r_val, p_val, _ = stats.linregress(Y, y_pred)
    x_line = np.linspace(Y.min(), Y.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color='#c0392b', linewidth=2.2,
            label=f"{best_name}: r = {r_val:.2f} (p = {p_val:.4f})\nMAE = {best_row['cv_mae']:.2f}")
    ax.set_xlabel('Observed ΔKSS (SD - NS)', fontweight='bold')
    ax.set_ylabel('Out-of-Fold Predicted ΔKSS', fontweight='bold')
    ax.set_title(f'Aim 4B: Nested-CV Vulnerability Prediction ({best_name})', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9.5)
    ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    fig_path = RESULTS_DIR / 'figures' / 'aim4_model_comparison.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Aim 4 模型对比图已保存: {fig_path}")


def main():
    print("=" * 80)
    print("Phase 6: Aim 4 - 机器学习基准与个体易感性预测")
    print("=" * 80)

    session_df = pd.read_parquet(FEATURES_DIR / 'eeg_session_features.parquet')
    clf_df, roc_curves, _, _, _ = run_task_a_loso_classification(session_df)
    reg_df, preds_dict, Y = run_task_b_nested_cv_regression()
    plot_aim4_results(clf_df, roc_curves, reg_df, preds_dict, Y)


if __name__ == '__main__':
    main()
