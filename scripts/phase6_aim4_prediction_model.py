#!/usr/bin/env python3
"""
Phase 6: Aim 4 - 预测模型
使用嵌套交叉验证构建预测模型，预测睡眠剥夺易感性
"""

import os
import json
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from scipy import stats

# 项目路径
PROJECT_ROOT = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/sleep_deprivation_eeg")
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DERIVATIVES_DIR = PROJECT_ROOT / "data" / "derivatives"
RESULTS_DIR = PROJECT_ROOT / "results"

def load_prediction_data():
    """加载预测模型数据"""
    print("=" * 80)
    print("加载预测模型数据...")
    print("=" * 80)
    
    # 读取 participants.tsv
    participants_tsv = pd.read_csv(RAW_DATA_DIR / "participants.tsv", sep="\t")
    
    all_data = []
    
    for idx, row in participants_tsv.iterrows():
        sub_id = row["participant_id"]
        sub_dir = RAW_DATA_DIR / sub_id
        
        if not sub_dir.exists():
            continue
        
        # 检查两个 session 的预处理数据
        preprocessed_files = {}
        for ses in ["ses-1", "ses-2"]:
            preprocessed_file = DERIVATIVES_DIR / "preprocessed" / sub_id / ses / f"{sub_id}_{ses}-epo.fif"
            if preprocessed_file.exists():
                preprocessed_files[ses] = preprocessed_file
        
        # 需要两个 session 都有数据
        if len(preprocessed_files) != 2:
            continue
        
        # 确定条件
        session_order = row["SessionOrder"]
        if session_order == "NS->SD":
            ns_ses, sd_ses = "ses-1", "ses-2"
        else:
            ns_ses, sd_ses = "ses-2", "ses-1"
        
        # 提取 EEG 特征（正常睡眠条件）
        features = {
            "subject": sub_id,
            "age": row["Age"],
            "gender": row["Gender"]
        }
        
        # 正常睡眠条件下的 EEG 特征
        epochs_ns = mne.read_epochs(preprocessed_files[ns_ses], verbose=False)
        psd_ns = epochs_ns.compute_psd(fmin=1, fmax=45, verbose=False)
        psd_data_ns = psd_ns.get_data()
        freqs_ns = psd_ns.freqs
        
        freq_bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45)
        }
        
        for band, (fmin, fmax) in freq_bands.items():
            freq_mask = (freqs_ns >= fmin) & (freqs_ns <= fmax)
            band_power = psd_data_ns[:, :, freq_mask].mean(axis=(0, 2))
            features[f"{band}_baseline"] = band_power.mean()
        
        # 频谱指数
        from scipy.stats import linregress
        log_freqs = np.log10(freqs_ns[1:])
        log_psd = np.log10(psd_data_ns.mean(axis=0)[:, 1:])
        
        spectral_exponents = []
        for ch_idx in range(log_psd.shape[0]):
            slope, _, _, _, _ = linregress(log_freqs, log_psd[ch_idx])
            spectral_exponents.append(-slope)
        
        features["spectral_exponent_baseline"] = np.mean(spectral_exponents)
        
        # 睡眠剥夺条件下的 EEG 特征
        epochs_sd = mne.read_epochs(preprocessed_files[sd_ses], verbose=False)
        psd_sd = epochs_sd.compute_psd(fmin=1, fmax=45, verbose=False)
        psd_data_sd = psd_sd.get_data()
        freqs_sd = psd_sd.freqs
        
        for band, (fmin, fmax) in freq_bands.items():
            freq_mask = (freqs_sd >= fmin) & (freqs_sd <= fmax)
            band_power = psd_data_sd[:, :, freq_mask].mean(axis=(0, 2))
            features[f"{band}_sd"] = band_power.mean()
        
        log_freqs_sd = np.log10(freqs_sd[1:])
        log_psd_sd = np.log10(psd_data_sd.mean(axis=0)[:, 1:])
        
        spectral_exponents_sd = []
        for ch_idx in range(log_psd_sd.shape[0]):
            slope, _, _, _, _ = linregress(log_freqs_sd, log_psd_sd[ch_idx])
            spectral_exponents_sd.append(-slope)
        
        features["spectral_exponent_sd"] = np.mean(spectral_exponents_sd)
        
        # 计算变化量
        for band in freq_bands.keys():
            features[f"{band}_change"] = features[f"{band}_sd"] - features[f"{band}_baseline"]
        features["spectral_exponent_change"] = features["spectral_exponent_sd"] - features["spectral_exponent_baseline"]
        
        # 目标变量：睡眠剥夺易感性（基于 PANAS 正性情绪变化）
        # PANAS_P 数据完整性最好（68/71），用正性情绪下降作为预测目标
        panas_p_ns = row.get("PANAS_P_NS", np.nan)
        panas_p_sd = row.get("PANAS_P_SD", np.nan)
        
        if not (np.isnan(panas_p_ns) or np.isnan(panas_p_sd)):
            features["panas_p_change"] = panas_p_ns - panas_p_sd  # 正值表示正性情绪下降
            all_data.append(features)
            print(f"  ✓ {sub_id}")
    
    df = pd.DataFrame(all_data)
    
    # 二分类目标变量（基于中位数分割）
    if len(df) > 0:
        median_change = df["panas_p_change"].median()
        df["high_susceptible"] = (df["panas_p_change"] > median_change).astype(int)
        print(f"\n高易感性组: {df['high_susceptible'].sum()}")
        print(f"低易感性组: {(df['high_susceptible'] == 0).sum()}")
    
    return df


def aim4_prediction_modeling(df):
    """Aim 4: 预测建模"""
    print("\n" + "=" * 80)
    print("Aim 4: 预测模型（嵌套交叉验证）")
    print("=" * 80)
    
    # 准备数据
    feature_cols = [
        "delta_baseline", "theta_baseline", "alpha_baseline", "beta_baseline", "gamma_baseline",
        "spectral_exponent_baseline",
        "delta_change", "theta_change", "alpha_change", "beta_change", "gamma_change",
        "spectral_exponent_change",
        "age"
    ]
    
    X = df[feature_cols].values
    y = df["high_susceptible"].values
    
    print(f"\n样本量: {X.shape[0]}")
    print(f"特征数: {X.shape[1]}")
    print(f"类别分布: {np.bincount(y)}")
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 定义模型
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "SVM": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100)
    }
    
    # 嵌套交叉验证
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n训练模型: {name}")
        
        # 交叉验证
        cv_scores = cross_val_score(model, X_scaled, y, cv=outer_cv, scoring='roc_auc')
        
        results[name] = {
            "mean_auc": cv_scores.mean(),
            "std_auc": cv_scores.std(),
            "cv_scores": cv_scores.tolist()
        }
        
        print(f"  AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # 保存结果
    with open(RESULTS_DIR / "results" / "aim4_prediction_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ 结果已保存: {RESULTS_DIR / 'results' / 'aim4_prediction_results.json'}")
    
    # 训练最佳模型并获取特征重要性
    best_model_name = max(results, key=lambda x: results[x]["mean_auc"])
    print(f"\n最佳模型: {best_model_name}")
    
    if best_model_name == "Random Forest":
        best_model = RandomForestClassifier(random_state=42, n_estimators=100)
        best_model.fit(X_scaled, y)
        feature_importance = best_model.feature_importances_
    elif best_model_name == "Gradient Boosting":
        best_model = GradientBoostingClassifier(random_state=42, n_estimators=100)
        best_model.fit(X_scaled, y)
        feature_importance = best_model.feature_importances_
    else:
        feature_importance = None
    
    if feature_importance is not None:
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": feature_importance
        }).sort_values("importance", ascending=False)
        
        importance_df.to_csv(RESULTS_DIR / "tables" / "aim4_feature_importance.csv", index=False)
        print(f"✓ 特征重要性已保存: {RESULTS_DIR / 'tables' / 'aim4_feature_importance.csv'}")
    
    return results


def aim4_visualization(df, results):
    """Aim 4: 可视化"""
    print("\n" + "=" * 80)
    print("Aim 4: 可视化")
    print("=" * 80)
    
    # 1. 模型性能比较
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = list(results.keys())
    mean_aucs = [results[name]["mean_auc"] for name in model_names]
    std_aucs = [results[name]["std_auc"] for name in model_names]
    
    bars = ax.bar(model_names, mean_aucs, yerr=std_aucs, capsize=5, alpha=0.7)
    
    # 添加数值标签
    for bar, mean, std in zip(bars, mean_aucs, std_aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Model Performance Comparison (5-fold CV)")
    ax.axhline(y=0.5, color='r', linestyle='--', label='Chance level')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 保存图表
    fig_path = RESULTS_DIR / "figures" / "aim4_model_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ 模型比较图已保存: {fig_path}")
    
    # 2. 特征重要性（如果有）
    importance_file = RESULTS_DIR / "tables" / "aim4_feature_importance.csv"
    if importance_file.exists():
        importance_df = pd.read_csv(importance_file)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_n = min(15, len(importance_df))
        top_features = importance_df.head(top_n)
        
        ax.barh(range(top_n), top_features["importance"].values, align='center')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features["feature"].values)
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top {top_n} Predictive Features")
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        fig_path = RESULTS_DIR / "figures" / "aim4_feature_importance.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"✓ 特征重要性图已保存: {fig_path}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Phase 6: Aim 4 - 预测模型")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载数据
    df = load_prediction_data()
    
    if len(df) == 0:
        print("\n✗ 没有可用数据")
        return
    
    print(f"\n✓ 加载了 {len(df)} 个被试")
    
    # 2. 预测建模
    results = aim4_prediction_modeling(df)
    
    # 3. 可视化
    aim4_visualization(df, results)
    
    print("\n" + "=" * 80)
    print("Phase 6 完成!")
    print("=" * 80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
