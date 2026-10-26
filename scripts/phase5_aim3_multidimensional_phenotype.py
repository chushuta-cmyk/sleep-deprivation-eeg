#!/usr/bin/env python3
"""
Phase 5: Aim 3 - 多维表型结构
使用偏最小二乘法（PLS）分析 EEG 特征与行为/情绪变量的关系
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from scipy import stats

# 项目路径
PROJECT_ROOT = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/sleep_deprivation_eeg")
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DERIVATIVES_DIR = PROJECT_ROOT / "data" / "derivatives"
RESULTS_DIR = PROJECT_ROOT / "results"

def load_complete_data():
    """加载完整数据（EEG + 行为 + 情绪）"""
    print("=" * 80)
    print("加载完整数据...")
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
        
        # 提取 EEG 特征
        features = {
            "subject": sub_id,
            "age": row["Age"],
            "gender": row["Gender"],
            "session_order": session_order
        }
        
        for condition, ses in [("NS", ns_ses), ("SD", sd_ses)]:
            epochs = mne.read_epochs(preprocessed_files[ses], verbose=False)
            
            # 提取频谱特征
            psd = epochs.compute_psd(fmin=1, fmax=45, verbose=False)
            psd_data = psd.get_data()
            freqs = psd.freqs
            
            freq_bands = {
                "delta": (1, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30),
                "gamma": (30, 45)
            }
            
            for band, (fmin, fmax) in freq_bands.items():
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                band_power = psd_data[:, :, freq_mask].mean(axis=(0, 2))
                features[f"{band}_{condition}"] = band_power.mean()
            
            # 频谱指数
            from scipy.stats import linregress
            log_freqs = np.log10(freqs[1:])
            log_psd = np.log10(psd_data.mean(axis=0)[:, 1:])
            
            spectral_exponents = []
            for ch_idx in range(log_psd.shape[0]):
                slope, _, _, _, _ = linregress(log_freqs, log_psd[ch_idx])
                spectral_exponents.append(-slope)
            
            features[f"spectral_exponent_{condition}"] = np.mean(spectral_exponents)
        
        # 添加行为/情绪变量（从 participants.tsv）
        # 睡眠剥夺条件下的变量
        features["PANAS_P_SD"] = row.get("PANAS_P_SD", np.nan)
        features["PANAS_N_SD"] = row.get("PANAS_N_SD", np.nan)
        features["SAI_SD"] = row.get("SAI_SD", np.nan)
        features["SSS_SD"] = row.get("SSS_SD", np.nan)
        features["KSS_SD"] = row.get("KSS_SD", np.nan)
        
        # 正常睡眠条件下的变量
        features["PANAS_P_NS"] = row.get("PANAS_P_NS", np.nan)
        features["PANAS_N_NS"] = row.get("PANAS_N_NS", np.nan)
        features["SAI_NS"] = row.get("SAI_NS", np.nan)
        features["SSS_NS"] = row.get("SSS_NS", np.nan)
        features["KSS_NS"] = row.get("KSS_NS", np.nan)
        
        # 睡眠质量
        features["PSQI_GlobalScore"] = row.get("PSQI_GlobalScore", np.nan)
        
        all_data.append(features)
        print(f"  ✓ {sub_id}")
    
    return pd.DataFrame(all_data)


def aim3_pls_analysis(df):
    """Aim 3: PLS 分析"""
    print("\n" + "=" * 80)
    print("Aim 3: 多维表型结构（PLS）")
    print("=" * 80)
    
    # 准备数据
    # X: EEG 特征（睡眠剥夺条件下的变化）
    eeg_features = ["delta", "theta", "alpha", "beta", "gamma", "spectral_exponent"]
    
    X_data = []
    for feature in eeg_features:
        change = df[f"{feature}_SD"] - df[f"{feature}_NS"]
        X_data.append(change)
    
    X = np.column_stack(X_data)
    
    # Y: 行为/情绪变量（睡眠剥夺条件下）
    # 只使用缺失较少的 PANAS 变量
    y_vars = ["PANAS_P_SD", "PANAS_N_SD"]
    Y = df[y_vars].values
    
    # 删除缺失值
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
    X = X[mask]
    Y = Y[mask]
    
    print(f"\n样本量: {X.shape[0]}")
    print(f"X 维度: {X.shape[1]} (EEG 特征)")
    print(f"Y 维度: {Y.shape[1]} (行为/情绪变量)")
    
    # 标准化
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = StandardScaler().fit_transform(Y)
    
    # PLS 回归
    n_components = min(3, X.shape[1], Y.shape[1])
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, Y_scaled)
    
    # 计算 R²
    Y_pred = pls.predict(X_scaled)
    r2 = 1 - np.sum((Y_scaled - Y_pred) ** 2) / np.sum(Y_scaled ** 2)
    
    print(f"\nPLS 模型 R²: {r2:.3f}")
    
    # 交叉验证
    cv_scores = cross_val_score(pls, X_scaled, Y_scaled, cv=5, scoring='r2')
    print(f"交叉验证 R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # 保存结果
    results = {
        "n_components": n_components,
        "r2": r2,
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
        "x_weights": pls.x_weights_.tolist(),
        "y_weights": pls.y_weights_.tolist(),
        "n_samples": X.shape[0]
    }
    
    with open(RESULTS_DIR / "aim3_pls_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ 结果已保存: {RESULTS_DIR / 'results' / 'aim3_pls_results.json'}")
    
    return pls, X_scaled, Y_scaled, eeg_features, y_vars


def aim3_visualization(pls, X, Y, x_features, y_vars):
    """Aim 3: 可视化"""
    print("\n" + "=" * 80)
    print("Aim 3: 可视化")
    print("=" * 80)
    
    # 1. PLS 成分图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Aim 3: Partial Least Squares (PLS) Analysis", 
                 fontsize=16, fontweight='bold')
    
    # X 权重
    x_weights = pls.x_weights_
    im1 = axes[0].imshow(x_weights, cmap='RdBu_r', aspect='auto')
    axes[0].set_xticks(range(len(y_vars)))
    axes[0].set_xticklabels(y_vars, rotation=45, ha='right')
    axes[0].set_yticks(range(len(x_features)))
    axes[0].set_yticklabels(x_features)
    axes[0].set_title("X Weights (EEG Features)")
    axes[0].set_xlabel("PLS Components")
    plt.colorbar(im1, ax=axes[0])
    
    # Y 权重
    y_weights = pls.y_weights_
    im2 = axes[1].imshow(y_weights, cmap='RdBu_r', aspect='auto')
    axes[1].set_xticks(range(len(y_vars)))
    axes[1].set_xticklabels(y_vars, rotation=45, ha='right')
    axes[1].set_yticks(range(len(x_features)))
    axes[1].set_yticklabels(x_features)
    axes[1].set_title("Y Weights (Behavioral/Emotional)")
    axes[1].set_xlabel("PLS Components")
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    # 保存图表
    fig_path = RESULTS_DIR / "figures" / "aim3_pls_weights.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ PLS 权重图已保存: {fig_path}")
    
    # 2. 潜变量散点图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # X 潜变量
    X_scores = pls.transform(X)
    axes[0].scatter(X_scores[:, 0], X_scores[:, 1], alpha=0.6, s=50)
    axes[0].set_xlabel("PLS Component 1")
    axes[0].set_ylabel("PLS Component 2")
    axes[0].set_title("X Scores (EEG Features)")
    axes[0].grid(alpha=0.3)
    
    # Y 潜变量（通过 predict 获得 Y 在潜变量空间的投影）
    Y_pred_scaled = pls.predict(X)
    axes[1].scatter(X_scores[:, 0], Y_pred_scaled[:, 0], alpha=0.6, s=50, color='orange')
    axes[1].set_xlabel("PLS Component 1 (X Score)")
    axes[1].set_ylabel("Predicted Y (Component 1)")
    axes[1].set_title("X Score vs Predicted Y")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    fig_path = RESULTS_DIR / "figures" / "aim3_pls_scores.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ PLS 潜变量图已保存: {fig_path}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Phase 5: Aim 3 - 多维表型结构")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载数据
    df = load_complete_data()
    
    if len(df) == 0:
        print("\n✗ 没有可用数据")
        return
    
    print(f"\n✓ 加载了 {len(df)} 个被试")
    
    # 2. PLS 分析
    pls, X, Y, x_features, y_vars = aim3_pls_analysis(df)
    
    # 3. 可视化
    aim3_visualization(pls, X, Y, x_features, y_vars)
    
    print("\n" + "=" * 80)
    print("Phase 5 完成!")
    print("=" * 80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
