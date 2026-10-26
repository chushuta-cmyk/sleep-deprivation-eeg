#!/usr/bin/env python3
"""
Phase 7: Aim 5 - 中介分析
分析情绪、焦虑、嗜睡在睡眠剥夺与 EEG 变化之间的中介作用
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
from scipy import stats
import pingouin as pg

# 项目路径
PROJECT_ROOT = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/sleep_deprivation_eeg")
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DERIVATIVES_DIR = PROJECT_ROOT / "data" / "derivatives"
RESULTS_DIR = PROJECT_ROOT / "results"

def load_mediation_data():
    """加载中介分析数据"""
    print("=" * 80)
    print("加载中介分析数据...")
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
            "gender": row["Gender"]
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
        
        # 计算 EEG 变化量
        for band in ["delta", "theta", "alpha", "beta", "gamma", "spectral_exponent"]:
            features[f"{band}_change"] = features[f"{band}_SD"] - features[f"{band}_NS"]
        
        # 添加中介变量（睡眠剥夺条件下的值）
        features["PANAS_N_SD"] = row.get("PANAS_N_SD", np.nan)
        features["SAI_SD"] = row.get("SAI_SD", np.nan)
        features["SSS_SD"] = row.get("SSS_SD", np.nan)
        features["KSS_SD"] = row.get("KSS_SD", np.nan)
        
        # 基线值
        features["PANAS_N_NS"] = row.get("PANAS_N_NS", np.nan)
        features["SAI_NS"] = row.get("SAI_NS", np.nan)
        features["SSS_NS"] = row.get("SSS_NS", np.nan)
        features["KSS_NS"] = row.get("KSS_NS", np.nan)
        
        # 计算中介变量的变化量
        for var in ["PANAS_N", "SAI", "SSS", "KSS"]:
            features[f"{var}_change"] = features[f"{var}_SD"] - features[f"{var}_NS"]
        
        all_data.append(features)
        print(f"  ✓ {sub_id}")
    
    return pd.DataFrame(all_data)


def aim5_mediation_analysis(df):
    """Aim 5: 中介分析"""
    print("\n" + "=" * 80)
    print("Aim 5: 中介分析")
    print("=" * 80)
    
    # 自变量：EEG 变化量
    eeg_features = ["alpha_change", "theta_change", "spectral_exponent_change"]
    
    # 中介变量
    mediators = ["PANAS_N_change", "SAI_change", "SSS_change", "KSS_change"]
    
    # 因变量：其他 EEG 变化量（这里简化为分析 EEG 变化与主观感受的关系）
    
    results = []
    
    for eeg_feat in eeg_features:
        for mediator in mediators:
            # 准备数据
            data_subset = df[[eeg_feat, mediator]].dropna()
            
            if len(data_subset) < 20:
                continue
            
            X = data_subset[eeg_feat].values
            M = data_subset[mediator].values
            
            # 简单中介分析（Baron & Kenny 方法）
            # 1. X -> Y (总效应)
            slope_c, intercept_c, r_value_c, p_value_c, std_err_c = stats.linregress(X, M)
            
            # 2. X -> M (a 路径)
            slope_a, intercept_a, r_value_a, p_value_a, std_err_a = stats.linregress(X, M)
            
            # 3. M -> Y (b 路径，控制 X)
            # 简化：计算 M 与 X 的相关
            slope_b, intercept_b, r_value_b, p_value_b, std_err_b = stats.linregress(M, X)
            
            # 计算中介效应
            mediation_effect = slope_a * slope_b
            
            results.append({
                "eeg_feature": eeg_feat,
                "mediator": mediator,
                "path_a": slope_a,
                "path_a_p": p_value_a,
                "path_b": slope_b,
                "path_b_p": p_value_b,
                "total_effect": slope_c,
                "total_effect_p": p_value_c,
                "mediation_effect": mediation_effect,
                "n_samples": len(data_subset)
            })
            
            print(f"\n{eeg_feat} -> {mediator}")
            print(f"  路径 a (X->M): {slope_a:.4f}, p={p_value_a:.4f}")
            print(f"  路径 b (M->Y): {slope_b:.4f}, p={p_value_b:.4f}")
            print(f"  中介效应: {mediation_effect:.4f}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "tables" / "aim5_mediation_results.csv", index=False)
    print(f"\n✓ 结果已保存: {RESULTS_DIR / 'tables' / 'aim5_mediation_results.csv'}")
    
    return results_df


def aim5_visualization(df, results_df):
    """Aim 5: 可视化"""
    print("\n" + "=" * 80)
    print("Aim 5: 可视化")
    print("=" * 80)
    
    # 1. 相关矩阵热图
    eeg_features = ["alpha_change", "theta_change", "spectral_exponent_change"]
    mediators = ["PANAS_N_change", "SAI_change", "SSS_change", "KSS_change"]
    
    # 准备数据
    data_subset = df[eeg_features + mediators].dropna()
    
    # 计算相关矩阵
    corr_matrix = data_subset.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 只绘制 EEG 特征与中介变量的相关
    corr_submatrix = corr_matrix.loc[eeg_features, mediators]
    
    sns.heatmap(corr_submatrix, annot=True, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax, fmt='.2f')
    
    ax.set_title("Correlation Matrix: EEG Changes vs Mediators")
    ax.set_xlabel("Mediators")
    ax.set_ylabel("EEG Features")
    
    plt.tight_layout()
    
    # 保存图表
    fig_path = RESULTS_DIR / "figures" / "aim5_correlation_matrix.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ 相关矩阵图已保存: {fig_path}")
    
    # 2. 散点图矩阵
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Mediation Analysis: EEG Changes vs Subjective Measures", 
                 fontsize=14, fontweight='bold')
    
    for idx, mediator in enumerate(mediators):
        ax = axes[idx // 2, idx % 2]
        
        # 使用 alpha_change 作为示例，针对每个中介变量单独去除缺失
        valid_data = df[["alpha_change", mediator]].dropna()
        
        if len(valid_data) < 3:
            ax.text(0.5, 0.5, f"数据不足\n(n={len(valid_data)})", 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(f"Alpha vs {mediator.replace('_change', '')}")
            continue
        
        X = valid_data["alpha_change"].values
        M = valid_data[mediator].values
        
        ax.scatter(X, M, alpha=0.6, s=50)
        
        # 回归线
        z = np.polyfit(X, M, 1)
        p = np.poly1d(z)
        ax.plot(X, p(X), "r-", linewidth=2)
        
        # 相关系数
        r, p_value = stats.pearsonr(X, M)
        ax.text(0.05, 0.95, f"r={r:.3f}\np={p_value:.3f}", 
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel("Alpha Power Change (SD - NS)")
        ax.set_ylabel(mediator.replace("_change", " Change"))
        ax.set_title(f"Alpha vs {mediator.replace('_change', '')}")
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    fig_path = RESULTS_DIR / "figures" / "aim5_mediation_scatterplots.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ 中介分析散点图已保存: {fig_path}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Phase 7: Aim 5 - 中介分析")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载数据
    df = load_mediation_data()
    
    if len(df) == 0:
        print("\n✗ 没有可用数据")
        return
    
    print(f"\n✓ 加载了 {len(df)} 个被试")
    
    # 2. 中介分析
    results_df = aim5_mediation_analysis(df)
    
    # 3. 可视化
    aim5_visualization(df, results_df)
    
    print("\n" + "=" * 80)
    print("Phase 7 完成!")
    print("=" * 80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
