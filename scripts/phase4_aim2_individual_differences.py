#!/usr/bin/env python3
"""
Phase 4: Aim 2 - 个体差异的稳定性
计算组内相关系数（ICC）评估 EEG 特征的个体差异稳定性
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
import pingouin as pg
from scipy import stats

# 项目路径
PROJECT_ROOT = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/sleep_deprivation_eeg")
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DERIVATIVES_DIR = PROJECT_ROOT / "data" / "derivatives"
RESULTS_DIR = PROJECT_ROOT / "results"

def load_wide_format_data():
    """加载宽格式数据（每个被试一行，包含两个条件）"""
    print("=" * 80)
    print("加载数据并转换为宽格式...")
    print("=" * 80)
    
    # 获取所有有数据的被试
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
        
        # 提取两个条件下的特征
        features = {"subject": sub_id, "age": row["Age"], "gender": row["Gender"]}
        
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
        
        all_data.append(features)
        print(f"  ✓ {sub_id}")
    
    return pd.DataFrame(all_data)


def aim2_icc_analysis(df):
    """Aim 2: ICC 分析"""
    print("\n" + "=" * 80)
    print("Aim 2: 个体差异的稳定性（ICC）")
    print("=" * 80)
    
    freq_bands = ["delta", "theta", "alpha", "beta", "gamma", "spectral_exponent"]
    
    results = []
    
    for band in freq_bands:
        print(f"\n分析频带: {band}")
        
        # 准备数据（长格式）
        ns_data = df[["subject", f"{band}_NS"]].copy()
        ns_data.columns = ["subject", "value"]
        ns_data["condition"] = "NS"
        
        sd_data = df[["subject", f"{band}_SD"]].copy()
        sd_data.columns = ["subject", "value"]
        sd_data["condition"] = "SD"
        
        long_data = pd.concat([ns_data, sd_data], ignore_index=True)
        
        # 计算 ICC(2,1) - 双向随机效应模型
        try:
            icc_result = pg.intraclass_corr(data=long_data, targets='subject', 
                                           raters='condition', ratings='value')
            
            # 提取 ICC(2,1) - 使用 pingouin 实际列名 Type/pval/CI95
            icc_row = icc_result[icc_result['Type'] == 'ICC(A,1)']
            if icc_row.empty:
                icc_row = icc_result.iloc[[1]]  # 第二行通常是 ICC(A,1)
            
            icc_val = icc_row['ICC'].values[0]
            icc_ci = icc_row['CI95'].values[0]
            p_val = icc_row['pval'].values[0]
            
            results.append({
                "band": band,
                "icc": icc_val,
                "ci_lower": icc_ci[0],
                "ci_upper": icc_ci[1],
                "p_value": p_val,
                "n_subjects": df.shape[0]
            })
            
            print(f"  ICC(A,1): {icc_val:.3f} [{icc_ci[0]:.3f}, {icc_ci[1]:.3f}]")
            print(f"  p值: {p_val:.4f}")
            
            # 解释 ICC 值
            if icc_val < 0.5:
                reliability = "差"
            elif icc_val < 0.75:
                reliability = "中等"
            elif icc_val < 0.9:
                reliability = "良好"
            else:
                reliability = "优秀"
            print(f"  可靠性: {reliability}")
            
        except Exception as e:
            print(f"  ✗ ICC 计算失败: {e}")
            results.append({
                "band": band,
                "icc": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "p_value": np.nan,
                "n_subjects": df.shape[0]
            })
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "tables" / "aim2_icc_results.csv", index=False)
    print(f"\n✓ 结果已保存: {RESULTS_DIR / 'tables' / 'aim2_icc_results.csv'}")
    
    return results_df


def aim2_visualization(df, results_df):
    """Aim 2: 可视化"""
    print("\n" + "=" * 80)
    print("Aim 2: 可视化")
    print("=" * 80)
    
    freq_bands = ["delta", "theta", "alpha", "beta", "gamma", "spectral_exponent"]
    
    # 1. 散点图：NS vs SD
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Aim 2: Test-Retest Reliability (ICC)", 
                 fontsize=16, fontweight='bold')
    
    for idx, band in enumerate(freq_bands):
        ax = axes[idx // 3, idx % 3]
        
        ns_vals = df[f"{band}_NS"]
        sd_vals = df[f"{band}_SD"]
        
        # 散点图
        ax.scatter(ns_vals, sd_vals, alpha=0.6, s=50)
        
        # 对角线
        min_val = min(ns_vals.min(), sd_vals.min())
        max_val = max(ns_vals.max(), sd_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
               label='Identity line')
        
        # 相关线
        z = np.polyfit(ns_vals, sd_vals, 1)
        p = np.poly1d(z)
        ax.plot(ns_vals, p(ns_vals), "g--", alpha=0.8, linewidth=2, 
               label=f'Regression (slope={z[0]:.2f})')
        
        # 添加 ICC 值
        if idx < len(results_df):
            icc = results_df.iloc[idx]["icc"]
            if not np.isnan(icc):
                ax.text(0.05, 0.95, f"ICC={icc:.3f}", 
                       transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        ax.set_xlabel(f"{band.capitalize()} - Normal Sleep")
        ax.set_ylabel(f"{band.capitalize()} - Sleep Deprivation")
        ax.set_title(f"{band.capitalize()} Band")
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    fig_path = RESULTS_DIR / "figures" / "aim2_icc_scatterplots.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ 散点图已保存: {fig_path}")
    
    # 2. Bland-Altman 图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Bland-Altman Plots: Agreement between Conditions", 
                 fontsize=16, fontweight='bold')
    
    for idx, band in enumerate(freq_bands):
        ax = axes[idx // 3, idx % 3]
        
        ns_vals = df[f"{band}_NS"]
        sd_vals = df[f"{band}_SD"]
        
        # 计算差值和平均值
        mean_vals = (ns_vals + sd_vals) / 2
        diff_vals = ns_vals - sd_vals
        
        mean_diff = diff_vals.mean()
        std_diff = diff_vals.std()
        
        # 绘制
        ax.scatter(mean_vals, diff_vals, alpha=0.6, s=50)
        ax.axhline(mean_diff, color='red', linestyle='-', linewidth=2, 
                  label=f'Mean diff: {mean_diff:.3e}')
        ax.axhline(mean_diff + 1.96 * std_diff, color='orange', linestyle='--', 
                  label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.3e}')
        ax.axhline(mean_diff - 1.96 * std_diff, color='orange', linestyle='--', 
                  label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.3e}')
        
        ax.set_xlabel(f"Mean of {band.capitalize()} Power")
        ax.set_ylabel(f"Difference (NS - SD)")
        ax.set_title(f"{band.capitalize()} Band")
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    fig_path = RESULTS_DIR / "figures" / "aim2_bland_altman.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Bland-Altman 图已保存: {fig_path}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Phase 4: Aim 2 - 个体差异的稳定性")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载数据
    df = load_wide_format_data()
    
    if len(df) == 0:
        print("\n✗ 没有可用数据，请先运行 Phase 2 预处理所有被试")
        return
    
    print(f"\n✓ 加载了 {len(df)} 个被试（两个条件都有数据）")
    
    # 2. ICC 分析
    results_df = aim2_icc_analysis(df)
    
    # 3. 可视化
    aim2_visualization(df, results_df)
    
    print("\n" + "=" * 80)
    print("Phase 4 完成!")
    print("=" * 80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
