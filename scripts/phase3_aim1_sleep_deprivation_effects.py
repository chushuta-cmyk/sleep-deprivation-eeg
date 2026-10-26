#!/usr/bin/env python3
"""
Phase 3: Aim 1 - 睡眠剥夺对 EEG 频谱的影响
使用线性混合模型（LMM）分析睡眠剥夺对各频带功率的影响
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
import statsmodels.formula.api as smf
from scipy import stats

# 项目路径
PROJECT_ROOT = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/sleep_deprivation_eeg")
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DERIVATIVES_DIR = PROJECT_ROOT / "data" / "derivatives"
RESULTS_DIR = PROJECT_ROOT / "results"

def load_all_preprocessed_data():
    """加载所有预处理后的数据"""
    print("=" * 80)
    print("加载所有预处理后的数据...")
    print("=" * 80)
    
    participants_tsv = pd.read_csv(RAW_DATA_DIR / "participants.tsv", sep="\t")
    
    all_data = []
    
    for idx, row in participants_tsv.iterrows():
        sub_id = row["participant_id"]
        sub_dir = RAW_DATA_DIR / sub_id
        
        if not sub_dir.exists():
            continue
        
        for ses in ["ses-1", "ses-2"]:
            preprocessed_dir = DERIVATIVES_DIR / "preprocessed" / sub_id / ses
            epochs_file = preprocessed_dir / f"{sub_id}_{ses}-epo.fif"
            
            if not epochs_file.exists():
                continue
            
            # 确定条件
            session_order = row["SessionOrder"]
            if session_order == "NS->SD":
                condition = "NS" if ses == "ses-1" else "SD"
            else:
                condition = "SD" if ses == "ses-1" else "NS"
            
            # 加载 epochs
            epochs = mne.read_epochs(epochs_file, verbose=False)
            
            # 提取频谱特征
            psd = epochs.compute_psd(fmin=1, fmax=45, verbose=False)
            psd_data = psd.get_data()
            freqs = psd.freqs
            
            # 计算各频带功率
            freq_bands = {
                "delta": (1, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30),
                "gamma": (30, 45)
            }
            
            features = {
                "subject": sub_id,
                "session": ses,
                "condition": condition,
                "age": row["Age"],
                "gender": row["Gender"],
                "session_order": session_order,
                "n_epochs": len(epochs)
            }
            
            for band, (fmin, fmax) in freq_bands.items():
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                band_power = psd_data[:, :, freq_mask].mean(axis=(0, 2))
                features[f"{band}_power"] = band_power.mean()
                features[f"{band}_power_std"] = band_power.std()
            
            # 频谱指数
            from scipy.stats import linregress
            log_freqs = np.log10(freqs[1:])
            log_psd = np.log10(psd_data.mean(axis=0)[:, 1:])
            
            spectral_exponents = []
            for ch_idx in range(log_psd.shape[0]):
                slope, _, _, _, _ = linregress(log_freqs, log_psd[ch_idx])
                spectral_exponents.append(-slope)
            
            features["spectral_exponent"] = np.mean(spectral_exponents)
            
            all_data.append(features)
            print(f"  ✓ {sub_id}/{ses} ({condition})")
    
    return pd.DataFrame(all_data)


def aim1_lmm_analysis(df):
    """Aim 1: 线性混合模型分析"""
    print("\n" + "=" * 80)
    print("Aim 1: 睡眠剥夺对 EEG 频谱的影响（LMM）")
    print("=" * 80)
    
    # 准备数据
    df_analysis = df.copy()
    df_analysis["condition_num"] = (df_analysis["condition"] == "SD").astype(int)
    df_analysis["gender_num"] = (df_analysis["gender"] == "M").astype(int)
    
    # 对每个频带进行 LMM 分析
    freq_bands = ["delta", "theta", "alpha", "beta", "gamma", "spectral_exponent"]
    
    results = []
    
    for band in freq_bands:
        print(f"\n分析频带: {band}")
        
        if band == "spectral_exponent":
            target_col = "spectral_exponent"
        else:
            target_col = f"{band}_power"
        
        # 线性混合模型：条件 + 年龄 + 性别 + (1|subject)
        # 使用 statsmodels 的 MixedLM
        try:
            model = smf.mixedlm(
                f"{target_col} ~ condition_num + age + gender_num",
                df_analysis,
                groups=df_analysis["subject"]
            )
            result = model.fit(reml=True, verbose=False)
            
            # 提取结果
            coef = result.params["condition_num"]
            se = result.bse["condition_num"]
            p_value = result.pvalues["condition_num"]
            ci = result.conf_int().loc["condition_num"]
            
            results.append({
                "band": band,
                "coefficient": coef,
                "se": se,
                "p_value": p_value,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "n_subjects": df_analysis["subject"].nunique(),
                "n_observations": len(df_analysis)
            })
            
            print(f"  系数: {coef:.4e} ± {se:.4e}")
            print(f"  p值: {p_value:.4f}")
            print(f"  95% CI: [{ci[0]:.4e}, {ci[1]:.4e}]")
            
        except Exception as e:
            print(f"  ✗ 模型拟合失败: {e}")
            results.append({
                "band": band,
                "coefficient": np.nan,
                "se": np.nan,
                "p_value": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "n_subjects": df_analysis["subject"].nunique(),
                "n_observations": len(df_analysis)
            })
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "tables" / "aim1_lmm_results.csv", index=False)
    print(f"\n✓ 结果已保存: {RESULTS_DIR / 'tables' / 'aim1_lmm_results.csv'}")
    
    return results_df


def aim1_visualization(df, results_df):
    """Aim 1: 可视化"""
    print("\n" + "=" * 80)
    print("Aim 1: 可视化")
    print("=" * 80)
    
    # 1. 箱线图：NS vs SD
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Aim 1: Sleep Deprivation Effects on EEG Spectral Power", 
                 fontsize=16, fontweight='bold')
    
    freq_bands = ["delta", "theta", "alpha", "beta", "gamma", "spectral_exponent"]
    
    for idx, band in enumerate(freq_bands):
        ax = axes[idx // 3, idx % 3]
        
        if band == "spectral_exponent":
            target_col = "spectral_exponent"
            ylabel = "Spectral Exponent"
        else:
            target_col = f"{band}_power"
            ylabel = f"{band.capitalize()} Power (μV²/Hz)"
        
        # 箱线图
        sns.boxplot(data=df, x="condition", y=target_col, ax=ax, palette=["#4C72B0", "#DD8452"])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{band.capitalize()} Band")
        ax.grid(axis='y', alpha=0.3)
        
        # 添加显著性标记
        if idx < len(results_df):
            p_val = results_df.iloc[idx]["p_value"]
            if not np.isnan(p_val):
                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"
                else:
                    sig = "ns"
                ax.text(0.5, 0.95, f"p={p_val:.3f} {sig}", 
                       transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图表
    fig_path = RESULTS_DIR / "figures" / "aim1_sleep_deprivation_effects.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ 图表已保存: {fig_path}")
    
    # 2. 配对图：每个被试的 NS vs SD
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Paired Comparisons: Normal Sleep vs Sleep Deprivation", 
                 fontsize=14, fontweight='bold')
    
    for idx, band in enumerate(["alpha", "theta", "spectral_exponent"]):
        ax = axes[idx]
        
        if band == "spectral_exponent":
            target_col = "spectral_exponent"
            ylabel = "Spectral Exponent"
        else:
            target_col = f"{band}_power"
            ylabel = f"{band.capitalize()} Power (μV²/Hz)"
        
        # 计算每个被试的平均值
        subject_means = df.groupby(["subject", "condition"])[target_col].mean().unstack()
        
        # 配对线
        for subject in subject_means.index:
            ns_val = subject_means.loc[subject, "NS"]
            sd_val = subject_means.loc[subject, "SD"]
            ax.plot([0, 1], [ns_val, sd_val], 'o-', alpha=0.3, color='gray')
        
        # 均值
        mean_ns = subject_means["NS"].mean()
        mean_sd = subject_means["SD"].mean()
        ax.plot([0, 1], [mean_ns, mean_sd], 'o-', linewidth=3, color='red', 
               markersize=10, label='Mean')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal Sleep", "Sleep Deprivation"])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{band.capitalize()} Band")
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    # 保存图表
    fig_path = RESULTS_DIR / "figures" / "aim1_paired_comparisons.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ 配对图已保存: {fig_path}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Phase 3: Aim 1 - 睡眠剥夺对 EEG 频谱的影响")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载所有数据
    df = load_all_preprocessed_data()
    
    if len(df) == 0:
        print("\n✗ 没有可用数据，请先运行 Phase 2 预处理所有被试")
        return
    
    print(f"\n✓ 加载了 {len(df)} 个被试-session 组合")
    print(f"  被试数: {df['subject'].nunique()}")
    print(f"  条件分布: {df['condition'].value_counts().to_dict()}")
    
    # 2. LMM 分析
    results_df = aim1_lmm_analysis(df)
    
    # 3. 可视化
    aim1_visualization(df, results_df)
    
    print("\n" + "=" * 80)
    print("Phase 3 完成!")
    print("=" * 80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
