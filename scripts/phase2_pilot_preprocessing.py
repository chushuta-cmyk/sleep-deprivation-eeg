#!/usr/bin/env python3
"""
Phase 2: Pilot 预处理
测试 EEG 预处理流程，验证特征提取，生成 QC 报告
"""

import os
import json
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# 项目路径
PROJECT_ROOT = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/sleep_deprivation_eeg")
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DERIVATIVES_DIR = PROJECT_ROOT / "data" / "derivatives"
RESULTS_DIR = PROJECT_ROOT / "results"

def load_eeg_data(sub_id, ses):
    """加载 EEG 数据"""
    eeg_dir = RAW_DATA_DIR / sub_id / ses / "eeg"
    
    # 查找 .set 文件
    set_files = list(eeg_dir.glob("*_eeg.set"))
    if not set_files:
        print(f"  ✗ {sub_id}/{ses}: 未找到 .set 文件")
        return None
    
    set_file = set_files[0]
    print(f"  加载: {set_file.name}")
    
    try:
        # 使用 MNE 读取 EEGLAB 格式
        raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
        return raw
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return None


def preprocess_eeg(raw, sub_id, ses):
    """预处理 EEG 数据"""
    print(f"\n预处理 {sub_id}/{ses}...")
    
    # 1. 获取基本信息
    info = raw.info
    sfreq = info["sfreq"]
    n_channels = len(info["ch_names"])
    duration = raw.times[-1]
    
    print(f"  原始数据: {n_channels} 通道, {sfreq} Hz, {duration:.1f} 秒")
    
    # 2. 设置通道类型（确保都是 EEG）
    ch_types = {ch: "eeg" for ch in info["ch_names"]}
    raw.set_channel_types(ch_types)
    
    # 3. 设置montage（61通道，使用标准10-10系统）
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="warn")
    
    # 4. 重参考（平均参考）
    raw.set_eeg_reference("average", verbose=False)
    
    # 5. 滤波（0.5-45 Hz）
    raw.filter(0.5, 45, fir_design="firwin", verbose=False)
    
    # 6. 降采样（500 Hz → 250 Hz）
    if sfreq > 250:
        raw.resample(250, verbose=False)
        print(f"  降采样: {sfreq} Hz → 250 Hz")
    
    # 7. 坏道检测（EEG 适用方法）
    # 使用更宽松的阈值：检测明显异常的通道
    data = raw.get_data()
    std_per_channel = data.std(axis=1)
    
    # 检测平直通道（标准差 < 1e-8）
    flat_channels = np.where(std_per_channel < 1e-8)[0]
    # 检测高方差通道（标准差 > 5倍中位数，更宽松）
    high_var_channels = np.where(std_per_channel > 5 * np.median(std_per_channel))[0]
    
    bad_channel_indices = np.union1d(flat_channels, high_var_channels)
    bad_channels = [raw.ch_names[i] for i in bad_channel_indices]
    
    if bad_channels:
        print(f"  检测到坏道: {bad_channels}")
        raw.info["bads"] = bad_channels
        raw.interpolate_bads(verbose=False)
    else:
        print(f"  未检测到坏道")
    
    # 8. ICA 去伪迹
    ica = mne.preprocessing.ICA(n_components=20, random_state=42, verbose=False)
    ica.fit(raw, verbose=False)
    
    # 检查是否有 EOG 通道
    eog_channels = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
    
    if eog_channels:
        # 有 EOG 通道，自动检测眼电伪迹
        eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=False)
        exclude = eog_indices
        if exclude:
            print(f"  ICA 排除眼电伪迹成分: {exclude}")
            ica.apply(raw, exclude=exclude, verbose=False)
        else:
            print(f"  ICA 未检测到眼电伪迹成分")
    else:
        # 没有 EOG 通道，跳过自动检测，只做基本 ICA 去噪
        print(f"  无 EOG 通道，跳过自动伪迹检测")
        # 可选：手动检查前几个成分是否为伪迹（这里简化处理）
    
    # 保存 ICA 信息
    ica_info = {
        "n_components": ica.n_components_,
        "n_excluded": len(exclude) if eog_channels else 0,
        "excluded_components": exclude.tolist() if eog_channels else []
    }
    
    # 9. 分段（2s epochs，使用第一个点作为基线）
    events = mne.make_fixed_length_events(raw, duration=2.0)
    epochs = mne.Epochs(
        raw, events, tmin=0, tmax=2.0, baseline=(0, 0),
        preload=True, verbose=False
    )
    
    # 10. 去除坏段
    # 简单阈值：峰峰值 > 100 μV
    data = epochs.get_data()
    peak_to_peak = data.max(axis=2) - data.min(axis=2)  # (n_epochs, n_channels)
    bad_epochs = np.where(peak_to_peak.max(axis=1) > 100)[0]  # 任何通道超过阈值
    
    if len(bad_epochs) > 0:
        epochs.drop(bad_epochs)
        print(f"  去除 {len(bad_epochs)} 个坏段")
    
    print(f"  最终: {len(epochs)} 个 epochs")
    
    return raw, epochs, ica


def extract_features(epochs, sub_id, ses):
    """提取 EEG 特征"""
    print(f"\n提取特征 {sub_id}/{ses}...")
    
    # 1. 频谱功率
    freq_bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45)
    }
    
    # 计算频谱功率
    psd = epochs.compute_psd(fmin=1, fmax=45, verbose=False)
    psd_data = psd.get_data()  # (n_epochs, n_channels, n_freqs)
    freqs = psd.freqs
    
    # 计算每个频带的平均功率
    band_powers = {}
    for band, (fmin, fmax) in freq_bands.items():
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = psd_data[:, :, freq_mask].mean(axis=(0, 2))  # 平均 over epochs 和 frequencies
        band_powers[band] = band_power
        print(f"  {band} ({fmin}-{fmax} Hz): 平均功率 = {band_power.mean():.2e}")
    
    # 2. 频谱指数（spectral exponent）
    # 使用 1/f 拟合
    from scipy.stats import linregress
    
    log_freqs = np.log10(freqs[1:])  # 跳过 0 Hz
    log_psd = np.log10(psd_data.mean(axis=0)[:, 1:])  # 平均 over epochs
    
    spectral_exponents = []
    for ch_idx in range(log_psd.shape[0]):
        slope, intercept, r_value, p_value, std_err = linregress(log_freqs, log_psd[ch_idx])
        spectral_exponents.append(-slope)  # 取负值，因为 1/f 的斜率是负的
    
    spectral_exponents = np.array(spectral_exponents)
    print(f"  频谱指数: 平均 = {spectral_exponents.mean():.3f} ± {spectral_exponents.std():.3f}")
    
    # 3. 保存特征
    features = {
        "subject": sub_id,
        "session": ses,
        "n_epochs": len(epochs),
        "band_powers": band_powers,
        "spectral_exponents": spectral_exponents.tolist(),
        "channels": epochs.ch_names
    }
    
    return features


def generate_qc_report(raw, epochs, ica, features, sub_id, ses):
    """生成 QC 报告"""
    print(f"\n生成 QC 报告 {sub_id}/{ses}...")
    
    # 创建简化的 QC 图表（避免使用 MNE 的复杂绘图函数）
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"QC Report: {sub_id} / {ses}", fontsize=16, fontweight='bold')
    
    # 1. 频谱功率柱状图
    freq_bands = ["delta", "theta", "alpha", "beta", "gamma"]
    band_powers = [features["band_powers"][band].mean() for band in freq_bands]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    axes[0, 0].bar(freq_bands, band_powers, color=colors, alpha=0.7)
    axes[0, 0].set_ylabel("Power (μV²/Hz)")
    axes[0, 0].set_title("Band Powers (Mean)")
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. 频谱指数分布
    axes[0, 1].hist(features["spectral_exponents"], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 1].set_xlabel("Spectral Exponent")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title(f"Spectral Exponent\nMean={np.mean(features['spectral_exponents']):.3f}")
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. 通道位置图
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_locs = montage.get_positions()["ch_pos"]
    x = [pos[0] for pos in ch_locs.values()]
    y = [pos[1] for pos in ch_locs.values()]
    axes[0, 2].scatter(x, y, s=50, alpha=0.6, c='steelblue')
    axes[0, 2].set_aspect("equal")
    axes[0, 2].set_title(f"Channel Locations\n(n={len(epochs.ch_names)})")
    axes[0, 2].axis('off')
    
    # 4. 各通道频谱功率热图
    # 计算每个通道的总功率
    channel_powers = []
    for ch_idx, ch_name in enumerate(features["channels"]):
        total_power = sum(features["band_powers"][band][ch_idx] for band in freq_bands)
        channel_powers.append(total_power)
    
    # 重塑为 8x8 网格（61通道，补零到64）
    n_channels = len(channel_powers)
    grid_size = int(np.ceil(np.sqrt(n_channels)))
    padded_powers = np.zeros(grid_size * grid_size)
    padded_powers[:n_channels] = channel_powers
    power_grid = padded_powers[:grid_size*grid_size].reshape(grid_size, grid_size)
    
    im = axes[1, 0].imshow(power_grid, cmap='viridis', aspect='auto')
    axes[1, 0].set_title("Channel Power Heatmap")
    axes[1, 0].set_xlabel("Channel Index")
    axes[1, 0].set_ylabel("Channel Index")
    plt.colorbar(im, ax=axes[1, 0], label='Total Power')
    
    # 5. Epochs 质量统计
    data = epochs.get_data()
    epoch_stds = data.std(axis=(1, 2))  # 每个 epoch 的标准差
    axes[1, 1].plot(epoch_stds, alpha=0.6, color='steelblue')
    axes[1, 1].axhline(epoch_stds.mean(), color='red', linestyle='--', label=f'Mean: {epoch_stds.mean():.2f}')
    axes[1, 1].set_xlabel("Epoch Index")
    axes[1, 1].set_ylabel("Std (μV)")
    axes[1, 1].set_title(f"Epoch Quality\n(n={len(epochs)} epochs)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # 6. 基本信息表格
    info_text = f"""
Subject: {sub_id}
Session: {ses}
Channels: {len(epochs.ch_names)}
Sampling Rate: {epochs.info['sfreq']} Hz
Duration: {len(epochs) * 2.0:.1f} s
Epochs: {len(epochs)}
Epoch Duration: 2.0 s

Band Powers:
  Delta (1-4 Hz): {band_powers[0]:.2e}
  Theta (4-8 Hz): {band_powers[1]:.2e}
  Alpha (8-13 Hz): {band_powers[2]:.2e}
  Beta (13-30 Hz): {band_powers[3]:.2e}
  Gamma (30-45 Hz): {band_powers[4]:.2e}

Spectral Exponent: {np.mean(features['spectral_exponents']):.3f}
"""
    axes[1, 2].text(0.05, 0.5, info_text, fontsize=9, verticalalignment='center',
                    fontfamily='monospace', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title("Summary")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存图表
    fig_path = RESULTS_DIR / "figures" / f"qc_{sub_id}_{ses}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ QC 报告已保存: {fig_path}")
    
    return fig_path


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Phase 2: Pilot 预处理")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建必要的目录
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (DERIVATIVES_DIR / "preprocessed").mkdir(parents=True, exist_ok=True)
    
    # 读取 pilot 数据清单
    pilot_df = pd.read_csv(PROJECT_ROOT / "data" / "metadata" / "pilot_subjects.csv")
    
    print(f"\n处理 {len(pilot_df)} 个被试-session 组合...")
    
    # 处理每个被试
    all_features = []
    qc_reports = []
    
    for idx, row in pilot_df.iterrows():
        sub_id = row["subject"]
        ses = row["session"]
        
        print(f"\n{'='*80}")
        print(f"处理 {sub_id}/{ses} ({idx+1}/{len(pilot_df)})")
        print(f"{'='*80}")
        
        # 1. 加载数据
        raw = load_eeg_data(sub_id, ses)
        if raw is None:
            continue
        
        # 2. 预处理
        raw, epochs, ica = preprocess_eeg(raw, sub_id, ses)
        
        # 3. 提取特征
        features = extract_features(epochs, sub_id, ses)
        all_features.append(features)
        
        # 4. 生成 QC 报告
        fig_path = generate_qc_report(raw, epochs, ica, features, sub_id, ses)
        qc_reports.append({
            "subject": sub_id,
            "session": ses,
            "figure": str(fig_path)
        })
        
        # 5. 保存预处理后的数据
        preprocessed_dir = DERIVATIVES_DIR / "preprocessed" / sub_id / ses
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存 epochs
        epochs.save(preprocessed_dir / f"{sub_id}_{ses}-epo.fif", overwrite=True, verbose=False)
        
        print(f"  ✓ 预处理数据已保存: {preprocessed_dir}")
    
    # 保存所有特征
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(RESULTS_DIR / "tables" / "pilot_features.csv", index=False)
    print(f"\n✓ 特征数据已保存: {RESULTS_DIR / 'tables' / 'pilot_features.csv'}")
    
    # 保存 QC 报告清单
    qc_df = pd.DataFrame(qc_reports)
    qc_df.to_csv(RESULTS_DIR / "tables" / "qc_reports.csv", index=False)
    print(f"✓ QC 报告清单已保存: {RESULTS_DIR / 'tables' / 'qc_reports.csv'}")
    
    print("\n" + "=" * 80)
    print("Phase 2 完成!")
    print("=" * 80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n处理的被试数: {len(all_features)}")
    print(f"生成的 QC 报告数: {len(qc_reports)}")


if __name__ == "__main__":
    main()
