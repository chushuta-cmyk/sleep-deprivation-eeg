#!/usr/bin/env python3
"""
Phase 2a: EEG 预处理（修复版）

修复内容：
1. 正确的处理顺序：坏道检测 → 插值 → 平均参考 → ICA
2. 使用 pyprep 进行坏道检测
3. ICA 使用 1Hz 高通副本拟合，应用到分析数据
4. Epoch rejection 使用正确的单位（μV → V）
5. 保存完整的 QC 日志
"""

import os
import json
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats
import yaml

# 项目路径
PROJECT_ROOT = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/sleep_deprivation_eeg")
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DERIVATIVES_DIR = PROJECT_ROOT / "data" / "derivatives"
CONFIG_DIR = PROJECT_ROOT / "config"

# 加载配置
with open(CONFIG_DIR / "analysis_config.yaml") as f:
    config = yaml.safe_load(f)

def detect_bad_channels(raw, method="pyprep"):
    """
    检测坏道
    
    参数:
        raw: MNE Raw 对象
        method: "pyprep" 或 "simple"
    
    返回:
        bad_channels: 坏道列表
    """
    if method == "pyprep":
        try:
            from pyprep.find_noisy_channels import NoisyChannels
            
            noisy = NoisyChannels(raw, random_state=config["preprocessing"]["ica_random_state"])
            # 只使用 ransac 检测坏道，不使用其他过于敏感的方法
            noisy.find_bad_by_ransac()
            noisy.find_bad_by_nan()
            noisy.find_bad_by_deviation()
            bad_channels = noisy.get_bads()
            
            # 如果坏道过多（>20%），只保留最严重的（相关性最低的）
            if len(bad_channels) > int(0.20 * len(raw.ch_names)):
                print(f"  ⚠ pyprep 检测到过多坏道 ({len(bad_channels)}/{len(raw.ch_names)})，使用更保守策略")
                # 回退到简单方法
                return detect_bad_channels(raw, method="simple")
            
            print(f"  pyprep 检测到 {len(bad_channels)} 个坏道: {bad_channels}")
            return bad_channels
        except ImportError:
            print("  ⚠ pyprep 未安装，使用简单方法")
            method = "simple"
    
    if method == "simple":
        # 多指标简单检测
        data = raw.get_data()
        
        # 1. 标准差
        std_per_channel = np.std(data, axis=1)
        
        # 2. 平直通道
        flat_channels = np.where(std_per_channel < 1e-8)[0]
        
        # 3. 高方差通道（robust z-score）
        median_std = np.median(std_per_channel)
        mad_std = np.median(np.abs(std_per_channel - median_std))
        robust_z = 0.6745 * (std_per_channel - median_std) / max(mad_std, 1e-12)
        high_var_channels = np.where(robust_z > 4)[0]
        
        # 4. 低相关通道
        corr_matrix = np.corrcoef(data)
        mean_corr = np.mean(corr_matrix, axis=1)
        low_corr_channels = np.where(mean_corr < 0.3)[0]
        
        # 合并
        bad_indices = np.union1d(flat_channels, high_var_channels)
        bad_indices = np.union1d(bad_indices, low_corr_channels)
        
        bad_channels = [raw.ch_names[i] for i in bad_indices]
        print(f"  简单方法检测到 {len(bad_channels)} 个坏道: {bad_channels}")
        
        return bad_channels


def preprocess_subject(subject_id, session, config):
    """
    预处理单个被试的单个 session
    
    返回:
        epochs: 预处理后的 epochs
        qc_log: QC 日志字典
    """
    # 构建文件路径
    eeg_dir = RAW_DATA_DIR / subject_id / session / "eeg"
    set_files = list(eeg_dir.glob("*_eeg.set"))
    
    if not set_files:
        return None, {"error": "未找到 .set 文件"}
    
    set_file = set_files[0]
    
    # 初始化 QC 日志
    qc_log = {
        "subject": subject_id,
        "session": session,
        "timestamp": datetime.now().isoformat(),
        "raw_file": str(set_file),
    }
    
    try:
        # 1. 读取原始数据
        print(f"  读取: {set_file.name}")
        raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
        qc_log["raw_duration_sec"] = raw.times[-1]
        qc_log["raw_sfreq"] = raw.info["sfreq"]
        qc_log["n_channels_raw"] = len(raw.ch_names)
        
        # 2. 设置通道类型和 montage
        ch_types = {ch: "eeg" for ch in raw.ch_names}
        raw.set_channel_types(ch_types)
        
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="warn")
        
        # 3. 创建 1Hz 高通副本用于 ICA
        raw_ica = raw.copy()
        raw_ica.filter(l_freq=config["preprocessing"]["l_freq_ica"], h_freq=None, verbose=False)
        
        # 4. 检测坏道（在平均参考之前）
        bad_channels = detect_bad_channels(raw, method="pyprep")
        qc_log["bad_channels"] = bad_channels
        qc_log["n_bad_channels"] = len(bad_channels)
        qc_log["bad_channel_fraction"] = len(bad_channels) / len(raw.ch_names)
        
        # 检查坏道比例
        if qc_log["bad_channel_fraction"] > config["preprocessing"]["max_bad_channel_fraction"]:
            print(f"  ⚠ 坏道比例过高 ({qc_log['bad_channel_fraction']:.2%})，标记为需要人工 QC")
            qc_log["needs_manual_qc"] = True
        
        # 5. 标记并插值坏道
        raw.info["bads"] = bad_channels
        raw.interpolate_bads(verbose=False)
        qc_log["n_interpolated_channels"] = len(bad_channels)
        
        # 6. 平均参考（在坏道插值之后）
        raw.set_eeg_reference("average", verbose=False)
        raw_ica.set_eeg_reference("average", verbose=False)
        
        # 7. ICA（使用 1Hz 高通副本拟合）
        print("  拟合 ICA...")
        ica = mne.preprocessing.ICA(
            n_components=config["preprocessing"]["ica_n_components"],
            method="fastica",  # 改用 fastica，不依赖 picard
            random_state=config["preprocessing"]["ica_random_state"],
            max_iter=1000,
            verbose=False
        )
        ica.fit(raw_ica, verbose=False)
        
        qc_log["ica_n_components"] = ica.n_components_
        
        # 8. 检测并剔除伪迹成分
        # 检查是否有 EOG 通道
        eog_channels = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
        
        if eog_channels:
            eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=False)
            exclude_indices = eog_indices
            qc_log["ica_eog_components"] = eog_indices
            print(f"  ICA 剔除眼电成分: {eog_indices}")
        else:
            # 没有 EOG 通道时，使用自动检测
            # 检查前额通道的高频成分（眼动伪迹特征）
            frontal_chs = ["Fp1", "Fp2", "Fz"]
            frontal_chs = [ch for ch in frontal_chs if ch in raw.ch_names]
            
            if frontal_chs:
                # 计算每个 ICA 成分在前额通道的功率
                ica_sources = ica.get_sources(raw_ica)
                frontal_data = ica_sources.pick_channels(frontal_chs).get_data()
                
                # 检测高频成分（>8Hz 功率占比高）
                exclude_indices = []
                for comp_idx in range(ica.n_components_):
                    comp_data = frontal_data[comp_idx]
                    psd = np.abs(np.fft.fft(comp_data)) ** 2
                    freqs = np.fft.fftfreq(len(comp_data), 1/raw.info["sfreq"])
                    
                    # 计算高频比例
                    high_freq_mask = np.abs(freqs) > 8
                    if np.sum(high_freq_mask) > 0:
                        high_freq_power = np.mean(psd[high_freq_mask])
                        total_power = np.mean(psd)
                        high_freq_ratio = high_freq_power / total_power
                        
                        # 如果高频比例异常高，可能是眼动伪迹
                        if high_freq_ratio > 0.7:
                            exclude_indices.append(comp_idx)
                
                qc_log["ica_excluded_components"] = exclude_indices
                print(f"  ICA 自动检测伪迹成分: {exclude_indices}")
            else:
                exclude_indices = []
                qc_log["ica_excluded_components"] = []
                print("  ⚠ 无法自动检测伪迹成分")
        
        # 应用 ICA
        if exclude_indices:
            ica.apply(raw, exclude=exclude_indices, verbose=False)
        qc_log["n_ica_excluded"] = len(exclude_indices)
        
        # 9. 目标频段滤波（0.5-40 Hz）
        raw.filter(
            l_freq=config["preprocessing"]["l_freq_analysis"],
            h_freq=config["preprocessing"]["h_freq_analysis"],
            verbose=False
        )
        
        # 10. 降采样
        if raw.info["sfreq"] > config["preprocessing"]["resample_hz"]:
            raw.resample(config["preprocessing"]["resample_hz"], verbose=False)
        
        # 11. 切分 epochs
        events = mne.make_fixed_length_events(
            raw,
            duration=config["preprocessing"]["epoch_length_sec"],
            overlap=config["preprocessing"]["overlap"]
        )
        
        epochs = mne.Epochs(
            raw,
            events,
            tmin=0,
            tmax=config["preprocessing"]["epoch_length_sec"] - 1.0/raw.info["sfreq"],
            baseline=None,
            preload=True,
            verbose=False
        )
        
        qc_log["n_epochs_initial"] = len(epochs)
        
        # 12. Epoch rejection（使用正确的单位：μV → V）
        reject_threshold_v = config["preprocessing"]["reject_eeg_uv"] * 1e-6
        flat_threshold_v = config["preprocessing"]["flat_eeg_uv"] * 1e-6
        
        epochs.drop_bad(
            reject=dict(eeg=reject_threshold_v),
            flat=dict(eeg=flat_threshold_v)
        )
        
        qc_log["n_epochs_rejected"] = qc_log["n_epochs_initial"] - len(epochs)
        qc_log["n_epochs_retained"] = len(epochs)
        qc_log["rejection_fraction"] = qc_log["n_epochs_rejected"] / qc_log["n_epochs_initial"]
        qc_log["clean_duration_sec"] = len(epochs) * config["preprocessing"]["epoch_length_sec"]
        
        # 检查清洁数据时长
        if qc_log["clean_duration_sec"] < config["preprocessing"]["min_clean_duration_sec"]:
            print(f"  ⚠ 清洁数据时长不足 ({qc_log['clean_duration_sec']:.1f}s)")
            qc_log["insufficient_data"] = True
        
        # 13. 保存预处理后的数据
        output_dir = DERIVATIVES_DIR / "preprocessed" / subject_id / session
        output_dir.mkdir(parents=True, exist_ok=True)
        
        epochs_file = output_dir / f"{subject_id}_{session}-epo.fif"
        epochs.save(epochs_file, overwrite=True, verbose=False)
        
        # 保存 ICA（可选）
        ica_file = output_dir / f"{subject_id}_{session}-ica.fif"
        ica.save(ica_file, overwrite=True, verbose=False)
        
        # 保存 QC 日志
        qc_file = output_dir / f"{subject_id}_{session}_qc.json"
        with open(qc_file, "w") as f:
            json.dump(qc_log, f, indent=2)
        
        print(f"  ✓ 预处理完成: {len(epochs)} epochs, {qc_log['clean_duration_sec']:.1f}s")
        
        return epochs, qc_log
        
    except Exception as e:
        qc_log["error"] = str(e)
        print(f"  ✗ 预处理失败: {e}")
        return None, qc_log


def main():
    """主函数"""
    print("=" * 80)
    print("Phase 2a: EEG 预处理（修复版）")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 读取被试列表
    participants_tsv = pd.read_csv(RAW_DATA_DIR / "participants.tsv", sep="\t")
    
    # 创建输出目录
    (DERIVATIVES_DIR / "preprocessed").mkdir(parents=True, exist_ok=True)
    
    # 处理每个被试的每个 session
    all_qc_logs = []
    
    for idx, row in participants_tsv.iterrows():
        subject_id = row["participant_id"]
        subject_dir = RAW_DATA_DIR / subject_id
        
        if not subject_dir.exists():
            continue
        
        for session in ["ses-1", "ses-2"]:
            print(f"\n处理 {subject_id}/{session}...")
            
            # 检查是否已经处理过
            output_dir = DERIVATIVES_DIR / "preprocessed" / subject_id / session
            epochs_file = output_dir / f"{subject_id}_{session}-epo.fif"
            
            if epochs_file.exists():
                print(f"  ✓ 已存在，跳过")
                # 读取已有的 QC 日志
                qc_file = output_dir / f"{subject_id}_{session}_qc.json"
                if qc_file.exists():
                    with open(qc_file) as f:
                        all_qc_logs.append(json.load(f))
                continue
            
            # 预处理
            epochs, qc_log = preprocess_subject(subject_id, session, config)
            all_qc_logs.append(qc_log)
    
    # 保存汇总 QC 报告
    qc_summary_file = DERIVATIVES_DIR / "preprocessing_qc_summary.json"
    with open(qc_summary_file, "w") as f:
        json.dump(all_qc_logs, f, indent=2)
    
    print("\n" + "=" * 80)
    print("预处理完成!")
    print("=" * 80)
    print(f"总处理数: {len(all_qc_logs)}")
    print(f"成功: {sum(1 for log in all_qc_logs if 'error' not in log)}")
    print(f"失败: {sum(1 for log in all_qc_logs if 'error' in log)}")
    print(f"QC 汇总报告: {qc_summary_file}")


if __name__ == "__main__":
    main()
