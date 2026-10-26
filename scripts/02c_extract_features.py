#!/usr/bin/env python3
"""
Phase 2c (Refined): 高级电生理特征提取
修复与升级项：
1. 单次计算全脑 Welch PSD (0.5-40 Hz)，相比原脚本重复调用 15 次提速 10 倍以上。
2. 修正相对功率计算公式（原脚本错误地将各频带 log-power 相加当作总功率，现修正为线性积分比值与对数差值）。
3. 引入 FOOOF/Specparam 思想：在拟合 1/f 非周期性指数 (Spectral Exponent) 与截距 (Offset) 时，
   剔除 7-14 Hz Alpha 振荡隆起区，避免强 Alpha 峰扭曲 1/f 斜率估计，同时提取剥离非周期背景后的纯周期性振荡功率 (Periodic Power)。
4. 新增时域 Hjorth 参数 (Activity, Mobility, Complexity) 与个体 Alpha 重心频率 (CoG)。
5. 完整保留 session_order 元数据，生成干净的 session-level 与 paired-change Parquet 主表。
"""

import os
import json
import warnings
from pathlib import Path
from datetime import datetime

import mne
import numpy as np
import pandas as pd
import yaml
from scipy import integrate

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "analysis_config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DERIVATIVES_DIR = PROJECT_ROOT / config['paths']['derivatives'] / 'preprocessed_v3_conservative'
FEATURES_DIR = PROJECT_ROOT / config['paths']['features'] / 'v3_conservative'
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS = config['analysis']


def get_roi_indices(ch_names, roi_channels):
    """大小写不敏感匹配 ROI 通道索引"""
    if roi_channels is None:
        return list(range(len(ch_names)))
    lower_map = {ch.lower(): idx for idx, ch in enumerate(ch_names)}
    return [lower_map[rc.lower()] for rc in roi_channels if rc.lower() in lower_map]


def fit_aperiodic_and_periodic(freqs, mean_psd_1d, fit_range=(1.0, 40.0), exclude_alpha=(7.0, 14.0)):
    """
    参数化功率谱分离：
    1. 剔除 7-14 Hz 强振荡区，在 log10(f) - log10(PSD) 空间拟合 1/f 背景线:
       L(f) = offset + slope * log10(f)  (其中 exponent = -slope)
    2. 计算剥离 1/f 背景后的周期性剩余谱:
       Periodic(f) = log10(PSD(f)) - L(f)
    """
    mask_fit = (freqs >= fit_range[0]) & (freqs <= fit_range[1]) & (
        (freqs < exclude_alpha[0]) | (freqs > exclude_alpha[1])
    )
    if np.sum(mask_fit) < 5:
        mask_fit = (freqs >= fit_range[0]) & (freqs <= fit_range[1])

    log_f = np.log10(freqs[mask_fit])
    log_p = np.log10(mean_psd_1d[mask_fit] + 1e-16)
    slope, offset = np.polyfit(log_f, log_p, 1)

    # 全频段非周期背景预测与周期性残差
    full_log_f = np.log10(np.maximum(freqs, 0.1))
    aperiodic_fit = offset + slope * full_log_f
    periodic_residual = np.log10(mean_psd_1d + 1e-16) - aperiodic_fit

    return {
        'slope': float(slope),          # 负值，越接近 0 表示谱越平坦 (E/I 比升高)
        'exponent': float(-slope),      # 正值，传统 FOOOF exponent χ
        'offset': float(offset),
        'periodic_residual': periodic_residual
    }


def compute_hjorth_parameters(data_3d):
    """
    计算时域 Hjorth 参数 (跨 epochs 和 channels 平均)
    data_3d: (n_epochs, n_channels, n_times)
    """
    dx = np.diff(data_3d, axis=-1)
    ddx = np.diff(dx, axis=-1)

    var_0 = np.var(data_3d, axis=-1) + 1e-16
    var_d1 = np.var(dx, axis=-1) + 1e-16
    var_d2 = np.var(ddx, axis=-1) + 1e-16

    activity = np.mean(np.log10(var_0))
    mobility = np.sqrt(var_d1 / var_0)
    complexity = np.sqrt(var_d2 / var_d1) / (mobility + 1e-12)

    return float(activity), float(np.mean(mobility)), float(np.mean(complexity))


def extract_features_from_session(row):
    """从单个清洗后的 session 提取多维特征"""
    subject_str = str(row['subject'])
    session_id = int(row['session'])
    condition = str(row['condition'])
    session_order = str(row.get('session_order', 'NS->SD'))

    epochs_file = DERIVATIVES_DIR / subject_str / f"ses-{session_id}" / f"{subject_str}_ses-{session_id}_cleaned-epo.fif"
    if not epochs_file.exists():
        print(f"  ✗ Epochs 文件不存在: {epochs_file}")
        return None

    epochs = mne.read_epochs(epochs_file, preload=True, verbose=False)
    epochs.info['bads'] = []  # 确保所有插值修复后的通道均参与计算

    # 一次性计算 0.5 - 40 Hz Welch PSD
    psd_obj = epochs.compute_psd(fmin=0.5, fmax=40.0, method='welch', verbose=False)
    psd_data = psd_obj.get_data()  # shape: (n_epochs, n_channels, n_freqs)
    freqs = psd_obj.freqs
    ch_names = epochs.ch_names

    # 先对每个通道跨 epochs 求中位数/均值功率谱 (n_channels, n_freqs)
    ch_psd = np.mean(psd_data, axis=0)

    features = {
        'subject': subject_str,
        'session': session_id,
        'condition': condition,
        'session_order': session_order,
        'n_epochs_retained': int(row['n_epochs_retained']),
        'clean_duration_sec': float(row['clean_duration_sec']),
        'n_bad_channels': int(row.get('n_interpolated_channels', 0)),
        'n_ica_excluded': int(row.get('n_ica_excluded', 0)),
    }

    roi_defs = ANALYSIS['roi']
    freq_bands = ANALYSIS['freq_bands']

    # 全脑平均 PSD (n_freqs,)
    global_psd_1d = np.mean(ch_psd, axis=0)
    total_linear_power = np.mean(global_psd_1d[(freqs >= 0.5) & (freqs <= 40.0)]) + 1e-16

    # 1. 全脑绝对对数功率与正确的相对功率
    for band, (fmin, fmax) in freq_bands.items():
        b_mask = (freqs >= fmin) & (freqs <= fmax)
        band_linear = np.mean(global_psd_1d[b_mask]) + 1e-16
        features[f'global_{band}_logpower'] = float(np.log10(band_linear))
        features[f'global_{band}_relative_power'] = float(band_linear / total_linear_power)
        features[f'global_{band}_logrel_power'] = float(np.log10(band_linear / total_linear_power))

    # 2. ROI 频带功率与非周期/周期特征
    def calc_roi_band_logpower(roi_name, band_name):
        idx = get_roi_indices(ch_names, roi_defs[roi_name])
        if not idx:
            return np.nan
        fmin, fmax = freq_bands[band_name]
        b_mask = (freqs >= fmin) & (freqs <= fmax)
        return float(np.log10(np.mean(ch_psd[idx][:, b_mask]) + 1e-16))

    features['posterior_alpha_logpower'] = calc_roi_band_logpower('posterior', 'alpha')
    features['occipital_alpha_logpower'] = features['posterior_alpha_logpower']
    features['frontal_theta_logpower'] = calc_roi_band_logpower('frontal', 'theta')
    features['central_theta_logpower'] = calc_roi_band_logpower('central', 'theta')
    features['frontal_alpha_logpower'] = calc_roi_band_logpower('frontal', 'alpha')
    features['posterior_theta_logpower'] = calc_roi_band_logpower('posterior', 'theta')
    features['central_beta_logpower'] = calc_roi_band_logpower('central', 'beta')

    # 3. 1/f 非周期性斜率 (Spectral Slope / Exponent) 与剥离 1/f 后的纯周期振荡功率
    global_ap = fit_aperiodic_and_periodic(freqs, global_psd_1d)
    # 注意：保持与之前脚本符号约定兼容（slope 为负值，越负表示越陡峭，SD 后变平则数值增大或减少取决于定义）
    # 这里 global_spectral_exponent 保存 log-log 负斜率 (slope)，同时保存 positive_exponent (-slope)
    features['global_spectral_exponent'] = global_ap['slope']
    features['global_aperiodic_exponent_pos'] = global_ap['exponent']
    features['global_aperiodic_offset'] = global_ap['offset']

    post_idx = get_roi_indices(ch_names, roi_defs['posterior'])
    front_idx = get_roi_indices(ch_names, roi_defs['frontal'])

    post_psd_1d = np.mean(ch_psd[post_idx], axis=0) if post_idx else global_psd_1d
    front_psd_1d = np.mean(ch_psd[front_idx], axis=0) if front_idx else global_psd_1d

    post_ap = fit_aperiodic_and_periodic(freqs, post_psd_1d)
    front_ap = fit_aperiodic_and_periodic(freqs, front_psd_1d)

    features['posterior_spectral_exponent'] = post_ap['slope']
    features['frontal_spectral_exponent'] = front_ap['slope']

    alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)
    theta_mask = (freqs >= 4.0) & (freqs <= 8.0)
    features['posterior_alpha_periodic_power'] = float(np.mean(post_ap['periodic_residual'][alpha_mask]))
    features['frontal_theta_periodic_power'] = float(np.mean(front_ap['periodic_residual'][theta_mask]))

    # 4. 后部 Alpha 峰值频率 (IAF) 与重心频率 (CoG)
    alpha_freqs = freqs[alpha_mask]
    post_alpha_psd = post_psd_1d[alpha_mask]
    features['posterior_alpha_peak_frequency'] = float(alpha_freqs[np.argmax(post_alpha_psd)])
    features['posterior_alpha_cog_frequency'] = float(
        np.sum(alpha_freqs * post_alpha_psd) / (np.sum(post_alpha_psd) + 1e-16)
    )

    # 5. 衍生比值指标
    features['frontal_theta_posterior_alpha_ratio'] = (
        features['frontal_theta_logpower'] - features['posterior_alpha_logpower']
    )
    features['global_theta_alpha_ratio'] = (
        features['global_theta_logpower'] - features['global_alpha_logpower']
    )

    # 6. 时域 Hjorth 特征
    ep_data = epochs.get_data()
    act, mob, comp = compute_hjorth_parameters(ep_data)
    features['global_hjorth_activity'] = act
    features['global_hjorth_mobility'] = mob
    features['global_hjorth_complexity'] = comp

    return features


def create_master_table(all_features):
    session_df = pd.DataFrame(all_features)
    session_file = FEATURES_DIR / 'eeg_session_features.parquet'
    session_df.to_parquet(session_file, index=False)
    print(f"✓ Session-level 特征表已保存 ({len(session_df)} 行): {session_file}")

    subjects = sorted(session_df['subject'].unique())
    change_records = []
    feature_cols = [
        col for col in session_df.columns
        if col not in ['subject', 'session', 'condition', 'session_order']
    ]

    for subject in subjects:
        sub_df = session_df[session_df['subject'] == subject]
        ns_df = sub_df[sub_df['condition'] == 'NS']
        sd_df = sub_df[sub_df['condition'] == 'SD']
        if len(ns_df) == 0 or len(sd_df) == 0:
            print(f"  ⚠ {subject} 缺少配对条件，跳过")
            continue

        ns_row = ns_df.iloc[0]
        sd_row = sd_df.iloc[0]
        rec = {
            'subject': subject,
            'session_order': ns_row['session_order'],
        }
        for col in feature_cols:
            rec[f'{col}_ns'] = ns_row[col]
            rec[f'{col}_sd'] = sd_row[col]
            rec[f'{col}_change'] = sd_row[col] - ns_row[col]
        change_records.append(rec)

    change_df = pd.DataFrame(change_records)
    change_file = FEATURES_DIR / 'eeg_change_features.parquet'
    change_df.to_parquet(change_file, index=False)
    print(f"✓ Paired change 特征表已保存 ({len(change_df)} 对完整被试): {change_file}")
    return session_df, change_df


def main():
    print("=" * 80)
    print("Phase 2c: 提取规范化 EEG 频谱、非周期 1/f 与时域特征")
    print("=" * 80)

    qc_file = PROJECT_ROOT / "data" / "derivatives" / "qc_v3_conservative" / "qc_summary_v3_conservative.csv"
    qc_df = pd.read_csv(qc_file)
    success_df = qc_df[qc_df['status'] == 'success']
    print(f"待提取成功 session 数: {len(success_df)}")

    all_features = []
    for _, row in success_df.iterrows():
        feat = extract_features_from_session(row)
        if feat:
            all_features.append(feat)

    print(f"\n成功提取 {len(all_features)}/{len(success_df)} 个 session 特征")
    if all_features:
        create_master_table(all_features)


if __name__ == '__main__':
    main()
