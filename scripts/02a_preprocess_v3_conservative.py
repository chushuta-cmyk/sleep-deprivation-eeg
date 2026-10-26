#!/usr/bin/env python3
"""
Phase 2a v3 (Refined): 鲁棒版 EEG 预处理与质量控制 (QC) 流程
修复项：
1. 修正 ds004902 数据集条件标签映射：根据官方 README 与 .set 内部原始文件名验证，
   ses-1 恒为 NS（正常睡眠），ses-2 恒为 SD（睡眠剥夺），SessionOrder 为时间平衡顺序协变量。
2. 兼容 .fdt 采样点数截断文件（如 sub-44 ses-2 含 236.9 秒完整有效数据）。
3. 统一重采样至 250 Hz（消除 5000 Hz 与 500 Hz 混杂）。
4. 多证据坏道检测 + 球面样条插值后执行 reset_bads=True，防止下游 ROI 特征提取时因 info['bads'] 残留导致通道丢失或崩溃。
5. 真正执行 ICA 眼电伪迹检测 (find_bads_eog, 基于额极导联 Fp1/Fp2/Fpz) 与 ica.apply(raw) 信号重构。
6. 配合眼电剔除后的 150 µV 标准峰峰值拒绝阈值与自适应回退保护，实现全样本高质量保留。
"""

import os
import sys
import json
import hashlib
import warnings
from pathlib import Path
from datetime import datetime

import mne
import scipy.io
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# 加载配置
CONFIG_PATH = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/sleep_deprivation_eeg/config/analysis_config.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

PROJECT_ROOT = Path("/Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/sleep_deprivation_eeg")
RAW_DIR = PROJECT_ROOT / config['paths']['raw_data']
DERIVATIVES_DIR = PROJECT_ROOT / config['paths']['derivatives'] / 'preprocessed_v3_conservative'
QC_DIR = PROJECT_ROOT / config['paths']['derivatives'] / 'qc_v3_conservative'

DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_VERSION = "v3_conservative_refined"
ANALYSIS_DATE = datetime.now().strftime("%Y-%m-%d")


def compute_file_checksum(filepath):
    """计算文件 SHA256 校验和"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_raw_eeglab_robust(set_file):
    """
    鲁棒读取 EEGLAB .set/.fdt 文件。
    当 .fdt 文件末尾因传输截断导致实际采样点少于 .set 头文件记录的 pnts 时（如 sub-44 ses-2），
    自动按实际完整采样点数重构 RawArray，避免直接崩溃丢弃有效被试。
    """
    try:
        raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
        return raw, False
    except Exception as e:
        fdt_file = Path(set_file).with_suffix('.fdt')
        if not fdt_file.exists():
            raise e
        print(f"    ⚠ 标准读取触发头信息长度不匹配 ({e})，启用二进制自适应修复...")
        mat = scipy.io.loadmat(str(set_file), struct_as_record=False, squeeze_me=True)
        eeg = mat['EEG']
        n_chan = int(eeg.nbchan)
        sfreq = float(eeg.srate)
        ch_names = [str(ch.labels) for ch in eeg.chanlocs]

        raw_bytes = np.fromfile(str(fdt_file), dtype=np.float32)
        n_samples = len(raw_bytes) // n_chan
        if n_samples < int(sfreq * 30):
            raise RuntimeError(f"有效采样点不足 30 秒 ({n_samples / sfreq:.1f}s)")

        # EEGLAB .fdt 默认以 (n_chan, n_samples) Fortran order (column-major) 存储
        data = raw_bytes[:n_samples * n_chan].reshape((n_chan, n_samples), order='F')
        # EEGLAB 单位为 µV，MNE RawArray 单位为 V
        data = data.astype(np.float64) * 1e-6

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        print(f"    ✓ 成功恢复 {n_samples} 个采样点 ({n_samples / sfreq:.1f} 秒, {sfreq:.0f} Hz)")
        return raw, True


def detect_bad_channels_multi_evidence(raw, max_bad_fraction=0.15):
    """
    多证据坏道检测：
    结合平坦度 (Flat)、鲁棒方差/峰峰值异常 (Variance/PTP)、全脑平均相关性 (Low Correlation) 与 PSD 功率异常。
    严格限制最大插值通道比例（默认 <= 15%，即 61 导最多插值 9 导），防止过度插值抹平空间拓扑。
    """
    data = raw.get_data()
    ch_names = raw.ch_names
    n_ch = len(ch_names)

    # 1. 平坦通道检测
    std_per_channel = np.std(data, axis=1)
    ptp_per_channel = np.ptp(data, axis=1)
    flat_channels = np.where(std_per_channel < 1e-7)[0].tolist()

    # 2. 高方差 / 高 PTP 候选
    median_std = np.median(std_per_channel)
    mad_std = np.median(np.abs(std_per_channel - median_std)) + 1e-12
    robust_z_std = 0.6745 * (std_per_channel - median_std) / mad_std

    median_ptp = np.median(ptp_per_channel)
    mad_ptp = np.median(np.abs(ptp_per_channel - median_ptp)) + 1e-12
    robust_z_ptp = 0.6745 * (ptp_per_channel - median_ptp) / mad_ptp

    high_var_channels = np.where((robust_z_std > 5.0) | (robust_z_ptp > 5.0))[0].tolist()
    candidate_bads_variance = sorted(list(set(flat_channels + high_var_channels)))

    # 3. 低相关性候选
    corr_matrix = np.corrcoef(data)
    np.fill_diagonal(corr_matrix, 0)
    # 对比每个通道与其相关性最高的前 25% 邻居的平均相关性（避免受对侧半球影响）
    sorted_corr = np.sort(np.abs(corr_matrix), axis=1)[:, -max(5, n_ch // 4):]
    mean_top_corr = np.mean(sorted_corr, axis=1)
    candidate_bads_lowcorr = np.where(mean_top_corr < 0.35)[0].tolist()

    # 4. PSD 异常候选
    candidate_bads_psd = []
    robust_z_power = np.zeros(n_ch)
    try:
        psd = raw.compute_psd(fmin=1.0, fmax=40.0, verbose=False)
        psd_data = psd.get_data()
        total_log_power = np.mean(np.log10(psd_data + 1e-15), axis=1)
        med_p = np.median(total_log_power)
        mad_p = np.median(np.abs(total_log_power - med_p)) + 1e-12
        robust_z_power = 0.6745 * (total_log_power - med_p) / mad_p
        candidate_bads_psd = np.where(np.abs(robust_z_power) > 4.0)[0].tolist()
    except Exception:
        pass

    # 多证据融合判定
    var_set = set(candidate_bads_variance)
    psd_set = set(candidate_bads_psd)
    corr_set = set(candidate_bads_lowcorr)

    # 满足任意两项异常，或极端方差 (robust_z_std > 8.0)，或完全平坦
    extreme_var = set(np.where(robust_z_std > 8.0)[0].tolist())
    multi_evidence = set(flat_channels) | extreme_var | (var_set & psd_set) | (corr_set & psd_set) | (var_set & corr_set)

    # 如果超出最大允许坏道数，按异常综合得分排序，仅插值最严重的通道
    max_bads = max(1, int(np.floor(n_ch * max_bad_fraction)))
    anomaly_score = np.abs(robust_z_std) + np.abs(robust_z_power) + np.maximum(0, (0.5 - mean_top_corr) * 10)
    sorted_candidates = sorted(list(multi_evidence), key=lambda idx: anomaly_score[idx], reverse=True)
    final_bads_idx = sorted_candidates[:max_bads]
    final_bads_names = [ch_names[i] for i in final_bads_idx]

    all_candidates_union = sorted(list(var_set | psd_set | corr_set))
    return {
        'candidate_bads_all': [ch_names[i] for i in all_candidates_union],
        'candidate_bads_variance': [ch_names[i] for i in candidate_bads_variance],
        'candidate_bads_lowcorr': [ch_names[i] for i in candidate_bads_lowcorr],
        'candidate_bads_psd': [ch_names[i] for i in candidate_bads_psd],
        'final_interpolated_bads': final_bads_names
    }


def preprocess_subject(subject_id, session_id, condition, session_order):
    """预处理单个被试的单个 session"""
    print(f"\n{'='*70}")
    print(f"处理: {subject_id} | ses-{session_id} | 条件: {condition} (顺序: {session_order})")
    print(f"{'='*70}")

    possible_paths = [
        RAW_DIR / subject_id / f"ses-{session_id}" / "eeg" / f"{subject_id}_ses-{session_id}_task-eyesopen_eeg.set",
        RAW_DIR / subject_id / f"ses-{session_id}" / "eeg" / f"{subject_id}_ses-{session_id}_task-rest_run-1_eeg.set",
    ]

    set_file = next((p for p in possible_paths if p.exists()), None)
    if set_file is None:
        print(f"  ✗ 文件不存在")
        return None

    file_checksum = compute_file_checksum(set_file)
    target_sfreq = float(config['preprocessing']['resample_hz'])

    qc_info = {
        'pipeline_version': PIPELINE_VERSION,
        'analysis_date': ANALYSIS_DATE,
        'subject': subject_id,
        'session': session_id,
        'condition': condition,
        'session_order': session_order,
        'timestamp': datetime.now().isoformat(),
        'raw_file': str(set_file),
        'raw_file_checksum': file_checksum,
    }

    try:
        # Step 1: 鲁棒读取原始数据
        raw, was_repaired = load_raw_eeglab_robust(set_file)
        qc_info['raw_duration_sec'] = float(raw.times[-1])
        qc_info['n_channels_raw'] = len(raw.ch_names)
        qc_info['sfreq_original'] = float(raw.info['sfreq'])
        qc_info['fdt_repaired'] = was_repaired

        # Step 2: 统一重采样至 250 Hz（必须在滤波/ICA前统一采样率）
        if abs(raw.info['sfreq'] - target_sfreq) > 1e-2:
            print(f"  重采样: {raw.info['sfreq']:.0f} Hz -> {target_sfreq:.0f} Hz")
            raw.resample(target_sfreq, npad="auto", verbose=False)
        qc_info['sfreq'] = float(raw.info['sfreq'])

        # Step 3: 设置标准 10-20 电极坐标
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False, on_missing='warn')

        # Step 4: 陷波滤波 (50 Hz 工频噪声) + 预滤波副本 (1-40 Hz) 用于坏道检测与 ICA
        notch_freq = float(config['preprocessing']['notch_freq'])
        raw.notch_filter(freqs=[notch_freq], notch_widths=2.0, verbose=False)

        raw_hp = raw.copy().filter(l_freq=1.0, h_freq=40.0, verbose=False)

        # Step 5: 多证据坏道检测与球面样条插值 (关键修复: reset_bads=True)
        bad_info = detect_bad_channels_multi_evidence(
            raw_hp,
            max_bad_fraction=float(config['preprocessing'].get('max_bad_channel_fraction', 0.15))
        )
        final_bads = bad_info['final_interpolated_bads']
        qc_info['bad_channels_candidate'] = str(bad_info['candidate_bads_all'])
        qc_info['n_bad_channels_candidate'] = len(bad_info['candidate_bads_all'])
        qc_info['bad_channel_fraction_candidate'] = len(bad_info['candidate_bads_all']) / len(raw.ch_names)

        if final_bads:
            raw.info['bads'] = list(final_bads)
            raw.interpolate_bads(reset_bads=True, verbose=False)
            raw_hp.info['bads'] = list(final_bads)
            raw_hp.interpolate_bads(reset_bads=True, verbose=False)
            qc_info['n_interpolated_channels'] = len(final_bads)
            qc_info['interpolated_channels'] = str(final_bads)
            print(f"  插值修复 {len(final_bads)} 个坏道 (reset_bads=True): {final_bads}")
        else:
            qc_info['n_interpolated_channels'] = 0
            qc_info['interpolated_channels'] = "[]"

        # Step 6: 全脑平均参考 (Average Reference)
        raw.set_eeg_reference('average', verbose=False)
        raw_hp.set_eeg_reference('average', verbose=False)

        # Step 7: 分析频段滤波 (0.5 - 40 Hz)
        raw.filter(
            l_freq=float(config['preprocessing']['l_freq_analysis']),
            h_freq=float(config['preprocessing']['h_freq_analysis']),
            verbose=False
        )

        # Step 8: ICA 拟合 + 自动眼电 (EOG) 伪迹识别与剔除 (关键修复: 真正调用 ica.apply)
        eog_Grouping = [ch for ch in ['Fp1', 'Fp2', 'Fpz', 'AF3', 'AF4', 'AF7', 'AF8'] if ch in raw.ch_names]
        try:
            n_comp = min(20, len(raw.ch_names) - len(final_bads) - 1)
            ica = mne.preprocessing.ICA(
                n_components=n_comp,
                method='fastica',
                random_state=42,
                max_iter=500,
                verbose=False
            )
            ica.fit(raw_hp, verbose=False)

            eog_indices = []
            if eog_Grouping:
                for eog_ch in eog_Grouping[:3]:  # 使用 Fp1, Fp2, Fpz 作为虚拟眼电参考
                    inds, _ = ica.find_bads_eog(raw_hp, ch_name=eog_ch, threshold=3.0, verbose=False)
                    eog_indices.extend(inds)
            eog_indices = sorted(list(set(eog_indices)))[:4]  # 最多剔除前 4 个显著眼电成分

            if eog_indices:
                ica.exclude = eog_indices
                ica.apply(raw, verbose=False)
                print(f"  ICA 自动剔除 {len(eog_indices)} 个眼电成分: {eog_indices}")

            qc_info['n_ica_components'] = int(ica.n_components_)
            qc_info['ica_used'] = True
            qc_info['ica_excluded_components'] = str(eog_indices)
            qc_info['n_ica_excluded'] = len(eog_indices)
        except Exception as e:
            print(f"  ⚠ ICA 警告 ({e})，继续执行分段")
            qc_info['n_ica_components'] = 0
            qc_info['ica_used'] = False
            qc_info['ica_excluded_components'] = "[]"
            qc_info['n_ica_excluded'] = 0

        # Step 9: 固定长度切段 (2.0s epochs)
        epoch_len = float(config['preprocessing']['epoch_length_sec'])
        epochs_all = mne.make_fixed_length_epochs(
            raw,
            duration=epoch_len,
            overlap=float(config['preprocessing']['epoch_overlap']),
            preload=True,
            verbose=False
        )
        n_initial = len(epochs_all)
        qc_info['n_epochs_initial'] = n_initial

        # Step 10: Epoch 伪迹拒绝 (标准 150 µV，若保留不足则自适应放宽至 200 µV 或保留最优 50% epochs)
        base_thresh_uv = float(config['preprocessing']['reject_eeg_uv'])
        flat_thresh_uv = float(config['preprocessing']['flat_eeg_uv'])

        epochs = epochs_all.copy()
        epochs.drop_bad(
            reject=dict(eeg=base_thresh_uv * 1e-6),
            flat=dict(eeg=flat_thresh_uv * 1e-6),
            verbose=False
        )
        used_policy = f"eeg={base_thresh_uv:.0f}uV"

        if len(epochs) * epoch_len < 60.0:
            # 尝试放宽至 200 µV（容忍残余微弱眼动）
            epochs_relaxed = epochs_all.copy()
            epochs_relaxed.drop_bad(
                reject=dict(eeg=200.0 * 1e-6),
                flat=dict(eeg=flat_thresh_uv * 1e-6),
                verbose=False
            )
            if len(epochs_relaxed) > len(epochs):
                epochs = epochs_relaxed
                used_policy = "eeg=200uV_adaptive"

        if len(epochs) * epoch_len < 30.0 and n_initial >= 30:
            # 极端情况：基于每个 epoch 的最大峰峰值排序，保留最干净的 60% epochs
            ep_data = epochs_all.get_data()
            ep_max_ptp = np.percentile(np.ptp(ep_data, axis=2), 95, axis=1)
            keep_n = max(25, int(n_initial * 0.60))
            keep_idx = np.argsort(ep_max_ptp)[:keep_n]
            epochs = epochs_all[np.sort(keep_idx)]
            used_policy = "quantile_top60pct_clean"

        # 确保 info['bads'] 为空，所有保留通道均可参与下游 ROI 计算
        epochs.info['bads'] = []

        n_retained = len(epochs)
        qc_info['n_epochs_rejected'] = n_initial - n_retained
        qc_info['n_epochs_retained'] = n_retained
        qc_info['rejection_fraction'] = (n_initial - n_retained) / n_initial if n_initial > 0 else 0.0
        qc_info['clean_duration_sec'] = n_retained * epoch_len
        qc_info['epoch_rejection_policy'] = used_policy

        print(f"  保留 {n_retained}/{n_initial} epochs ({qc_info['clean_duration_sec']:.0f}s, 策略: {used_policy})")

        if qc_info['clean_duration_sec'] < 30.0:
            qc_info['status'] = 'rejected_insufficient_data'
            qc_info['epochs_file'] = np.nan
            qc_info['error_message'] = np.nan
            return qc_info

        # 保存清洗后的 epochs
        output_dir = DERIVATIVES_DIR / subject_id / f"ses-{session_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        epochs_file = output_dir / f"{subject_id}_ses-{session_id}_cleaned-epo.fif"
        epochs.save(epochs_file, overwrite=True, verbose=False)

        qc_info['epochs_file'] = str(epochs_file)
        qc_info['status'] = 'success'
        qc_info['error_message'] = np.nan
        return qc_info

    except Exception as e:
        import traceback
        traceback.print_exc()
        qc_info['status'] = 'error'
        qc_info['error_message'] = str(e)
        return qc_info


def main():
    print("=" * 80)
    print(f"Phase 2a: 鲁棒版 EEG 预处理与 QC ({PIPELINE_VERSION})")
    print("=" * 80)

    participants_file = RAW_DIR / "participants.tsv"
    participants = pd.read_csv(participants_file, sep='\t')

    existing_subjects = sorted([
        item.name for item in RAW_DIR.iterdir()
        if item.is_dir() and item.name.startswith('sub-')
    ])

    qc_file = QC_DIR / "qc_summary_v3_conservative.csv"
    cached_qc = {}
    if qc_file.exists():
        try:
            old_df = pd.read_csv(qc_file)
            for _, r in old_df.iterrows():
                if r.get('status') == 'success' and pd.notna(r.get('epochs_file')) and Path(str(r['epochs_file'])).exists():
                    cached_qc[(str(r['subject']), int(r['session']))] = r.to_dict()
            print(f"已加载 {len(cached_qc)} 个已完成会话的缓存预处理记录。")
        except Exception:
            pass

    all_qc = []
    for _, row in participants.iterrows():
        subject_id = row['participant_id']
        if subject_id not in existing_subjects:
            continue

        # 检查是否至少有一个 .set 文件
        set_files = list((RAW_DIR / subject_id).rglob("*.set"))
        if not set_files:
            print(f"跳过 {subject_id}: 目录下无有效 .set 文件")
            continue

        session_order = str(row.get('SessionOrder', 'NS->SD')).strip()

        # 关键修正：在 OpenNeuro ds004902 中，ses-1 恒为 NS，ses-2 恒为 SD
        # (已在数据集 README 及 .set 内部原始文件名验证)
        if (subject_id, 1) in cached_qc:
            all_qc.append(cached_qc[(subject_id, 1)])
        else:
            qc_s1 = preprocess_subject(subject_id, 1, 'NS', session_order)
            if qc_s1:
                all_qc.append(qc_s1)

        if (subject_id, 2) in cached_qc:
            all_qc.append(cached_qc[(subject_id, 2)])
        else:
            qc_s2 = preprocess_subject(subject_id, 2, 'SD', session_order)
            if qc_s2:
                all_qc.append(qc_s2)

    qc_df = pd.DataFrame(all_qc)
    qc_df.to_csv(qc_file, index=False)

    print("\n" + "=" * 80)
    print("QC 汇总统计")
    print("=" * 80)
    print(f"总处理 session 数: {len(qc_df)}")
    print(f"成功 (success): {(qc_df['status'] == 'success').sum()}")
    print(f"失败 (error): {(qc_df['status'] == 'error').sum()}")
    print(f"时长不足 (rejected): {(qc_df['status'] == 'rejected_insufficient_data').sum()}")
    print(f"QC 汇总表已保存至: {qc_file}")


if __name__ == '__main__':
    main()
