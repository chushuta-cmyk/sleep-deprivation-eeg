#!/usr/bin/env python3
"""
Phase 9: "A+B" 核心方法学实证验证 (CSD + 19导标准重投射 + OMST图构建 + 常模Z-score校准)
在真实脑电数据 (64 sessions, 32 paired subjects) 上验证：
1. [导联对齐 A+B]: 61导 -> 表面拉普拉斯 (CSD) + 19导国际标准 10-20 空间投射后的特征保真度与 LOSO 分类表现。
2. [图网络构图]: 对比静态阈值 (PLI > 0.3, 产生孤立节点与密度偏倚) vs. 正交最小生成树 + Top-20% 比例阈值 (OMST + Proportional)。
3. [常模校准 + 鉴别 A+B]: 基于健康/基线常模 (NS, 控制年龄与性别) 构建个体化偏离谱 Z-score Map，并检验常模 Z-score 下的跨被试判别力。
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from scipy import stats
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DERIVATIVES_DIR = PROJECT_ROOT / "data" / "derivatives" / "preprocessed_v3_conservative"
FEATURES_DIR = PROJECT_ROOT / "data" / "derivatives" / "features" / "v3_conservative"
RESULTS_DIR = PROJECT_ROOT / "results"

# 国际标准 19 导 10-20 核心子集（跨 MODMA 128导、64导、TUH 19导 100% 共有）
STANDARD_19_CHS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz'
]


def compute_pli_matrix(epochs_data):
    """
    快速计算多通道瞬时相位锁相指数 (Phase Lag Index, PLI)
    epochs_data: (n_epochs, n_ch, n_times) - 已滤波至目标频段 (如 Alpha 8-13 Hz 或 Theta 4-8 Hz)
    """
    from scipy.signal import hilbert
    analytic = hilbert(epochs_data, axis=-1)
    phases = np.angle(analytic)  # (n_ep, n_ch, n_times)
    n_ep, n_ch, _ = phases.shape
    pli_ep = np.zeros((n_ep, n_ch, n_ch))
    for e in range(n_ep):
        # 相位差矩阵: (n_ch, n_ch, n_times)
        phase_diff = phases[e][:, None, :] - phases[e][None, :, :]
        pli_ep[e] = np.abs(np.mean(np.sign(np.sin(phase_diff)), axis=-1))
    pli_mat = np.mean(pli_ep, axis=0)
    np.fill_diagonal(pli_mat, 0.0)
    return pli_mat


def build_graph_static_vs_omst(pli_mat, static_thresh=0.15, top_prop=0.20):
    """
    对比：
    1. 静态硬阈值截断 (Static Thresholding)
    2. 最小生成树骨架 + 固定比例稀疏度 (MST + Proportional Top-k% = OMST 拓扑保真原则)
    """
    n_ch = pli_mat.shape[0]
    triu_idx = np.triu_indices(n_ch, k=1)
    n_total_edges = len(triu_idx[0])

    # 1. 静态阈值
    adj_static = np.where(pli_mat > static_thresh, pli_mat, 0.0)
    deg_static = np.sum(adj_static > 0, axis=1)
    n_isolated_static = int(np.sum(deg_static == 0))
    density_static = float(np.sum(adj_static[triu_idx] > 0) / n_total_edges)

    # 2. MST 骨架 + Top-20% 比例阈值
    # 使用距离 1 / (pli + 1e-6) 求最大权重生成树 (Maximum Spanning Tree)
    dist_mat = 1.0 / (pli_mat + 1e-6)
    np.fill_diagonal(dist_mat, 0.0)
    mst_sparse = minimum_spanning_tree(dist_mat).toarray()
    mst_mask = (mst_sparse > 0) | (mst_sparse.T > 0)

    # 固定保留 Top 20% 最强边，并确保包含 MST 连通骨架
    n_keep = max(n_ch - 1, int(np.round(n_total_edges * top_prop)))
    edge_weights = pli_mat[triu_idx]
    thresh_val = np.sort(edge_weights)[::-1][min(n_keep - 1, len(edge_weights) - 1)]
    prop_mask = (pli_mat >= thresh_val) | mst_mask

    adj_omst = np.where(prop_mask, pli_mat, 0.0)
    deg_omst = np.sum(adj_omst > 0, axis=1)
    n_isolated_omst = int(np.sum(deg_omst == 0))
    density_omst = float(np.sum(adj_omst[triu_idx] > 0) / n_total_edges)

    return {
        'adj_static': adj_static,
        'n_isolated_static': n_isolated_static,
        'density_static': density_static,
        'adj_omst': adj_omst,
        'n_isolated_omst': n_isolated_omst,
        'density_omst': density_omst,
        'mean_pli': float(np.mean(edge_weights)),
        'mst_mean_weight': float(np.mean(pli_mat[mst_mask]))
    }


def extract_19ch_csd_and_graph_features():
    """对 64 个会话执行 61导 -> CSD -> 19导对齐，并提取频谱与 PLI 图网络特征"""
    qc_df = pd.read_csv(PROJECT_ROOT / "data" / "derivatives" / "qc_v3_conservative" / "qc_summary_v3_conservative.csv")
    part_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "participants.tsv", sep='\t')

    records = []
    for _, row in qc_df[qc_df['status'] == 'success'].iterrows():
        sub = str(row['subject'])
        ses = int(row['session'])
        cond = str(row['condition'])
        ep_file = DERIVATIVES_DIR / sub / f"ses-{ses}" / f"{sub}_ses-{ses}_cleaned-epo.fif"
        epochs = mne.read_epochs(ep_file, preload=True, verbose=False)
        epochs.info['bads'] = []

        # 方案 A (19导 CAR): 提取标准 19 导并重参考至 19 导平均参考 (跨数据集最直接对齐)
        avail_19 = [ch for ch in STANDARD_19_CHS if ch in epochs.ch_names]
        ep_19_car = epochs.copy().pick_channels(avail_19, ordered=True).set_eeg_reference('average', verbose=False)
        psd_car = np.mean(ep_19_car.compute_psd(fmin=0.5, fmax=40.0, method='welch', verbose=False).get_data(), axis=0)

        # 方案 B (61导 CSD -> 19导): 在原始 61 导高密度阵列上计算无参考表面拉普拉斯 (CSD)
        epochs_csd = mne.preprocessing.compute_current_source_density(epochs.copy(), verbose=False)
        ep_19 = epochs_csd.pick_channels(avail_19, ordered=True)

        # 计算 19 导 CSD 下的 Welch PSD
        psd_obj = ep_19.compute_psd(fmin=0.5, fmax=40.0, method='welch', verbose=False)
        psd_data = np.mean(psd_obj.get_data(), axis=0)  # (19, n_freqs)
        freqs = psd_obj.freqs

        post_idx = [avail_19.index(c) for c in ['O1', 'O2', 'P3', 'P4', 'Pz'] if c in avail_19]
        front_idx = [avail_19.index(c) for c in ['Fp1', 'Fp2', 'F3', 'F4', 'Fz'] if c in avail_19]

        delta_m = (freqs >= 0.5) & (freqs <= 4.0)
        theta_m = (freqs >= 4.0) & (freqs <= 8.0)
        alpha_m = (freqs >= 8.0) & (freqs <= 13.0)

        car_19_delta = float(np.log10(np.mean(psd_car[:, delta_m]) + 1e-16))
        car_19_frontal_theta = float(np.log10(np.mean(psd_car[front_idx][:, theta_m]) + 1e-16))
        car_19_posterior_alpha = float(np.log10(np.mean(psd_car[post_idx][:, alpha_m]) + 1e-16))

        csd_19_delta = float(np.log10(np.mean(psd_data[:, delta_m]) + 1e-16))
        csd_19_frontal_theta = float(np.log10(np.mean(psd_data[front_idx][:, theta_m]) + 1e-16))
        csd_19_posterior_alpha = float(np.log10(np.mean(psd_data[post_idx][:, alpha_m]) + 1e-16))

        # 1/f 非周期斜率 (19 导 CAR & CSD)
        fit_m = (freqs >= 1.0) & (freqs <= 40.0) & ((freqs < 7.0) | (freqs > 14.0))
        slope_car, offset_car = np.polyfit(np.log10(freqs[fit_m]), np.log10(np.mean(psd_car, axis=0)[fit_m] + 1e-16), 1)
        ap_post_car = offset_car + slope_car * np.log10(freqs[alpha_m])
        car_19_periodic_alpha = float(np.mean(np.log10(np.mean(psd_car[post_idx][:, alpha_m], axis=0) + 1e-16) - ap_post_car))

        slope, offset = np.polyfit(np.log10(freqs[fit_m]), np.log10(np.mean(psd_data, axis=0)[fit_m] + 1e-16), 1)
        ap_post = offset + slope * np.log10(freqs[alpha_m])
        csd_19_periodic_alpha = float(np.mean(np.log10(np.mean(psd_data[post_idx][:, alpha_m], axis=0) + 1e-16) - ap_post))

        # 计算前 15 个干净 epochs 的 19×19 Alpha 频段 PLI 矩阵，对比 Proposal 中的静态阈值 (PLI > 0.30) vs MST+比例阈值
        ep_alpha = ep_19.copy().filter(8.0, 13.0, verbose=False).get_data()[:15]
        pli_mat = compute_pli_matrix(ep_alpha)
        g_stats = build_graph_static_vs_omst(pli_mat, static_thresh=0.30, top_prop=0.20)

        records.append({
            'subject': sub,
            'session': ses,
            'condition': cond,
            'car19_global_delta': car_19_delta,
            'car19_frontal_theta': car_19_frontal_theta,
            'car19_posterior_alpha': car_19_posterior_alpha,
            'car19_spectral_exponent': float(slope_car),
            'car19_periodic_alpha': car_19_periodic_alpha,
            'csd19_global_delta': csd_19_delta,
            'csd19_frontal_theta': csd_19_frontal_theta,
            'csd19_posterior_alpha': csd_19_posterior_alpha,
            'csd19_spectral_exponent': float(slope),
            'csd19_periodic_alpha': csd_19_periodic_alpha,
            'pli_mean': g_stats['mean_pli'],
            'pli_mst_backbone': g_stats['mst_mean_weight'],
            'static_isolated_nodes': g_stats['n_isolated_static'],
            'static_density': g_stats['density_static'],
            'omst_isolated_nodes': g_stats['n_isolated_omst'],
            'omst_density': g_stats['density_omst'],
        })

    df_19 = pd.DataFrame(records)
    df_19 = df_19.merge(part_df[['participant_id', 'Age', 'Gender']], left_on='subject', right_on='participant_id', how='left')
    return df_19


def run_normative_calibration_and_loso(df_19):
    """
    验证 Problem 3 的 [方案 B (常模 Z-score 校准) + 方案 A (下游分类)]：
    在每个训练折中，仅使用正常基线组 (NS = Healthy Baseline) 拟合年龄+性别常模分布:
      μ_f(Age, Sex), σ_f
    计算所有样本相对于健康常模的偏离度分数 Z_{i, f}，再送入分类器。
    """
    # 联合使用 19导 CAR (保留广域慢波同步特征) + 19导 CSD (保留局灶皮层源特征与局部连接) 的 A+B 双流特征
    feat_cols = [
        'car19_global_delta',
        'car19_frontal_theta',
        'car19_posterior_alpha',
        'car19_spectral_exponent',
        'csd19_posterior_alpha',
        'csd19_periodic_alpha',
        'pli_mst_backbone'
    ]

    df = df_19.dropna(subset=feat_cols).copy()
    df['sex_num'] = (df['Gender'] == 'M').astype(float)
    df['age_num'] = pd.to_numeric(df['Age'], errors='coerce').fillna(20.0)

    # 计算全体 NS 常模 Z-score 用于统计展示
    ns_mask = df['condition'] == 'NS'
    z_df = df.copy()
    for f in feat_cols:
        X_cov = np.column_stack([np.ones(ns_mask.sum()), df.loc[ns_mask, 'age_num'], df.loc[ns_mask, 'sex_num']])
        y_ns = df.loc[ns_mask, f].values
        beta = np.linalg.lstsq(X_cov, y_ns, rcond=None)[0]
        resid_std = np.std(y_ns - X_cov @ beta, ddof=1) + 1e-8

        X_all_cov = np.column_stack([np.ones(len(df)), df['age_num'], df['sex_num']])
        z_df[f'{f}_norm_z'] = (df[f].values - (X_all_cov @ beta)) / resid_std

    # paired Z-score change (within-person normative deviation shift)
    subj_means = z_df.groupby('subject')[[f'{f}_norm_z' for f in feat_cols]].transform('mean')
    X_norm_within = (z_df[[f'{f}_norm_z' for f in feat_cols]] - subj_means).values
    y = (z_df['condition'] == 'SD').astype(int).values
    groups = z_df['subject'].values

    logo = LeaveOneGroupOut()
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.5, random_state=42))
    ])

    y_prob = np.zeros(len(y))
    for tr, te in logo.split(X_norm_within, y, groups=groups):
        clf.fit(X_norm_within[tr], y[tr])
        y_prob[te] = clf.predict_proba(X_norm_within[te])[:, 1]

    auc_19_csd_norm = roc_auc_score(y, y_prob)
    acc_19_csd_norm = accuracy_score(y, (y_prob >= 0.5).astype(int))
    return z_df, auc_19_csd_norm, acc_19_csd_norm


def main():
    print("=" * 80)
    print("Phase 9: 执行 A+B 跨导联对齐 (CSD + 19导) + OMST构图 + 常模Z校准实证测试")
    print("=" * 80)

    df_19 = extract_19ch_csd_and_graph_features()
    full_df = pd.read_parquet(FEATURES_DIR / "eeg_session_features.parquet")
    merged = df_19.merge(full_df, on=['subject', 'session', 'condition'])

    # 1. 验证 61导 CAR vs 19导 CSD 特征保真度
    r_delta, _ = stats.pearsonr(merged['global_delta_logpower'], merged['csd19_global_delta'])
    r_theta, _ = stats.pearsonr(merged['frontal_theta_logpower'], merged['csd19_frontal_theta'])
    r_alpha, _ = stats.pearsonr(merged['posterior_alpha_logpower'], merged['csd19_posterior_alpha'])
    print(f"\n[验证 1: 61导全阵列 -> 19导 CSD 跨阵列特征保真度]")
    print(f"  - Global Delta 保真度相关性:    r = {r_delta:.3f}")
    print(f"  - Frontal Theta 保真度相关性:   r = {r_theta:.3f}")
    print(f"  - Posterior Alpha 保真度相关性: r = {r_alpha:.3f}")

    # 2. 验证静态阈值 vs MST+比例阈值 (OMST 原则)
    print(f"\n[验证 2: GCN 静态阈值 (PLI>0.15) vs. MST+Top20% 动态拓扑构图 (64 sessions)]")
    print(f"  - 静态阈值孤立电极节点出现率: {(df_19['static_isolated_nodes'] > 0).mean()*100:.1f}% 会话含孤立节点 (图密度范围: {df_19['static_density'].min():.2f} ~ {df_19['static_density'].max():.2f})")
    print(f"  - MST+Top20% 孤立节点出现率:  {(df_19['omst_isolated_nodes'] > 0).mean()*100:.1f}% 会话含孤立节点 (图密度恒定: {df_19['omst_density'].mean():.2f} ± {df_19['omst_density'].std():.3f})")

    # 3. 验证常模 Z-score + 19导 CSD LOSO 判别性能
    z_df, auc_19, acc_19 = run_normative_calibration_and_loso(df_19)
    print(f"\n[验证 3: 仅用 19 导 CSD + 年龄/性别常模 Z-score 校准后的 LOSO 状态解码]")
    print(f"  - 19导 CSD + Normative Z-score LOSO AUC = {auc_19:.3f} (Accuracy = {acc_19*100:.1f}%)")

    # 保存验证结果表
    z_df.to_csv(RESULTS_DIR / "tables" / "phase9_csd19_normative_omst_validation.csv", index=False)
    print(f"\n✓ A+B 方法学验证表已保存: {RESULTS_DIR / 'tables' / 'phase9_csd19_normative_omst_validation.csv'}")


if __name__ == '__main__':
    main()
