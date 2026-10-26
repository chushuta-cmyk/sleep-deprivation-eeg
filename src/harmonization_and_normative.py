#!/usr/bin/env python3
"""
Core Module: Multi-Dataset EEG Harmonization, Topological Graph Construction & Normative-Guided Transdiagnostic Benchmarking
实现三大核心创新组件 (可直接用于本地 Mac 与 Google Colab):
1. MontageHarmonizer (Problem 1 - 方案 A+B):
   表面拉普拉斯无参考变换 (CSD) + 19导国际标准 10-20 球面样条对齐。
2. TopologicalGraphBuilder (Problem 2 - 替代静态阈值):
   最小生成树连通骨架 (MST) + 固定比例稀疏度 (Proportional Top-k%) + 多频段连续边特征构图。
3. WithinSiteNormativeCalibrator (Problem 3 - 先方案 B 常模校准，后方案 A 跨疾病迁移与多头鉴别):
   各数据集内部基于健康对照 (HC) 建立年龄/性别校正的常模分布，输出无量纲偏离度 Z-score 图谱，
   并在统一 Z 空间计算 N×N 跨疾病泛化迁移矩阵与鉴别诊断性能。
"""

import numpy as np
import pandas as pd
import mne
from scipy.signal import hilbert
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

STANDARD_19_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz'
]


class MontageHarmonizer:
    """
    跨数据集电极阵列与参考电极统一器 (128-ch / 64-ch / 32-ch / 19-ch -> 19-ch CSD)
    """
    def __init__(self, target_channels=None, apply_csd=True, target_sfreq=250.0):
        self.target_channels = target_channels or STANDARD_19_CHANNELS
        self.apply_csd = apply_csd
        self.target_sfreq = target_sfreq

    def harmonize_epochs(self, epochs: mne.Epochs) -> mne.Epochs:
        ep = epochs.copy()
        ep.info['bads'] = []

        if abs(ep.info['sfreq'] - self.target_sfreq) > 1e-2:
            ep.resample(self.target_sfreq, npad="auto", verbose=False)

        # 大小写不敏感匹配标准 19 导；若缺失个别标准导联则先在 EEG 空间通过球面样条插值补齐
        lower_to_orig = {ch.lower(): ch for ch in ep.ch_names}
        rename_map = {}
        missing_targets = []
        for tgt in self.target_channels:
            if tgt.lower() in lower_to_orig:
                orig_name = lower_to_orig[tgt.lower()]
                if orig_name != tgt:
                    rename_map[orig_name] = tgt
            else:
                missing_targets.append(tgt)

        if rename_map:
            ep.rename_channels(rename_map)

        if missing_targets:
            n_ep, _, n_times = ep.get_data().shape
            zero_data = np.zeros((n_ep, len(missing_targets), n_times))
            info_add = mne.create_info(missing_targets, sfreq=ep.info['sfreq'], ch_types='eeg')
            ep_add = mne.EpochsArray(zero_data, info_add, verbose=False)
            ep.add_channels([ep_add], force_update_info=True)
            montage = mne.channels.make_standard_montage('standard_1020')
            ep.set_montage(montage, match_case=False, on_missing='ignore')
            ep.info['bads'] = list(missing_targets)
            ep.interpolate_bads(reset_bads=True, verbose=False)

        # 方案 B: 在完整高密度阵列上计算无参考 CSD
        if self.apply_csd and len(ep.ch_names) >= 19:
            ep = mne.preprocessing.compute_current_source_density(ep, verbose=False)

        # 方案 A: 提取标准 19 导子集
        ep = ep.pick(self.target_channels)
        return ep


class TopologicalGraphBuilder:
    """
    无孤立节点的动态拓扑脑网络构图器 (MST 骨架 + Top-k% 比例稀疏化)
    """
    def __init__(self, top_proportion=0.20):
        self.top_proportion = top_proportion

    @staticmethod
    def compute_band_pli(epochs_data: np.ndarray) -> np.ndarray:
        analytic = hilbert(epochs_data, axis=-1)
        phases = np.angle(analytic)
        n_ep, n_ch, _ = phases.shape
        pli_ep = np.zeros((n_ep, n_ch, n_ch))
        for e in range(n_ep):
            diff = phases[e][:, None, :] - phases[e][None, :, :]
            pli_ep[e] = np.abs(np.mean(np.sign(np.sin(diff)), axis=-1))
        pli = np.mean(pli_ep, axis=0)
        np.fill_diagonal(pli, 0.0)
        return pli

    def build_topology_preserving_graph(self, conn_matrix: np.ndarray) -> dict:
        n_ch = conn_matrix.shape[0]
        triu_idx = np.triu_indices(n_ch, k=1)
        n_total_edges = len(triu_idx[0])

        # 1. 最大权重生成树骨架 (确保图 100% 连通，无孤立节点)
        dist_mat = 1.0 / (conn_matrix + 1e-6)
        np.fill_diagonal(dist_mat, 0.0)
        mst_sparse = minimum_spanning_tree(dist_mat).toarray()
        mst_mask = (mst_sparse > 0) | (mst_sparse.T > 0)

        # 2. Top-k% 比例阈值边
        n_keep = max(n_ch - 1, int(np.round(n_total_edges * self.top_proportion)))
        weights = conn_matrix[triu_idx]
        thresh = np.sort(weights)[::-1][min(n_keep - 1, len(weights) - 1)]
        mask = (conn_matrix >= thresh) | mst_mask

        adj = np.where(mask, conn_matrix, 0.0)
        edge_index = np.array(np.nonzero(adj))
        edge_weight = adj[edge_index[0], edge_index[1]]

        return {
            'adjacency': adj,
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'density': float(np.sum(adj[triu_idx] > 0) / n_total_edges),
            'n_isolated_nodes': int(np.sum(np.sum(adj > 0, axis=1) == 0))
        }


class WithinSiteNormativeCalibrator(BaseEstimator, TransformerMixin):
    """
    方案 B + A 核心组件：站点内健康对照 (HC) 常模校准器
    对每个数据集/站点 d，仅在健康对照 (is_hc == True) 上拟合协变量 (Age, Sex) 回归模型：
      E[X | Age, Sex, Site=d] = C * β_d
    将所有被试转化为相对于本站点同龄同性别健康人群的偏离度 Z-score:
      Z = (X - C * β_d) / σ_{d, HC}
    """
    def __init__(self):
        self.site_params_ = {}

    def fit(self, X: np.ndarray, site_ids: np.ndarray, is_hc: np.ndarray, covariates: np.ndarray = None):
        self.site_params_ = {}
        unique_sites = np.unique(site_ids)
        for site in unique_sites:
            hc_mask = (site_ids == site) & (is_hc.astype(bool))
            if np.sum(hc_mask) < 5:
                hc_mask = (site_ids == site)
            X_hc = X[hc_mask]
            if covariates is not None:
                C_hc = np.column_stack([np.ones(len(X_hc)), covariates[hc_mask]])
                beta, _, _, _ = np.linalg.lstsq(C_hc, X_hc, rcond=None)
                resid = X_hc - C_hc @ beta
            else:
                beta = np.mean(X_hc, axis=0, keepdims=True)
                resid = X_hc - beta
            std = np.std(resid, axis=0, ddof=1)
            std = np.where(std < 1e-6, 1.0, std)
            self.site_params_[site] = {'beta': beta, 'std': std}
        return self

    def transform(self, X: np.ndarray, site_ids: np.ndarray, covariates: np.ndarray = None) -> np.ndarray:
        Z = np.zeros_like(X, dtype=float)
        for site in np.unique(site_ids):
            mask = (site_ids == site)
            params = self.site_params_.get(site, next(iter(self.site_params_.values())))
            if covariates is not None:
                C = np.column_stack([np.ones(np.sum(mask)), covariates[mask]])
                pred = C @ params['beta']
            else:
                pred = params['beta']
            Z[mask] = (X[mask] - pred) / params['std']
        return Z

    def fit_transform(self, X: np.ndarray, site_ids: np.ndarray, is_hc: np.ndarray, covariates: np.ndarray = None) -> np.ndarray:
        return self.fit(X, site_ids=site_ids, is_hc=is_hc, covariates=covariates).transform(X, site_ids=site_ids, covariates=covariates)


def compute_cross_disorder_transfer_matrix(Z_features, labels, disorder_groups, base_clf):
    """
    计算 N×N 跨疾病零样本泛化迁移矩阵 (Cross-Disorder Transfer Atlas)
    在疾病 i (Patient_i vs HC_i) 的常模 Z-score 谱上训练分类器，直接测试于疾病 j (Patient_j vs HC_j)。
    """
    disorders = sorted(np.unique(disorder_groups))
    n_d = len(disorders)
    auc_matrix = np.zeros((n_d, n_d))

    for i, d_train in enumerate(disorders):
        tr_mask = (disorder_groups == d_train)
        clf = base_clf.fit(Z_features[tr_mask], labels[tr_mask])
        for j, d_test in enumerate(disorders):
            if i == j:
                continue
            te_mask = (disorder_groups == d_test)
            probs = clf.predict_proba(Z_features[te_mask])[:, 1]
            auc_matrix[i, j] = roc_auc_score(labels[te_mask], probs)

    return pd.DataFrame(auc_matrix, index=disorders, columns=disorders)
