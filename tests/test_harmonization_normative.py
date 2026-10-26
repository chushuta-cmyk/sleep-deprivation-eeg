#!/usr/bin/env python3
"""
Unit tests for A+B Montage Harmonization, OMST Graph Construction, and Within-Site Normative Z-Score Calibration.
"""

import sys
from pathlib import Path
import numpy as np
import mne

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.harmonization_and_normative import (
    STANDARD_19_CHANNELS,
    MontageHarmonizer,
    TopologicalGraphBuilder,
    WithinSiteNormativeCalibrator,
)


def test_all():
    np.random.seed(42)
    # 1. Test MontageHarmonizer (simulating 64-ch -> 19-ch CSD)
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_names = montage.ch_names[:32]
    info = mne.create_info(ch_names, sfreq=500.0, ch_types='eeg')
    ep_data = np.random.randn(5, len(ch_names), 1000) * 1e-5
    epochs = mne.EpochsArray(ep_data, info, verbose=False)
    epochs.set_montage(montage, on_missing='ignore')

    harmonizer = MontageHarmonizer(apply_csd=True, target_sfreq=250.0)
    ep_19 = harmonizer.harmonize_epochs(epochs)
    assert ep_19.ch_names == STANDARD_19_CHANNELS
    assert ep_19.info['sfreq'] == 250.0

    # 2. Test TopologicalGraphBuilder (OMST + Top-20% proportional sparsity)
    builder = TopologicalGraphBuilder(top_proportion=0.20)
    pli = builder.compute_band_pli(ep_19.get_data())
    g = builder.build_topology_preserving_graph(pli)
    assert g['n_isolated_nodes'] == 0
    assert 0.15 <= g['density'] <= 0.35

    # 3. Test WithinSiteNormativeCalibrator across 2 simulated sites with large site offset
    X_site1 = np.random.randn(40, 6) + 10.0
    X_site2 = np.random.randn(40, 6) - 25.0
    X = np.vstack([X_site1, X_site2])
    sites = np.array(['MODMA'] * 40 + ['TUH'] * 40)
    is_hc = np.array([1] * 20 + [0] * 20 + [1] * 20 + [0] * 20)
    covs = np.random.uniform(18, 55, size=(80, 1))

    cal = WithinSiteNormativeCalibrator()
    Z = cal.fit_transform(X, site_ids=sites, is_hc=is_hc, covariates=covs)
    # Site means on HC should be ~0 and std ~1
    assert np.allclose(np.mean(Z[is_hc == 1], axis=0), 0.0, atol=1e-5)
    print("✓ All A+B unit tests (MontageHarmonizer, TopologicalGraphBuilder, WithinSiteNormativeCalibrator) passed!")


if __name__ == '__main__':
    test_all()
