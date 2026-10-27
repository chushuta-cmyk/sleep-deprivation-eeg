#!/usr/bin/env python3
"""
Generate publication-quality 61-channel Scalp Topographical Difference Maps (Figure 2).
Computes channel-wise paired log10 power changes (SD - NS) across Delta, Theta, Alpha, Beta,
and the aperiodic exponent across all 71 paired participants (142 sessions).
"""

import warnings
from pathlib import Path
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QC_FILE = PROJECT_ROOT / "data" / "derivatives" / "qc_v3_conservative" / "qc_summary_v3_conservative.csv"
OUT_FIG = PROJECT_ROOT / "results" / "figures" / "aim1_topomap_diff.png"


def main():
    print("Generating Figure 2: 61-channel scalp topographical difference maps (N = 71 pairs)...")
    qc_df = pd.read_csv(QC_FILE)
    success_df = qc_df[qc_df['status'] == 'success'].copy()

    # Get sample montage & info
    sample_epochs = mne.read_epochs(success_df['epochs_file'].iloc[0], preload=False, verbose=False)
    info = sample_epochs.info
    ch_names = sample_epochs.ch_names
    n_chs = len(ch_names)

    # Accumulate paired channel PSDs
    subjs = sorted(success_df['subject'].unique())
    n_pairs = len(subjs)

    delta_diffs = []
    theta_diffs = []
    alpha_diffs = []
    beta_diffs = []
    exp_diffs = []

    for sub in subjs:
        s_df = success_df[success_df['subject'] == sub]
        if len(s_df) < 2:
            continue
        ns_row = s_df[s_df['condition'] == 'NS'].iloc[0]
        sd_row = s_df[s_df['condition'] == 'SD'].iloc[0]

        ep_ns = mne.read_epochs(ns_row['epochs_file'], preload=True, verbose=False)
        ep_sd = mne.read_epochs(sd_row['epochs_file'], preload=True, verbose=False)

        psd_ns_obj = ep_ns.compute_psd(fmin=0.5, fmax=40.0, method='welch', verbose=False)
        psd_sd_obj = ep_sd.compute_psd(fmin=0.5, fmax=40.0, method='welch', verbose=False)

        freqs = psd_ns_obj.freqs
        p_ns = np.mean(psd_ns_obj.get_data(), axis=0)  # (n_ch, n_freqs)
        p_sd = np.mean(psd_sd_obj.get_data(), axis=0)

        delta_m = (freqs >= 0.5) & (freqs <= 4.0)
        theta_m = (freqs >= 4.0) & (freqs <= 8.0)
        alpha_m = (freqs >= 8.0) & (freqs <= 13.0)
        beta_m = (freqs >= 13.0) & (freqs <= 30.0)

        d_diff = np.log10(np.mean(p_sd[:, delta_m], axis=1) + 1e-16) - np.log10(np.mean(p_ns[:, delta_m], axis=1) + 1e-16)
        t_diff = np.log10(np.mean(p_sd[:, theta_m], axis=1) + 1e-16) - np.log10(np.mean(p_ns[:, theta_m], axis=1) + 1e-16)
        a_diff = np.log10(np.mean(p_sd[:, alpha_m], axis=1) + 1e-16) - np.log10(np.mean(p_ns[:, alpha_m], axis=1) + 1e-16)
        b_diff = np.log10(np.mean(p_sd[:, beta_m], axis=1) + 1e-16) - np.log10(np.mean(p_ns[:, beta_m], axis=1) + 1e-16)

        # 1/f exponent chi = -slope
        fit_m = (freqs >= 1.0) & (freqs <= 40.0) & ((freqs < 7.0) | (freqs > 14.0))
        log_f = np.log10(freqs[fit_m])

        chi_ns = []
        chi_sd = []
        for ch in range(n_chs):
            s_ns, _ = np.polyfit(log_f, np.log10(p_ns[ch, fit_m] + 1e-16), 1)
            s_sd, _ = np.polyfit(log_f, np.log10(p_sd[ch, fit_m] + 1e-16), 1)
            chi_ns.append(-s_ns)
            chi_sd.append(-s_sd)
        e_diff = np.array(chi_sd) - np.array(chi_ns)

        delta_diffs.append(d_diff)
        theta_diffs.append(t_diff)
        alpha_diffs.append(a_diff)
        beta_diffs.append(b_diff)
        exp_diffs.append(e_diff)

    mean_delta = np.mean(delta_diffs, axis=0)
    mean_theta = np.mean(theta_diffs, axis=0)
    mean_alpha = np.mean(alpha_diffs, axis=0)
    mean_beta = np.mean(beta_diffs, axis=0)
    mean_exp = np.mean(exp_diffs, axis=0)

    # Plot 5 topomaps in 1 row
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
    maps = [
        (mean_delta, r"$\Delta$ Delta (0.5–4 Hz)", 'RdBu_r'),
        (mean_theta, r"$\Delta$ Theta (4–8 Hz)", 'RdBu_r'),
        (mean_alpha, r"$\Delta$ Alpha (8–13 Hz)", 'RdBu_r'),
        (mean_beta, r"$\Delta$ Beta (13–30 Hz)", 'RdBu_r'),
        (mean_exp, r"$\Delta$ Aperiodic Exponent ($\chi$)", 'PuOr_r'),
    ]

    for ax, (data_vals, title, cmap) in zip(axes, maps):
        vmax = max(abs(data_vals.min()), abs(data_vals.max()))
        vmax = max(vmax, 0.05)
        im, _ = mne.viz.plot_topomap(
            data_vals,
            info,
            axes=ax,
            show=False,
            cmap=cmap,
            vlim=(-vmax, vmax),
            sphere=0.10,
            contours=4,
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        unit = r"$\Delta\chi$" if "Exponent" in title else r"$\Delta\log_{10}$"
        cbar.set_label(unit, fontsize=8.5)
        ax.set_title(title, fontsize=10.5, fontweight='bold', pad=8)

    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 2 topomap successfully created at: {OUT_FIG}")


if __name__ == '__main__':
    main()
