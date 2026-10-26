# Dissociable Trait and State Signatures of Acute Sleep Deprivation in Resting-State EEG & Transdiagnostic Benchmark Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![MNE-Python](https://img.shields.io/badge/MNE--Python-1.6%2B-orange.svg)](https://mne.tools/)
[![Validation](https://img.shields.io/badge/Validation-LOSO%20%26%20Nested--CV-success.svg)]()

A reproducible, leakage-free neurophysiological analysis and machine learning benchmarking repository investigating:
1. **Study 1 (Completed & Included in `results/`)**: *Dissociable Trait and State Signatures of Acute Sleep Deprivation in Resting-State EEG: Parameterized Spectral Dynamics, Individual Vulnerability, and Machine Learning Benchmarking* across the full OpenNeuro `ds004902` cohort ($N = 71$ paired participants, $142$ sessions).
2. **Core Methodological Library (`src/harmonization_and_normative.py`)**: The **"A+B" Cross-Dataset Harmonization, MST+Proportional Topological Graph Construction, and Within-Site Normative $Z$-Score Calibration Framework** designed for multi-cohort psychiatric EEG benchmarking (MDD, Schizophrenia, Bipolar Disorder, ADHD).

---

## 🌟 Key Empirical Results (`N = 71` Paired Subjects, `142` Sessions, 100% QC Retention)

| Analysis Module | Key Finding | Metric / Statistical Evidence |
| :--- | :--- | :--- |
| **Phase 2: Automated QC** | 100% paired retention via binary stream recovery, multi-evidence spherical spline interpolation (`reset_bads=True`) & FastICA EOG removal | **142 / 142 sessions retained** (mean clean duration: **270.3 s**, median: **290.0 s**) |
| **Aim 1: Cortical State Reactivity ($N=71$)** | Four of five primary spectral hypotheses passed Benjamini–Hochberg FDR correction ($q < 0.05$) in covariate-adjusted LMMs | • Frontal Theta $\uparrow$: **$d_z = +0.788$** ($p = 5.70 \times 10^{-9}, q = 2.85 \times 10^{-8}$)<br>• Global Delta $\uparrow$: **$d_z = +0.523$** ($p = 3.67 \times 10^{-5}, q = 9.19 \times 10^{-5}$)<br>• Posterior Alpha $\uparrow$: **$d_z = +0.345$** ($p = 0.0049, q = 0.0082$)<br>• Aperiodic $1/f$ Slope $\downarrow$: LMM $\beta = -0.066$ ($p = 0.0350, q = 0.0448$) |
| **Aim 2: Trait vs. State Dissociation ($N=71$)** | Alpha/Beta & $1/f$ aperiodic slope act as stable individual neural fingerprints; Delta & raw Theta are state-reactive markers | • Global Alpha $\text{ICC}(A,1) = \mathbf{0.816}$ ($p = 1.15 \times 10^{-19}$)<br>• Global Beta $\text{ICC}(A,1) = \mathbf{0.744}$ ($p = 6.48 \times 10^{-15}$)<br>• Periodic Posterior Alpha $\text{ICC}(A,1) = \mathbf{0.702}$ ($p = 5.02 \times 10^{-12}$)<br>• Aperiodic $1/f$ Slope $\text{ICC}(A,1) = \mathbf{0.658}$ ($p = 3.21 \times 10^{-10}$)<br>• Global Delta $\text{ICC}(A,1) = 0.239$ (state-reactive) |
| **Aim 4A: LOSO State Decoding ($142$ Sessions)** | Subject-independent classification of Normal Sleep (NS) vs. Sleep Deprivation (SD) across 71 unseen participants | • **61-ch Linear SVM**: **$\text{AUC} = \mathbf{0.881}$** ($\text{Acc} = 83.1\%$)<br>• **61-ch RBF SVM**: **$\text{Acc} = \mathbf{86.6\%}$** ($\text{Pair-AUC} = 0.873$)<br>• **19-ch Dual-Stream (CAR+CSD) + Normative $Z$-score**: **$\text{AUC} = \mathbf{0.875}$** ($\text{Acc} = 81.7\%$) |
| **Aim 4B: Vulnerability Prediction (Nested-CV)** | Out-of-fold prediction of subjective sleepiness ($\Delta Z_{\text{Sleepiness}}, \Delta\text{KSS}$) and objective vigilance ($\Delta\text{PVT RT SD}$) | • **Full-Cohort $\Delta Z_{\text{Sleepiness}}$ ($N=66$, Ridge)**: **$r = \mathbf{0.629}$** ($p = 1.58 \times 10^{-8}, R^2 = 0.395$)<br>• **Sub-cohort $\Delta\text{KSS}$ ($N=33$, Linear SVR)**: **$r = \mathbf{0.629}$** ($p = 8.87 \times 10^{-5}, R^2 = 0.391$)<br>• **Objective $\Delta\text{PVT RT SD}$ ($N=26$, Ridge)**: **$r = \mathbf{0.750}$** ($p = 1.03 \times 10^{-5}, R^2 = 0.551$) |
| **Aim 5: Incremental Validity (H4)** | Excising the $1/f$ background reveals pure periodic posterior alpha shift as a significant predictor of sleepiness vulnerability beyond baseline & PSQI | • **Full-Cohort $\Delta Z_{\text{Sleepiness}}$ ($N=63$)**: $\beta = 0.220$ ($p = 0.0189$), **$\Delta R^2 = +4.5\%$** ($p = 0.0288$)<br>• **KSS Cohort $\Delta\text{KSS}$ ($N=31\text{–}32$)**: $\beta = 0.369$ ($p = 0.0142$), **$\Delta R^2 = +14.4\text{–}15.8\%$** ($p \le 0.0182$) |
| **Phase 9: Graph Topology Validation ($142$ Sessions)** | Empirical refutation of static `PLI > 0.30` thresholding vs. MST + Top-20% proportional sparsity | • Static `PLI > 0.30`: **99.3% sessions ($141/142$) have isolated nodes** (density $0.00 \sim 0.41$)<br>• **MST + Top-20%**: **0.0% isolated nodes ($0/142$)** (invariant density $0.20 \pm 0.007$) |

---

## 📂 Repository Structure

```text
sleep_deprivation_eeg_github_repo/
├── README.md                                  # Project overview, key findings & quickstart
├── requirements.txt                           # Python package dependencies
├── environment.yml                            # Conda environment specification
├── data_dictionary.csv                        # Variable definitions
├── config/
│   └── analysis_config.yaml                   # Unified preprocessing, ROI & statistical hyperparameters
├── src/
│   └── harmonization_and_normative.py         # Core A+B module: 19-ch CSD Harmonizer, MST Graph Builder & Normative Z Calibrator
├── scripts/
│   ├── 01_download_data.py                    # Resumable OpenNeuro S3 downloader (supports sub-01~71 incremental sync)
│   ├── 02a_preprocess_v3_conservative.py      # Automated MNE preprocessing, spherical spline repair & FastICA EOG removal
│   ├── 02b_qc_review.py                       # Quality control audit & summary plots
│   ├── 02c_extract_features.py                # Parameterized 1/f aperiodic, periodic residual & Hjorth feature extraction
│   ├── phase3_aim1.py                         # Aim 1: Paired effect sizes (dz), Wilcoxon, LMM & BH-FDR correction
│   ├── phase4_aim2.py                         # Aim 2: Cross-condition trait stability ICC(A,1) & ICC(C,1)
│   ├── phase5_aim3.py                         # Aim 3: Procrustes-aligned Bootstrap & Permutation PLS
│   ├── phase6_aim4.py                         # Aim 4: LOSO State Classification & Nested-CV Vulnerability Regression
│   ├── phase7_aim5.py                         # Aim 5: Incremental Validity (ΔR² F-test) & Bootstrap Mediation
│   ├── 09_normative_csd_19ch_validation.py    # Phase 9: Empirical validation of 19-ch CAR+CSD, MST Graph & Normative Z
│   ├── phase8_generate_report.py              # Automated non-hallucinated summary report generator
│   └── run_all.py                             # One-click end-to-end execution script
├── notebooks/
│   └── Colab_Benchmark_Runner.ipynb           # Ready-to-run Google Colab notebook
├── results/
│   ├── analysis_summary_report.txt            # Full statistical output log (N=71, 142 sessions)
│   ├── tables/                                # All CSV/JSON statistical & ML benchmark tables
│   └── figures/                               # High-resolution publication figures (200 DPI PNG)
└── tests/
    └── test_harmonization_normative.py        # Unit tests for MontageHarmonizer, TopologicalGraphBuilder & NormativeCalibrator
```

---

## 🚀 Quickstart (Local Mac or Google Colab)

### 1. Run Unit Tests
```bash
python tests/test_harmonization_normative.py
```

### 2. Download / Resume Subjects from OpenNeuro `ds004902`
```bash
python scripts/01_download_data.py --start-sub 1 --end-sub 71 --workers 8
```

### 3. Run the Full End-to-End Analysis & Benchmark Pipeline
```bash
python scripts/run_all.py
```
