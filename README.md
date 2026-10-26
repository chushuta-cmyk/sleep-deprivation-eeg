# Dissociable Trait and State Signatures of Acute Sleep Deprivation in Resting-State EEG & Transdiagnostic Benchmark Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![MNE-Python](https://img.shields.io/badge/MNE--Python-1.6%2B-orange.svg)](https://mne.tools/)
[![Validation](https://img.shields.io/badge/Validation-LOSO%20%26%20Nested--CV-success.svg)]()

A reproducible, leakage-free neurophysiological analysis and machine learning benchmarking repository investigating:
1. **Study 1 (Completed & Included in `results/` & `manuscript/`)**: *Dissociable Trait and State Signatures of Acute Sleep Deprivation in Resting-State EEG: Parameterized Spectral Dynamics, Individual Vulnerability, and Machine Learning Benchmarking* (OpenNeuro `ds004902`).
2. **Core Methodological Library (`src/harmonization_and_normative.py`)**: The **"A+B" Cross-Dataset Harmonization, MST+Proportional Topological Graph Construction, and Within-Site Normative $Z$-Score Calibration Framework** designed for multi-cohort psychiatric EEG benchmarking (MDD, Schizophrenia, Bipolar Disorder, ADHD).

---

## 🌟 Key Empirical Results (`N = 32` Paired Subjects, `64` Sessions, 100% QC Retention)

| Analysis Module | Key Finding | Metric / Statistical Evidence |
| :--- | :--- | :--- |
| **Phase 2: Automated QC** | 100% paired retention via multi-evidence spherical spline interpolation (`reset_bads=True`) & FastICA ocular artifact removal | **64 / 64 sessions retained** (mean clean duration: **267.3 s**) |
| **Aim 1: Cortical State Reactivity** | All three primary spectral hypotheses passed Benjamini–Hochberg FDR correction ($q < 0.05$) | • Global Delta $\uparrow$: **$d_z = +0.653$** ($p = 0.00085, q = 0.0043$)<br>• Frontal Theta $\uparrow$: **$d_z = +0.557$** ($p = 0.0036, q = 0.0089$)<br>• Posterior Alpha $\uparrow$: **$d_z = +0.426$** ($p = 0.0221, q = 0.0369$) |
| **Aim 2: Trait vs. State Dissociation** | Alpha & $1/f$ aperiodic slope act as stable individual neural fingerprints; Delta & Theta are state-reactive markers | • Global Alpha $\text{ICC}(A,1) = \mathbf{0.877}$ ($p < 10^{-10}$)<br>• Periodic Posterior Alpha $\text{ICC}(A,1) = \mathbf{0.816}$<br>• Aperiodic $1/f$ Slope $\text{ICC}(A,1) = \mathbf{0.717}$<br>• Delta / Theta $\text{ICC}(A,1) = 0.295 \sim 0.333$ |
| **Aim 4A: LOSO State Decoding** | Subject-independent classification of Normal Sleep (NS) vs. Sleep Deprivation (SD) | • **19-ch Dual-Stream (CAR+CSD) + Normative $Z$-score**: **$\text{AUC} = \mathbf{0.873}$** ($\text{Acc} = 81.2\%$)<br>• **61-ch Full Montage (L2 Logistic)**: **$\text{AUC} = \mathbf{0.862}$** ($\text{Acc} = 78.1\%$) |
| **Aim 4B: Vulnerability Prediction** | Out-of-fold prediction of subjective sleepiness amplification ($\Delta\text{KSS}$) under Nested-CV | • **Linear SVR**: **$r = \mathbf{0.482}$** ($p = 0.0052, R^2 = 0.231, \text{MAE} = 1.29$)<br>• **Ridge Regression**: **$r = \mathbf{0.474}$** ($p = 0.0061, R^2 = 0.224$) |
| **Aim 5: Incremental Validity (H4)** | Excising the $1/f$ background reveals pure periodic posterior alpha shift as a potent predictor of sleepiness vulnerability beyond baseline sleepiness & PSQI | • $\beta = \mathbf{0.384}$ ($p = 0.0125$)<br>• Explains **$\Delta R^2 = \mathbf{+15.8\%}$** ($F = 6.89, p = 0.0143$), lifting model $R^2$ from $24.5\%$ to **$40.3\%$** |
| **Phase 9: Graph Topology Validation** | Empirical refutation of static `PLI > 0.30` thresholding vs. MST + Top-20% proportional sparsity | • Static `PLI > 0.30`: **98.4% sessions have isolated nodes** (density $0.00 \sim 0.41$)<br>• **MST + Top-20%**: **0.0% isolated nodes** (invariant density $0.20 \pm 0.006$) |

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
├── manuscript/
│   ├── outline.md                             # Structured Abstract & paper section outline
│   ├── methods.md                             # Complete publication-ready Methods text
│   └── results.md                             # Consolidated Results narrative & three-line tables
├── results/
│   ├── analysis_summary_report.txt            # Full statistical output log
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

### 2. Download / Resume Missing Subjects from OpenNeuro `ds004902`
```bash
# Incrementally download missing subjects (e.g., sub-01 to sub-39) without overwriting existing data
python scripts/01_download_data.py --start-sub 1 --end-sub 71
```

### 3. Run the Full End-to-End Analysis & Benchmark Pipeline
```bash
python scripts/run_all.py
```
