# Dissociable Trait and State Signatures of Acute Sleep Deprivation in Resting-State EEG: Parameterized Spectral Dynamics, Spatial Harmonization, and Machine Learning Benchmarking

**Running Title**: *Parameterized EEG Signatures of Sleep Deprivation*

**Keywords**: Sleep deprivation; Resting-state EEG; Aperiodic $1/f$ activity; Periodic oscillations; Intraclass correlation; Surface Laplacian (CSD); Orthogonal minimum spanning tree; Leave-one-subject-out cross-validation; Normative modeling

---

## Abstract

**Study Objectives**: Acute sleep deprivation (SD) profoundly degrades vigilance and affective regulation, yet conventional resting-state electroencephalography (EEG) analyses often conflate narrowband periodic oscillations with broadband aperiodic ($1/f$) background shifts, and rarely separate stable neurobiological traits from state-reactive homeostatic biomarkers. Here, we aimed to: (1) dissociate periodic and aperiodic cortical responses to acute SD; (2) quantify cross-condition trait stability versus state reactivity; (3) test the incremental validity of parameterized EEG changes in predicting subjective sleepiness vulnerability beyond habitual sleep quality; and (4) benchmark leakage-free machine learning (ML) models across full-montage (61-channel) and harmonized 19-channel dual-stream (common average reference [CAR] + surface Laplacian current source density [CSD]) representations.

**Methods**: We analyzed 64 eyes-open resting-state EEG sessions from $N = 32$ paired participants across Normal Sleep (NS) and acute Sleep Deprivation (SD) conditions (OpenNeuro `ds004902`). Signals underwent automated quality control (`v3_conservative_refined`: $250\text{ Hz}$ resampling, multi-evidence spherical spline bad-channel interpolation with `reset_bads=True`, CAR/CSD spatial transformation, FastICA ocular artifact removal, and adaptive epoch rejection), achieving $100\%$ paired subject retention (mean clean duration: $267.3\text{ s}$). Power spectra ($0.5\text{–}40\text{ Hz}$) were parameterized into aperiodic ($1/f$ spectral slope and offset, excising the $7\text{–}14\text{ Hz}$ alpha bump) and pure periodic (aperiodic-adjusted residual) components alongside canonical bandpowers, time-domain Hjorth parameters, and Phase Lag Index (PLI) minimum spanning tree (MST) graph backbones. Analyses included linear mixed-effects models (LMM) with Benjamini–Hochberg false discovery rate (FDR) correction, intraclass correlation coefficients ($\text{ICC}(A,1)$ and $\text{ICC}(C,1)$), Procrustes-aligned bootstrap Partial Least Squares (PLS), Leave-One-Subject-Out (LOSO) classification, and nested cross-validation (Nested-CV) regression.

**Results**: Acute SD elicited robust, FDR-significant elevations in global delta log-power ($d_z = +0.653, p = 0.00085, q = 0.0043$), frontal theta log-power ($d_z = +0.557, p = 0.0036, q = 0.0089$), and posterior alpha log-power ($d_z = +0.426, p = 0.0221, q = 0.0369$), accompanied by increased pure periodic frontal theta ($d_z = +0.430, p = 0.0209$), elevated Hjorth complexity ($d_z = +0.479, p = 0.0108$), and slowed Hjorth mobility ($d_z = -0.380, p = 0.0393$). Cross-condition reliability revealed a stark trait–state dissociation: global alpha ($\text{ICC}(A,1) = 0.877$), periodic posterior alpha ($\text{ICC}(A,1) = 0.816$), and the aperiodic $1/f$ exponent ($\text{ICC}(A,1) = 0.717$) functioned as highly stable individual neural fingerprints ($p < 10^{-5}$), whereas delta ($\text{ICC} = 0.295$) and frontal theta ($\text{ICC} = 0.333$) acted as state-reactive homeostatic markers. Crucially, while canonical posterior alpha log-power change showed only a marginal link to subjective sleepiness amplification ($\Delta\text{KSS}: r = 0.276, p = 0.087$), aperiodic-adjusted **periodic posterior alpha power change** strongly predicted $\Delta\text{KSS}$ ($\beta = 0.384, p = 0.0125$) and explained **$\Delta R^2 = +15.8\%$** ($F(1,26) = 6.89, p = 0.0143$) of unique variance beyond baseline sleepiness and Pittsburgh Sleep Quality Index (PSQI) scores (lifting total $R^2$ from $24.5\%$ to $40.3\%$). Under strict LOSO validation, 61-channel L2 Logistic Regression decoded brain state with $\text{AUC} = 0.862$, while our harmonized **19-channel CAR+CSD dual-stream representation with age/sex normative $Z$-score calibration and MST graph features** achieved **$\text{AUC} = 0.873$** ($\text{Accuracy} = 81.2\%$). Nested-CV Linear SVR significantly predicted out-of-fold $\Delta\text{KSS}$ ($r = 0.482, p = 0.0052$).

**Conclusions**: Parameterizing resting-state EEG into periodic and aperiodic dimensions disentangles stable individual neural fingerprints from state-dependent sleep pressure, unlocks substantial incremental prediction of subjective vulnerability to sleep loss ($\Delta R^2 = +15.8\%$), and enables high-accuracy cross-subject decoding using a portable 19-channel normative framework.

---

## 1. Introduction

Acute sleep deprivation (SD) disrupts large-scale thalamocortical and frontoparietal network dynamics, manifesting behaviorally as heightened subjective sleepiness, attentional instability, and affective dysregulation (Krause et al., 2017; Ben Simon et al., 2020). Although scalp electroencephalography (EEG) provides a direct, non-invasive window into cortical excitability and homeostatic sleep pressure, translating resting-state EEG signatures into reliable clinical or operational biomarkers has been hindered by three fundamental methodological challenges.

First, conventional spectral analyses rely almost exclusively on narrowband power integrated over predefined canonical frequency bands (delta: $0.5\text{–}4\text{ Hz}$, theta: $4\text{–}8\text{ Hz}$, alpha: $8\text{–}13\text{ Hz}$, beta: $13\text{–}30\text{ Hz}$). However, neural power spectra consist of two biologically distinct components: narrowband **periodic oscillations** (reflecting synchronous thalamocortical and local circuit rhythms) superimposed upon a broadband **aperiodic ($1/f^\chi$) background** whose spectral exponent ($\chi$) and offset index population-level synaptic excitation-to-inhibition (E/I) balance and aggregate neuronal spiking (Donoghue et al., 2020; Gao et al., 2017). Because broadband $1/f$ shifts alter power across all frequencies simultaneously, apparent increases in slow-wave power or changes in alpha amplitude during sleep loss may be confounded by—or masked by—rotations of the underlying aperiodic spectrum (Lendner et al., 2023). Whether excising the aperiodic background enhances the sensitivity of oscillatory biomarkers to subjective neurobehavioral vulnerability remains insufficiently tested.

Second, inter-individual differences in resting-state EEG spectra are substantial, raising a critical distinction between **trait-like neural fingerprints** (features that preserve individual rank-order across wakefulness and sleep loss, characterized by high intraclass correlation $[\text{ICC}]$) and **state-reactive biomarkers** (features that undergo large within-person shifts $d_z$ but exhibit low cross-condition rank stability). Conflating high-variance inter-individual baseline differences with within-subject state transitions frequently degrades the generalizability of machine learning (ML) classifiers when evaluated on unseen individuals.

Third, extending EEG biomarkers across heterogeneous clinical or multi-center cohorts faces severe spatial and topological bottlenecks: (i) montage incompatibility across recording systems (e.g., 128-channel, 64-channel, and clinical 19-channel 10-20 arrays); (ii) reference-electrode and volume-conduction confounds; and (iii) instability of functional connectivity graphs when constructed via arbitrary static thresholds (e.g., Phase Lag Index $[\text{PLI}] > 0.30$), which frequently yield fragmented graphs with isolated electrode nodes (Dimitriadis et al., 2017; van Wijk et al., 2010).

To address these gaps, we developed and validated a reproducible, leakage-free neurophysiological pipeline on paired resting-state EEG recordings from the OpenNeuro `ds004902` sleep deprivation cohort (Xiang et al., 2024). Specifically, we tested four confirmatory hypotheses: (**H1**) acute SD induces dissociable shifts across low-frequency (delta/theta), posterior alpha, aperiodic $1/f$, and time-domain Hjorth dynamics; (**H2**) alpha-band and aperiodic measures exhibit high trait-like stability ($\text{ICC} > 0.70$), whereas slow-wave delta and theta power behave as state-reactive markers; (**H3**) parameterized EEG changes couple selectively with subjective sleepiness versus anxious affective reactivity; and (**H4**) aperiodic-adjusted periodic oscillatory shifts provide significant incremental validity ($\Delta R^2$) in predicting individual sleepiness vulnerability beyond baseline state and habitual sleep quality. Finally, we benchmarked a harmonized **19-channel dual-stream (CAR + surface Laplacian CSD) + MST topological graph + within-site normative $Z$-score framework** designed for seamless scaling to multi-disorder clinical EEG repositories.

---

## 2. Materials and Methods

### 2.1 Participants and Experimental Design
We analyzed resting-state EEG and behavioral data from the open-access OpenNeuro dataset `ds004902` (*A Resting-State EEG Dataset for Sleep Deprivation*; Xiang et al., 2024). Each participant completed two counterbalanced laboratory sessions (`SessionOrder`: `NS->SD` or `SD->NS`) separated by at least one week: a Normal Sleep (`NS`, BIDS `ses-1`) session following habitual nocturnal sleep, and an acute Sleep Deprivation (`SD`, BIDS `ses-2`) session following total nocturnal sleep deprivation.

To prevent condition-label inversion across counterbalanced visit orders, BIDS session provenance was verified against the dataset documentation and internal EEGLAB `.set` file headers (`Original file: *_NS.eeg` vs. `*_SD.eeg`), confirming that `ses-1` strictly corresponds to `NS` and `ses-2` strictly corresponds to `SD` across all participants. The complete locally curated cohort comprised $N = 32$ participants ($64$ paired sessions; `sub-40` through `sub-71`; mean age $19.6 \pm 1.3$ years, $16$ females, $16$ males; $22$ `NS->SD`, $10$ `SD->NS`). Subjective sleepiness was evaluated at each session using the 9-point Karolinska Sleepiness Scale (`KSS`, $N = 32$ pairs), state anxiety using the State Anxiety Inventory (`SAI`, $N = 30$ pairs), and habitual sleep quality using the Pittsburgh Sleep Quality Index (`PSQI`).

### 2.2 Automated EEG Preprocessing and Quality Control
Continuous eyes-open resting-state EEG was recorded from $61$ scalp electrodes positioned according to the extended international 10-20 system (Brain Products GmbH, Munich, Germany). All recordings were processed through a deterministic MNE-Python pipeline (`v3_conservative_refined`):
1. **Binary Stream Integrity and Resampling**: Raw EEGLAB `.set/.fdt` files were ingested using a fault-tolerant reader (`load_raw_eeglab_robust`) that automatically reconstructed valid sample streams when `.fdt` binary lengths differed from `.set` header metadata (salvaging `sub-44 ses-2` with $236.9\text{ s}$ of intact $5000\text{ Hz}$ data). All recordings ($500\text{ Hz}$ or $5000\text{ Hz}$) were anti-alias filtered and resampled to a uniform **$250\text{ Hz}$** sampling rate.
2. **Line-Noise Removal and Pre-Filtering**: A $50\text{ Hz}$ notch filter ($\text{width} = 2.0\text{ Hz}$) attenuated powerline interference. A temporary $1.0\text{–}40.0\text{ Hz}$ zero-phase FIR bandpass copy was generated for artifact detection and Independent Component Analysis (ICA).
3. **Multi-Evidence Bad-Channel Detection and Spherical Spline Interpolation**: To avoid over-interpolation from single-metric false positives, a channel was flagged as bad only if supported by converging abnormality evidence across flatline detection ($\text{SD} < 0.1\ \mu\text{V}$), extreme robust $Z$-score of variance or peak-to-peak amplitude ($|Z_{\text{robust}}| > 5.0$), low top-quartile neighborhood correlation ($r < 0.35$), and abnormal log-PSD power ($|Z_{\text{PSD}}| > 4.0$), capped at a maximum of $15\%$ of scalp leads ($\le 9$ channels). Flagged leads were repaired via spherical spline interpolation with `reset_bads=True` so that repaired channels were fully restored for downstream region-of-interest (ROI) integration.
4. **Re-Referencing and FastICA Ocular Artifact Removal**: Signals were re-referenced to the Common Average Reference (CAR) and bandpass filtered between $0.5\text{ and }40.0\text{ Hz}$. FastICA ($n = 20$ components, `random_state=42`) was fitted on the $1\text{ Hz}$ high-passed stream; ocular components correlating with frontal polar leads (`Fp1`, `Fp2`, `Fpz`; threshold $Z = 3.0$) were identified via `find_bads_eog` (up to $4$ components) and projected out of the $0.5\text{–}40.0\text{ Hz}$ continuous data (`ica.apply`).
5. **Epoching and Adaptive Rejection**: Clean continuous data were segmented into non-overlapping $2.0\text{ s}$ epochs ($150$ initial epochs per $300\text{ s}$ recording). Epochs exceeding $150\ \mu\text{V}$ peak-to-peak or falling below $0.5\ \mu\text{V}$ were rejected (with an adaptive $200\ \mu\text{V}$ fallback if retained duration fell below $60\text{ s}$).

### 2.3 Parameterized Spectral and Time-Domain Feature Extraction
For each session, power spectral density (PSD, $0.5\text{–}40.0\text{ Hz}$) was computed per epoch and channel using Welch's method and averaged across clean epochs:
* **Canonical Bandpowers**: Absolute $\log_{10}$ power and linear/logarithmic relative power were extracted across Delta ($0.5\text{–}4\text{ Hz}$), Theta ($4\text{–}8\text{ Hz}$), Alpha ($8\text{–}13\text{ Hz}$), and Beta ($13\text{–}30\text{ Hz}$) globally and within anatomical ROIs: Posterior (`O1, O2, PO3, PO4, POz, Oz`), Frontal (`F1, F2, F3, F4, Fz, Fp1, Fp2, AF3, AF4, Fpz`), and Central (`C1, C2, C3, C4, Cz, CPz`).
* **Aperiodic ($1/f$) and Periodic Decomposition**: Following the spectral parameterization framework (Donoghue et al., 2020), the broadband aperiodic background $L(f) = b + s \log_{10}(f)$ (where $b$ is the offset and $\chi = -s$ is the aperiodic exponent) was estimated by fitting a robust linear regression in $\log_{10}(f)\text{–}\log_{10}(\text{PSD})$ space over $1.0\text{–}40.0\text{ Hz}$ while excising the $7.0\text{–}14.0\text{ Hz}$ alpha oscillatory bump. Pure **periodic oscillatory power** (`posterior_alpha_periodic_power`, `frontal_theta_periodic_power`) was quantified as the mean residual log-power above the aperiodic fit $L(f)$ within the respective frequency band.
* **Alpha Peak and Time-Domain Hjorth Parameters**: Individual Alpha Peak Frequency (IAF), Alpha Center-of-Gravity (CoG), and time-domain **Hjorth Activity, Mobility** ($\sqrt{\text{Var}(x')/\text{Var}(x)}$, a proxy for mean spectral frequency), and **Complexity** ($\text{Mobility}(x')/\text{Mobility}(x)$, indexing bandwidth and dynamical irregularity) were computed across epochs.

### 2.4 Dual-Stream 19-Channel Harmonization, MST Graphs, and Normative Calibration
To establish a cross-dataset-compatible architecture (`src/harmonization_and_normative.py`), we further processed each session through our three-stage **"A+B" harmonization pipeline**:
1. **Dual-Stream 19-Channel Spatial Alignment (`MontageHarmonizer`)**: From the 61-channel array, we constructed two complementary 19-channel representations restricted to the universal 10-20 intersection (`Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz`): (i) **Stream A (19-ch CAR)**, preserving broad, spatially synchronous slow-wave dynamics; and (ii) **Stream B (19-ch CSD)**, derived by computing the reference-free spherical spline Surface Laplacian ($\nabla^2_{\text{surf}} V$; Perrin et al., 1989; Kayser & Tenke, 2015) prior to 19-channel selection, acting as a spatial high-pass filter that eliminates volume conduction and sharpens focal cortical sources.
2. **Topology-Preserving Graph Construction (`TopologicalGraphBuilder`)**: Instantaneous phase synchronization across the 19 CSD channels was quantified via the Phase Lag Index (PLI). Rather than applying a conventional static threshold ($\text{PLI} > 0.30$), we constructed the **Maximum Spanning Tree (MST)** backbone unioned with the top $20\%$ strongest edges (Orthogonal MST / Proportional Thresholding principle; Dimitriadis et al., 2017), guaranteeing $0\%$ isolated nodes and strictly invariant graph density across all sessions.
3. **Within-Site Normative $Z$-Score Calibration (`WithinSiteNormativeCalibrator`)**: Following the normative modeling paradigm (Rutherford et al., 2022), normative regression parameters ($\hat{\mu}(\text{Age}, \text{Sex}), \hat{\sigma}_{\text{NS}}$) were estimated exclusively on the baseline Normal Sleep (`NS`) reference distribution, transforming all observations into dimensionless age- and sex-adjusted normative deviation scores $Z_{i, f}$.

### 2.5 Statistical and Machine Learning Protocols
* **Aim 1 (State Reactivity)**: Within-subject changes ($\text{SD} - \text{NS}$) were evaluated via paired $t$-tests, Wilcoxon signed-rank tests, standardized paired effect sizes (Cohen's $d_z$ with $95\%$ CIs), and Linear Mixed-Effects Models (`feature ~ condition + session_order + clean_duration_z + (1|subject)`) with Benjamini–Hochberg FDR correction ($\alpha = 0.05$).
* **Aim 2 (Trait Stability)**: Cross-condition rank stability was quantified via two-way random-effects absolute agreement $\text{ICC}(A,1)$ (`ICC2`) and mixed-effects consistency $\text{ICC}(C,1)$ (`ICC3`), supplemented by Pearson $r$ and Spearman $\rho$.
* **Aim 3 & Aim 5 (Multivariate Coupling & Incremental Validity)**: Brain–behavior covariance was assessed via Partial Least Squares (PLS) with $5,000$ permutations and $2,000$ Procrustes-aligned bootstraps. Incremental validity (Hypothesis **H4**) was tested via hierarchical OLS regression ($F$-test on $\Delta R^2$) evaluating whether EEG change features explained variance in $\Delta\text{KSS}$ and $\Delta\text{SAI}$ beyond baseline scores and habitual sleep quality (`PSQI`).
* **Aim 4 (Machine Learning Benchmark)**:
  * *Task 4A (Brain-State Decoding)*: Five classifiers (L2 Logistic Regression, Linear SVM, RBF SVM, Random Forest, Gradient Boosting) were evaluated under strict **Leave-One-Subject-Out (LOSO)** cross-validation (`LeaveOneGroupOut`) on within-subject centered features.
  * *Task 4B (Vulnerability Regression)*: Out-of-fold continuous prediction of $\Delta\text{KSS}$ was evaluated using **5-fold outer / 4-fold inner Nested Cross-Validation** (`RobustScaler` strictly inside `Pipeline`) across Ridge, ElasticNet, Linear SVR, and a Dummy mean regressor.

---

## 3. Results

### 3.1 Quality Control and Sample Retention
All **64 sessions ($32/32$ paired participants, $100.0\%$)** passed automated quality control (`qc_summary_v3_conservative.csv`). On average, $133.7 \pm 17.5$ out of $148.0$ initial $2\text{ s}$ epochs were retained per session (mean rejection rate: $9.8\%$; NS: $8.0\%$, SD: $11.6\%$), yielding a mean clean duration of **$267.3\text{ s}$** (median: $282.0\text{ s}$, minimum: $114.0\text{ s}$). An average of only $0.84$ bad channels per session required spherical spline interpolation. Behavioral assessments confirmed robust subjective sleepiness induction following acute SD ($\text{KSS}_{\text{NS}} = 4.63 \pm 1.48$ vs. $\text{KSS}_{\text{SD}} = 6.81 \pm 1.73$, $\Delta\text{KSS} = +2.19$, $p < 0.0001$) and elevated state anxiety ($\text{SAI}_{\text{NS}} = 38.87 \pm 8.35$ vs. $\text{SAI}_{\text{SD}} = 42.80 \pm 7.64$, $\Delta\text{SAI} = +3.93$, $p = 0.038$).

### 3.2 Aim 1: Cortical Spectral and Temporal Effects of Acute Sleep Deprivation
Among the five pre-specified primary EEG features, **all three canonical oscillatory targets survived Benjamini–Hochberg FDR correction ($q < 0.05$)** (Table 1; Figure `aim1_paired_effects.png`):
1. **Global Delta Log-Power** exhibited the largest within-person increase ($d_z = +0.653$, $95\%\text{ CI }[0.27, 1.04]$, $t(31) = 3.693$, $p = 0.00085$, **FDR $q = 0.0043$**; LMM condition $p = 0.0002$).
2. **Frontal Theta Log-Power** increased robustly ($d_z = +0.557$, $95\%\text{ CI }[0.18, 0.93]$, $t(31) = 3.153$, $p = 0.00358$, **FDR $q = 0.0089$**; LMM $p = 0.0012$).
3. **Posterior Alpha Log-Power** showed a significant eyes-open synchronization increase ($d_z = +0.426$, $95\%\text{ CI }[0.06, 0.79]$, $t(31) = 2.409$, $p = 0.02211$, **FDR $q = 0.0369$**; LMM $p = 0.0156$).

In the extended mechanistic feature set, **aperiodic-adjusted pure periodic frontal theta power** increased significantly ($d_z = +0.430, p = 0.0209, \text{LMM } p = 0.0134$), confirming that frontal theta elevation reflects genuine narrowband oscillatory amplification rather than a mere broadband $1/f$ shift. In the time domain, acute SD significantly reduced **Hjorth Mobility** ($d_z = -0.380, p = 0.0393, \text{LMM } p = 0.0270$—indicating global cortical slowing) and increased **Hjorth Complexity** ($d_z = +0.479, p = 0.0108, \text{LMM } p = 0.0049$—reflecting broader spectral dispersion).

#### Table 1: Within-Subject Paired Effects and Linear Mixed-Effects Model Results ($N = 32$ Pairs)
| Feature | Tier | Mean NS | Mean SD | Mean $\Delta$ | Cohen's $d_z$ [95% CI] | $p$-value | FDR $q$ | Wilcoxon $p$ | LMM $\beta$ ($p$-value) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`global_delta_logpower`** | Primary | $-11.401$ | $-11.268$ | $+0.133$ | **$+0.653$** $[0.27, 1.04]$ | **$0.00085$** | **$0.0043^*$** | $0.0012$ | $+0.132$ ($0.0002^*$) |
| **`frontal_theta_logpower`** | Primary | $-11.992$ | $-11.885$ | $+0.107$ | **$+0.557$** $[0.18, 0.93]$ | **$0.00358$** | **$0.0089^*$** | $0.0049$ | $+0.106$ ($0.0012^*$) |
| **`posterior_alpha_logpower`** | Primary | $-11.705$ | $-11.611$ | $+0.094$ | **$+0.426$** $[0.06, 0.79]$ | **$0.02211$** | **$0.0369^*$** | $0.0296$ | $+0.095$ ($0.0156^*$) |
| `global_spectral_exponent` | Primary | $-0.715$ | $-0.776$ | $-0.061$ | $-0.247$ $[-0.60, 0.11]$ | $0.17224$ | $0.2153$ | $0.1897$ | $-0.056$ ($0.1729$) |
| `posterior_alpha_peak_frequency` | Primary | $10.406$ | $10.391$ | $-0.016$ | $-0.013$ $[-0.36, 0.33]$ | $0.94182$ | $0.9418$ | $0.9056$ | $-0.023$ ($0.9355$) |
| **`global_hjorth_complexity`** | Extended | $2.159$ | $2.343$ | $+0.184$ | **$+0.479$** $[0.11, 0.85]$ | **$0.01085$** | $0.0532^\dagger$ | $0.0125$ | $+0.183$ ($0.0049^*$) |
| **`global_theta_logpower`** | Extended | $-12.016$ | $-11.936$ | $+0.080$ | **$+0.464$** $[0.10, 0.83]$ | **$0.01331$** | $0.0532^\dagger$ | $0.0169$ | $+0.080$ ($0.0067^*$) |
| **`frontal_theta_periodic_power`** | Extended | $-0.076$ | $-0.029$ | $+0.047$ | **$+0.430$** $[0.07, 0.79]$ | **$0.02087$** | $0.0557^\dagger$ | $0.0251$ | $+0.047$ ($0.0134^*$) |
| **`global_hjorth_mobility`** | Extended | $0.235$ | $0.216$ | $-0.019$ | **$-0.380$** $[-0.74, -0.02]$ | **$0.03931$** | $0.0786$ | $0.0450$ | $-0.019$ ($0.0270^*$) |

---

### 3.3 Aim 2: Trait Stability versus State Reactivity
Intraclass correlation analysis across `NS` and `SD` sessions demonstrated a clear functional segregation between trait-like neural fingerprints and state-reactive sleep-pressure indicators (Table 2; Figure `aim2_cross_condition_stability.png`):
* **Stable Neurobiological Traits ($\text{ICC} > 0.70$)**: Global alpha log-power ($\text{ICC}(A,1) = 0.877$, $r = 0.884, p = 1.92 \times 10^{-11}$), aperiodic-adjusted **periodic posterior alpha power** ($\text{ICC}(A,1) = 0.816$, $r = 0.828, p = 5.10 \times 10^{-9}$), global beta log-power ($\text{ICC}(A,1) = 0.792$), and the **global $1/f$ spectral slope** ($\text{ICC}(A,1) = 0.717$, $r = 0.724, p = 2.81 \times 10^{-6}$) showed high rank-order invariance across sleep conditions.
* **State-Reactive Homeostatic Markers ($\text{ICC} < 0.35$)**: In contrast, **global delta log-power** ($\text{ICC}(A,1) = 0.295$) and **frontal theta log-power** ($\text{ICC}(A,1) = 0.333$) exhibited low cross-condition agreement alongside large within-subject effect sizes ($d_z = 0.56\text{–}0.65$), indicating that slow-wave activity is strongly reorganized by acute state transitions. Notably, removing the $1/f$ background substantially increased the cross-session reliability of frontal theta from $\text{ICC} = 0.333$ (canonical log-power) to $\text{ICC} = 0.592$ (`frontal_theta_periodic_power`).

#### Table 2: Cross-Condition Intraclass Correlation and Rank Stability ($N = 32$)
| Feature | Functional Role | $\text{ICC}(A,1)$ [95% CI] | $\text{ICC}(C,1)$ | Pearson $r$ | Spearman $\rho$ | $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`global_alpha_logpower`** | Stable Neural Trait | **$0.877$** $[0.76, 0.94]$ | $0.879$ | $0.884$ | $0.838$ | $1.92 \times 10^{-11}$ |
| **`posterior_alpha_periodic_power`** | Stable Neural Trait | **$0.816$** $[0.66, 0.91]$ | $0.822$ | $0.828$ | $0.822$ | $5.10 \times 10^{-9}$ |
| **`global_beta_logpower`** | Stable Neural Trait | **$0.792$** $[0.62, 0.89]$ | $0.787$ | $0.787$ | $0.742$ | $8.99 \times 10^{-8}$ |
| **`posterior_alpha_logpower`** | Trait + State Modulated | **$0.767$** $[0.56, 0.88]$ | $0.791$ | $0.798$ | $0.765$ | $4.61 \times 10^{-8}$ |
| **`global_spectral_exponent`** | Stable Aperiodic Trait | **$0.717$** $[0.50, 0.85]$ | $0.723$ | $0.724$ | $0.686$ | $2.81 \times 10^{-6}$ |
| **`global_hjorth_mobility`** | Moderately Stable Trait | **$0.651$** $[0.39, 0.81]$ | $0.675$ | $0.679$ | $0.662$ | $1.92 \times 10^{-5}$ |
| **`frontal_theta_periodic_power`** | Moderately Stable | **$0.592$** $[0.29, 0.78]$ | $0.626$ | $0.630$ | $0.585$ | $0.00011$ |
| `frontal_theta_logpower` | State-Reactive Marker | $0.333$ $[0.00, 0.60]$ | $0.390$ | $0.391$ | $0.453$ | $0.0271$ |
| `global_delta_logpower` | State-Reactive Marker | $0.295$ $[-0.03, 0.57]$ | $0.369$ | $0.382$ | $0.391$ | $0.0309$ |

---

### 3.4 Aim 3 & Aim 5: Brain–Behavior Coupling and Incremental Validity Beyond Baseline Sleep Quality
A central prediction of our parameterization framework (**H4**) is that isolating pure periodic oscillations from the broadband $1/f$ background should sharpen the prediction of neurobehavioral vulnerability (Table 3; Figure `aim5_associations.png`):
* **Periodic Posterior Alpha Unlocks Subjective Sleepiness Prediction ($\Delta\text{KSS}$)**:
  * Canonical unadjusted `posterior_alpha_logpower_change` showed only a weak, non-significant trend with $\Delta\text{KSS}$ ($r = 0.276$, baseline-adjusted $\beta = 0.270, p = 0.0867$, incremental $\Delta R^2 = +5.3\%, p = 0.172$).
  * In stark contrast, **aperiodic-adjusted pure periodic posterior alpha power change (`posterior_alpha_periodic_power_change`)** correlated strongly with $\Delta\text{KSS}$ ($r = \mathbf{0.417}$, baseline-adjusted $\beta = \mathbf{0.384}, p = \mathbf{0.0125}$). In hierarchical regression controlling for both baseline sleepiness ($\text{KSS}_{\text{NS}}$) and habitual sleep quality ($\text{PSQI}$), adding `posterior_alpha_periodic_power_change` increased explained variance from $R^2 = 24.5\%$ to **$R^2 = 40.3\%$ ($\Delta R^2 = \mathbf{+15.8\%}, F(1,26) = 6.894, p = \mathbf{0.0143}$)**.
* **Dissociable Slow-Wave Coupling with State Anxiety ($\Delta\text{SAI}$)**:
  * Whereas subjective sleepiness coupled selectively with posterior periodic alpha and frontal theta ($\Delta R^2 = +8.0\%, p = 0.091$), acute increases in state anxiety ($\Delta\text{SAI}$) were selectively associated with global delta log-power elevation ($r = 0.358$) and Hjorth complexity shifts ($r = 0.347$), which formed the dominant stable saliences in the first PLS latent variable ($\text{BSR} = 2.99$ and $3.40$, respectively).

#### Table 3: Incremental Validity ($\Delta R^2$) of EEG Changes Beyond Baseline State and Habitual Sleep Quality (`PSQI`)
| Behavioral Outcome | EEG Change Predictor | Baseline + PSQI $R^2$ | Full Model $R^2$ | Incremental $\Delta R^2$ | $F_{\text{change}}$ | $p_{\text{change}}$ | Adjusted $\beta$ ($p$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\Delta\text{KSS}$ (Sleepiness)** | **`posterior_alpha_periodic_power_change`** | **$0.245$** | **$0.403$** | **$+0.158$ ($+15.8\%$)** | **$6.894$** | **$0.0143^*$** | **$+0.384$ ($0.0125^*$)** |
| $\Delta\text{KSS}$ (Sleepiness) | `frontal_theta_logpower_change` | $0.245$ | $0.325$ | $+0.080$ ($+8.0\%$) | $3.081$ | $0.0910^\dagger$ | $+0.293$ ($0.0671^\dagger$) |
| $\Delta\text{KSS}$ (Sleepiness) | `posterior_alpha_logpower_change` (Raw) | $0.245$ | $0.298$ | $+0.053$ ($+5.3\%$) | $1.970$ | $0.1722$ | $+0.270$ ($0.0867^\dagger$) |
| $\Delta\text{KSS}$ (Sleepiness) | `global_spectral_exponent_change` | $0.245$ | $0.285$ | $+0.041$ ($+4.1\%$) | $1.487$ | $0.2336$ | $-0.197$ ($0.2233$) |
| **$\Delta\text{SAI}$ (State Anxiety)** | `global_delta_logpower_change` | $0.464$ | $0.513$ | $+0.050$ ($+5.0\%$) | $2.665$ | $0.1146$ | $+0.212$ ($0.1397$) |
| $\Delta\text{SAI}$ (State Anxiety) | `global_hjorth_complexity_change` | $0.464$ | $0.495$ | $+0.031$ ($+3.1\%$) | $1.595$ | $0.2178$ | $+0.173$ ($0.2359$) |

---

### 3.5 Aim 4 & Phase 9: Machine Learning Benchmark and "A+B" 19-Channel Normative Harmonization
In **Task 4A (LOSO Brain-State Classification)** across all 64 sessions ($32$ subjects), regularized linear models substantially outperformed non-linear tree ensembles (Table 4; Figure `aim4_model_comparison.png`): L2 Logistic Regression achieved **$\text{AUC} = 0.862$** ($\text{Accuracy} = 78.1\%, \text{F1} = 0.781$), followed by Linear SVM ($\text{AUC} = 0.835$) and Random Forest ($\text{AUC} = 0.812$).

In **Task 4B (Nested-CV Vulnerability Prediction)**, regularized regression combining baseline sleepiness with core parameterized EEG shifts (`posterior_alpha_periodic_power_change`, `frontal_theta_logpower_change`, `global_spectral_exponent_change`) significantly predicted out-of-fold $\Delta\text{KSS}$ (**Linear SVR: $r = 0.482, p = 0.0052, R^2 = 0.231, \text{MAE} = 1.289$**; **Ridge: $r = 0.474, p = 0.0061, R^2 = 0.224$**; **ElasticNet: $r = 0.434, p = 0.0130$**), whereas a Dummy mean predictor yielded $r = -0.129$ ($R^2 = -0.009$).

Finally, in **Phase 9 (Empirical Validation of the "A+B" Cross-Dataset Harmonization Architecture)** (Table 5):
1. **Spatial Fidelity of 19-Channel CSD Compression**: Compressing the 61-channel array to the 19-channel standard 10-20 montage with Surface Laplacian (CSD) preserved high fidelity for focal posterior alpha ($r = 0.939$) and global delta ($r = 0.843$), while combining **19-ch CAR (broad synchronous rhythms) + 19-ch CSD (focal cortical sources)** captured both spatial scales.
2. **Refutation of Static Connectivity Thresholding**: Constructing 19-channel alpha PLI graphs via a conventional static threshold ($\text{PLI} > 0.30$) produced **isolated electrode nodes in $98.4\%$ of sessions ($63/64$)** with erratic graph densities ($0.00\text{–}0.41$). Conversely, our **MST + Top-$20\%$ proportional graph builder** yielded **$0.0\%$ isolated nodes** and invariant density ($0.20 \pm 0.006$).
3. **Superior Decoding via 19-Channel Normative $Z$-Score Calibration**: Applying age- and sex-adjusted normative $Z$-score calibration (`WithinSiteNormativeCalibrator`) to the 19-channel dual-stream + MST backbone representation boosted LOSO classification performance to **$\text{AUC} = 0.873$ ($\text{Accuracy} = 81.2\%$)**—surpassing the 61-channel full-montage baseline using fewer than one-third of the electrodes.

#### Table 4: Machine Learning Benchmark — LOSO State Classification and Nested-CV Vulnerability Regression
| Task / Model Family | Validation Protocol | Primary Metric | Secondary Metric | Tertiary Metric | Statistical Significance |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Task 4A: 19-ch Dual-Stream (CAR+CSD) + Normative $Z$ + MST** | **LOSO (32 Subj)** | **$\text{AUC} = 0.873$** | **$\text{Acc} = 81.2\%$** | **$\text{F1} = 0.812$** | $p < 0.0001$ |
| **Task 4A: 61-ch Logistic Regression (L2)** | LOSO (32 Subj) | $\text{AUC} = 0.862$ | $\text{Acc} = 78.1\%$ | $\text{F1} = 0.781$ | $p < 0.0001$ |
| **Task 4A: 61-ch Linear SVM** | LOSO (32 Subj) | $\text{AUC} = 0.835$ | $\text{Acc} = 78.1\%$ | $\text{F1} = 0.781$ | $p < 0.0001$ |
| Task 4A: 61-ch Random Forest | LOSO (32 Subj) | $\text{AUC} = 0.812$ | $\text{Acc} = 68.8\%$ | $\text{F1} = 0.688$ | $p = 0.0003$ |
| Task 4A: 61-ch RBF SVM | LOSO (32 Subj) | $\text{AUC} = 0.769$ | $\text{Acc} = 71.9\%$ | $\text{F1} = 0.719$ | $p = 0.0011$ |
| Task 4A: 61-ch Gradient Boosting | LOSO (32 Subj) | $\text{AUC} = 0.739$ | $\text{Acc} = 67.2\%$ | $\text{F1} = 0.677$ | $p = 0.0034$ |
| **Task 4B: Linear SVR (Predicting $\Delta\text{KSS}$)** | **Nested 5×4 CV** | **$r = 0.482$** | **$R^2 = 0.231$** | **$\text{MAE} = 1.289$** | **$p = 0.0052^*$** |
| **Task 4B: Ridge Regression (Predicting $\Delta\text{KSS}$)** | Nested 5×4 CV | $r = 0.474$ | $R^2 = 0.224$ | $\text{MAE} = 1.320$ | $p = 0.0061^*$ |
| **Task 4B: ElasticNet (Predicting $\Delta\text{KSS}$)** | Nested 5×4 CV | $r = 0.434$ | $R^2 = 0.188$ | $\text{MAE} = 1.365$ | $p = 0.0130^*$ |
| Task 4B: Dummy Regressor (Mean) | Nested 5×4 CV | $r = -0.129$ | $R^2 = -0.009$ | $\text{MAE} = 1.582$ | $p = 0.4828$ |

---

## 4. Discussion

### 4.1 Homeostatic Slow-Wave Intrusion and Eyes-Open Alpha Synchronization
Our findings confirm that acute nocturnal sleep deprivation reorganizes resting-state cortical dynamics across three distinct frequency regimes: global delta elevation ($d_z = +0.653$), frontal theta elevation ($d_z = +0.557$), and posterior eyes-open alpha augmentation ($d_z = +0.426$), accompanied by slowed time-domain Hjorth mobility ($d_z = -0.380$). In wakefulness under high homeostatic sleep pressure, frontal theta and delta elevations reflect local "sleep-like" bistability and synaptic overload within prefrontal-anterior cingulate networks (Vyazovskiy et al., 2011; Finelli et al., 2000). Simultaneously, the rise in eyes-open posterior alpha power reflects diminished ascending arousal from the brainstem-thalamic activating system, shifting visual-parietal thalamocortical loops toward an idling state characteristic of drowsiness and microsleep intrusion.

### 4.2 Why Parameterizing Periodic Alpha Unlocks Incremental Vulnerability Prediction
A key insight of this study is the striking divergence between canonical posterior alpha log-power ($\Delta R^2 = +5.3\%, p = 0.172$) and **aperiodic-adjusted pure periodic posterior alpha power** ($\Delta R^2 = +15.8\%, p = 0.0143$) in predicting individual subjective sleepiness amplification ($\Delta\text{KSS}$). In raw Welch spectra, posterior alpha power is a composite sum of the narrowband $8\text{–}13\text{ Hz}$ thalamocortical rhythm and the underlying $1/f$ aperiodic background. Because individual shifts in the $1/f$ slope and offset introduce broadband variance unrelated to narrowband alpha synchronization, raw alpha change suffers from degraded signal-to-noise ratio. Excising the $1/f$ background isolates the true oscillatory amplitude above background, revealing that thalamocortical alpha synchronization accounts for nearly $16\%$ of unique variance in subjective sleepiness vulnerability above and beyond baseline sleepiness and habitual sleep quality (PSQI).

Furthermore, our $\text{ICC}$ analysis resolves a longstanding ambiguity in sleep electrophysiology: while alpha power and the $1/f$ spectral exponent are highly stable neurobiological traits ($\text{ICC} = 0.72\text{–}0.88$), the *within-person residual modulation* of periodic alpha indexes individual susceptibility to sleep loss, whereas delta and frontal theta power ($\text{ICC} = 0.30\text{–}0.33$) serve as universal state markers of homeostatic drive.

### 4.3 Implications for Multi-Cohort Clinical EEG Benchmarking ("A+B" Architecture)
In evaluating machine learning pipelines for psychiatric and neurological EEG, two pitfalls routinely inflate or sabotage cross-dataset generalization: montage mismatch and static graph thresholding. Our Phase 9 empirical validation demonstrates that:
1. Surface Laplacian (CSD) transformation removes reference-electrode dependence and sharpens focal rhythms ($r = 0.939$ with 61-ch posterior alpha), while pairing **19-ch CAR + 19-ch CSD** preserves both global slow-wave synchrony and local source specificity.
2. Static thresholding ($\text{PLI} > 0.30$) fractures functional connectivity graphs in $98.4\%$ of recordings, whereas **MST + Top-$20\%$ proportional thresholding** guarantees connected graph backbones with zero isolated nodes.
3. Calibrating features into age- and sex-adjusted normative $Z$-scores relative to healthy baseline distributions (`WithinSiteNormativeCalibrator`) eliminates demographic and amplifier scaling confounds, enabling a 19-channel linear classifier to achieve **$\text{AUC} = 0.873$** under strict LOSO cross-validation.

### 4.4 Limitations and Future Directions
Several limitations warrant consideration. First, the primary paired cohort analyzed here ($N = 32$, $64$ sessions) represents the subset of `ds004902` with complete `KSS` and `SAI` assessments; expanding the pipeline to incorporate the remaining sub-cohort (`sub-01` through `sub-39`, which includes Psychomotor Vigilance Task [PVT] reaction-time lapses) will allow direct comparison between subjective sleepiness and objective attentional lapses. Second, future extensions to multi-disorder psychiatric cohorts (MDD, Schizophrenia, Bipolar Disorder, ADHD) should leverage our `WithinSiteNormativeCalibrator` on each site's unmedicated healthy controls to disentangle disorder-specific oscillatory pathology from psychotropic medication confounds.

---

## 5. Conclusion

By integrating rigorous automated quality control, spectral parameterization ($1/f$ aperiodic vs. periodic decomposition), and leakage-free cross-validation, this study demonstrates that: (1) acute sleep deprivation induces FDR-significant slow-wave and posterior alpha synchronization; (2) alpha oscillations and the $1/f$ exponent constitute stable individual neural traits ($\text{ICC} = 0.72\text{–}0.88$), whereas delta and theta act as state-reactive markers; (3) aperiodic-adjusted periodic alpha power provides substantial incremental validity ($\Delta R^2 = +15.8\%$) in predicting subjective sleepiness vulnerability; and (4) a harmonized 19-channel CAR+CSD normative framework achieves $\text{AUC} = 0.873$ in cross-subject brain-state decoding.

---

## References

1. Ben Simon, E., Vallat, R., Barnes, C. M., & Walker, M. P. (2020). Sleep loss and the socio-emotional brain. *Trends in Cognitive Sciences*, 24(6), 435–450.
2. Brodbeck, C., et al. (2024). Data leakage and inflated accuracies in machine learning pipelines for psychiatric EEG: A systematic review and benchmarking. *Biological Psychiatry: Cognitive Neuroscience and Neuroimaging*.
3. Dimitriadis, S. I., Salis, C., Tarnanas, I., & Linden, D. E. (2017). Topological filtering of dynamic functional brain networks unfolds informative chronnectomics: A novel data-driven thresholding scheme based on orthogonal minimal spanning trees (OMSTs). *Frontiers in Neuroinformatics*, 11, 28.
4. Donoghue, T., Haller, M., Peterson, E. J., Varma, P., Sebastian, P., Gao, R., ... & Voytek, B. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*, 23(12), 1655–1665.
5. Engemann, D. A., Mellot, A., Höchenberger, R., Banville, H., Sabbagh, D., Gemein, L., ... & Gramfort, A. (2022). A reusable benchmark of brain-age estimation from electroencephalography across multiple cohorts. *NeuroImage*, 262, 119520.
6. Finelli, L. A., Baumann, H., Borbély, A. A., & Achermann, P. (2000). Dual electroencephalogram markers of human sleep homeostasis: Correlation between theta activity in waking and slow-wave activity in sleep. *Neuroscience*, 101(3), 523–529.
7. Gao, R., Peterson, E. J., & Voytek, B. (2017). Inferring synaptic excitation/inhibition balance from field potentials. *NeuroImage*, 158, 70–78.
8. Kayser, J., & Tenke, C. E. (2015). Issues and considerations for using the scalp surface Laplacian in EEG/ERP research: A tutorial review. *International Journal of Psychophysiology*, 97(3), 189–209.
9. Klepl, D., He, F., Wu, M., Blackburn, D. J., & Sarrigiannis, P. (2022). EEG-based graph neural network classification of Alzheimer's disease: An empirical evaluation of functional connectivity methods. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 30, 2651–2660.
10. Krause, A. J., Simon, E. B., Mander, B. A., Greer, S. M., Saletin, J. M., Goldstein-Piekarski, A. N., & Walker, M. P. (2017). The sleep-deprived human brain. *Nature Reviews Neuroscience*, 18(7), 404–418.
11. Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.
12. Lendner, J. D., Niethard, N., Mander, B. A., van Schalkwijk, F. J., Schuh-Hofer, S., Schmidt, H., ... & Knight, R. T. (2023). Human sleep deprivation shifts cortical excitation/inhibition balance as indexed by resting-state EEG aperiodic activity. *NeuroImage*, 270, 119972.
13. Li, X., Zhou, Y., Dvornek, N., Zhang, M., Gao, S., Zhuang, J., ... & Duncan, J. S. (2021). BrainGNN: Interpretable brain graph neural network for fMRI analysis. *Medical Image Analysis*, 74, 102233.
14. Perrin, F., Pernier, J., Bertrand, O., & Echallier, J. F. (1989). Spherical splines for scalp potential and current density mapping. *Electroencephalography and Clinical Neurophysiology*, 72(2), 184–187.
15. Rutherford, S., Kia, S. M., Wolfers, T., Fraza, C., Zabihi, M., Dinga, R., ... & Marquand, A. F. (2022). The normative modeling framework for computational psychiatry. *Nature Protocols*, 17(7), 1711–1734.
16. Song, T., Zheng, W., Song, P., & Cui, Z. (2020). EEG emotion recognition using dynamical graph convolutional neural networks. *IEEE Transactions on Affective Computing*, 11(3), 532–541.
17. van Wijk, B. C., Stam, C. J., & Daffertshofer, A. (2010). Comparing brain networks of different size and connectivity density using graph theory. *PLoS ONE*, 5(10), e13701.
18. Vyazovskiy, V. V., Olcese, U., Hanlon, E. C., Nir, Y., Cirelli, C., & Tononi, G. (2011). Local sleep in awake rats. *Nature*, 472(7344), 443–447.
19. Xiang, C., Fan, X., Bai, D., Lv, K., & Lei, X. (2024). A resting-state EEG dataset for sleep deprivation. *Scientific Data*, 11, 427.
20. Yang, C., Westover, M. B., & Sun, J. (2023). BIOT: Biosignal transformer for cross-data learning in the wild. *Advances in Neural Information Processing Systems (NeurIPS)*, 36.
