# Manuscript Outline & Abstract Draft

**Working Title**:  
*Dissociable Trait and State Signatures of Acute Sleep Deprivation in Resting-State EEG: Parameterized Spectral Dynamics, Individual Vulnerability, and Machine Learning Benchmarking*

---

## 1. Structured Abstract (Draft / Outline)

* **Background & Objective**:  
  Acute sleep deprivation (SD) profoundly impairs vigilance and affective regulation, yet conventional resting-state electroencephalography (EEG) analyses often conflate narrowband oscillatory shifts with broadleaf aperiodic ($1/f$) background changes, and rarely distinguish stable neurobiological traits from state-sensitive biomarkers. Using within-subject resting-state EEG from the OpenNeuro `ds004902` cohort, we aimed to: (1) dissociate periodic and aperiodic cortical responses to acute SD; (2) quantify cross-condition trait stability versus state reactivity; (3) benchmark subject-independent machine learning (ML) models for brain-state decoding and individual sleepiness vulnerability prediction; and (4) test the incremental validity of parameterized EEG changes beyond baseline sleep quality.
* **Methods**:  
  We analyzed 64 resting-state eyes-open EEG sessions from $N = 32$ paired participants across Normal Sleep (NS) and acute Sleep Deprivation (SD) conditions (counterbalanced order). Signals were preprocessed via an automated quality-control pipeline (250 Hz resampling, multi-evidence bad-channel spherical spline interpolation with `reset_bads=True`, average referencing, FastICA ocular artifact removal, and adaptive epoch rejection; 100% subject retention, mean clean duration $267.3\text{ s}$). Power spectral density (0.5–40 Hz) was parameterized into aperiodic ($1/f$ spectral exponent and offset, excluding the 7–14 Hz alpha bump) and periodic (aperiodic-adjusted oscillatory residual) components alongside canonical bandpowers and time-domain Hjorth parameters. Statistical analyses included linear mixed-effects models (LMM), Benjamini–Hochberg FDR-corrected paired comparisons, intraclass correlation coefficients ($\text{ICC}(A,1)$ and $\text{ICC}(C,1)$), Procrustes-aligned bootstrap Partial Least Squares (PLS), Leave-One-Subject-Out (LOSO) classification, and nested cross-validation (Nested-CV) regression.
* **Results**:  
  1. **State Reactivity (Aim 1)**: Acute SD induced robust, FDR-significant increases in global delta log-power ($d_z = +0.653, p = 0.00085, q = 0.0043$), frontal theta log-power ($d_z = +0.557, p = 0.0036, q = 0.0089$), and posterior alpha log-power ($d_z = +0.426, p = 0.0221, q = 0.0369$), accompanied by increased pure periodic frontal theta ($d_z = +0.430, p = 0.0209$), elevated Hjorth complexity ($d_z = +0.479, p = 0.0108$), and slowed Hjorth mobility ($d_z = -0.380, p = 0.0393$).
  2. **Trait vs. State Dissociation (Aim 2)**: Alpha-band measures and the aperiodic $1/f$ exponent exhibited high cross-condition rank stability ($\text{ICC}(A,1) = 0.877$ for global alpha, $0.816$ for periodic posterior alpha, and $0.717$ for spectral exponent; all $p < 10^{-5}$), acting as stable individual neural fingerprints. Conversely, delta ($\text{ICC} = 0.295$) and frontal theta ($\text{ICC} = 0.333$) demonstrated low cross-condition stability, confirming their role as pure state-reactive markers of homeostatic sleep pressure.
  3. **Machine Learning Benchmark (Aim 4)**: Under strict LOSO cross-validation, regularized linear classifiers achieved strong generalization in decoding SD versus NS brain states (L2 Logistic Regression $\text{AUC} = 0.862$, Linear SVM $\text{AUC} = 0.835$, Random Forest $\text{AUC} = 0.812$). In Nested-CV continuous regression, combining baseline sleepiness with parameterized EEG shifts significantly predicted out-of-fold subjective sleepiness vulnerability ($\Delta\text{KSS}$: Linear SVR $r = 0.482, p = 0.0052$; Ridge $r = 0.474, p = 0.0061$).
  4. **Incremental Validity (Aim 5)**: Aperiodic-adjusted posterior alpha periodic power change ($\Delta\text{Periodic Alpha}$) significantly predicted $\Delta\text{KSS}$ ($\beta = 0.384, p = 0.0125$) and explained **$\Delta R^2 = +15.8\%$** ($F = 6.89, p = 0.0143$) of unique variance beyond baseline sleepiness ($\text{KSS}_{\text{NS}}$) and habitual sleep quality ($\text{PSQI}$), lifting total explained variance from $24.5\%$ to $40.3\%$.
* **Conclusions**:  
  Decomposing resting-state EEG into periodic and aperiodic dimensions resolves trait-like individual fingerprints from state-dependent homeostatic sleep pressure, enables robust cross-subject ML state decoding ($\text{AUC} = 0.862$), and provides significant incremental value in predicting subjective vulnerability to sleep loss.

---

## 2. Section-by-Section Paper Outline

1. **Introduction**
   * 1.1 Neurobehavioral heterogeneity under acute sleep deprivation (sleepiness vs. affective dysregulation).
   * 1.2 Limitations of canonical narrowband EEG metrics: conflation of periodic oscillations with the $1/f$ aperiodic background (E/I balance).
   * 1.3 Need for rigorous distinction between *trait-like neural fingerprints* (high ICC) and *state-reactive homeostatic biomarkers* (high $d_z$, low ICC).
   * 1.4 Study Aims (Aims 1–5) & Confirmatory Hypotheses (H1–H4).
2. **Methods** (Detailed in [`methods.md`](file:///Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/sleep_deprivation_eeg/manuscript/methods.md))
   * 2.1 Dataset & Experimental Design (OpenNeuro `ds004902`, within-subject NS vs. SD, metadata audit).
   * 2.2 Automated EEG Preprocessing & Quality Control Pipeline (`v3_conservative_refined`).
   * 2.3 Parameterized Spectral (Periodic/Aperiodic) & Time-Domain Feature Extraction.
   * 2.4 Statistical & Machine Learning Framework (LMM, ICC, Procrustes-PLS, LOSO Classification, Nested-CV Regression, Incremental $\Delta R^2$).
3. **Results** (Detailed in [`results.md`](file:///Users/tan/Documents/qwen-agent/aCAgqOVe0x/default/sleep_deprivation_eeg/manuscript/results.md))
   * 3.1 Data Quality Audit & Sample Retention (64/64 sessions, 100% paired retention).
   * 3.2 Aim 1: Within-Subject Effects of Sleep Deprivation on Periodic, Aperiodic, and Hjorth Dynamics.
   * 3.3 Aim 2: Trait-State Dissociation via Cross-Condition Intraclass Correlation.
   * 3.4 Aim 3 & Aim 5: Brain–Behavior Coupling and Incremental Validity Beyond Baseline Sleep Quality.
   * 3.5 Aim 4: Leakage-Free LOSO State Classification and Nested-CV Vulnerability Prediction.
4. **Discussion**
   * 4.1 Homeostatic theta/delta intrusion and wake-state posterior alpha synchronization.
   * 4.2 Why periodic alpha decomposition outperforms raw bandpower in predicting subjective sleepiness ($\Delta R^2 = +15.8\%$).
   * 4.3 Why regularized linear models outperform tree ensembles in small-sample neuroimaging LOSO benchmarks.
   * 4.4 Limitations (sample size, open-dataset metadata curation) & Future Directions (transdiagnostic clinical benchmarking).
