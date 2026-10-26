# Consolidated Results Summary (`sleep_deprivation_eeg`)

## 1. Data Quality & Attrition Profile (Phase 2)
* **Sample Retention**: 64 out of 64 available sessions ($N = 32$ complete paired subjects) passed quality control (**100.0% retention**).
* **Signal Cleanliness**:
  * Mean retained epochs: $133.7 \pm 17.5$ out of $148.0$ initial epochs (mean rejection rate: $9.8\%$; NS: $8.0\%$, SD: $11.6\%$).
  * Mean clean duration: $267.3\text{ s}$ (median: $282.0\text{ s}$, range: $114.0\text{–}298.0\text{ s}$).
  * Mean interpolated bad channels per session: $0.84$ channels (all reset and restored prior to CAR and PSD estimation).

---

## 2. Aim 1: Cortical Spectral & Temporal Effects of Acute Sleep Deprivation ($N = 32$)

### Table 1: Primary & Extended Paired Effects (SD vs. NS)
| Feature | Tier | Mean NS | Mean SD | Mean $\Delta$ | Cohen's $d_z$ [95% CI] | $t$-stat | $p$-value | FDR $q$ | LMM $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`global_delta_logpower`** | **Primary** | $-11.401$ | $-11.268$ | $+0.133$ | **$+0.653$** $[0.27, 1.04]$ | $3.693$ | **$0.00085$** | **$0.0043^*$** | **$0.0002^*$** |
| **`frontal_theta_logpower`** | **Primary** | $-11.992$ | $-11.885$ | $+0.107$ | **$+0.557$** $[0.18, 0.93]$ | $3.153$ | **$0.00358$** | **$0.0089^*$** | **$0.0012^*$** |
| **`posterior_alpha_logpower`** | **Primary** | $-11.705$ | $-11.611$ | $+0.094$ | **$+0.426$** $[0.06, 0.79]$ | $2.409$ | **$0.02211$** | **$0.0369^*$** | **$0.0156^*$** |
| `global_spectral_exponent` | Primary | $-0.715$ | $-0.776$ | $-0.061$ | $-0.247$ $[-0.60, 0.11]$ | $-1.397$ | $0.17224$ | $0.2153$ | $0.1729$ |
| `posterior_alpha_peak_frequency` | Primary | $10.406$ | $10.391$ | $-0.016$ | $-0.013$ $[-0.36, 0.33]$ | $-0.074$ | $0.94182$ | $0.9418$ | $0.9355$ |
| **`global_hjorth_complexity`** | Extended | $2.159$ | $2.343$ | $+0.184$ | **$+0.479$** $[0.11, 0.85]$ | $2.711$ | **$0.01085$** | $0.0532^\dagger$ | **$0.0049^*$** |
| **`global_theta_logpower`** | Extended | $-12.016$ | $-11.936$ | $+0.080$ | **$+0.464$** $[0.10, 0.83]$ | $2.626$ | **$0.01331$** | $0.0532^\dagger$ | **$0.0067^*$** |
| **`frontal_theta_periodic_power`** | Extended | $-0.076$ | $-0.029$ | $+0.047$ | **$+0.430$** $[0.07, 0.79]$ | $2.434$ | **$0.02087$** | $0.0557^\dagger$ | **$0.0134^*$** |
| **`global_hjorth_mobility`** | Extended | $0.235$ | $0.216$ | $-0.019$ | **$-0.380$** $[-0.74, -0.02]$ | $-2.152$ | **$0.03931$** | $0.0786$ | **$0.0270^*$** |

*(Note: $^* q < 0.05$ significant after Benjamini–Hochberg FDR correction; $^\dagger q < 0.06$ trend).*

---

## 3. Aim 2: Trait Stability vs. State Reactivity ($N = 32$)

### Table 2: Cross-Condition Intraclass Correlation & Rank Stability
| Feature | Role Classification | $\text{ICC}(A,1)$ [95% CI] | $\text{ICC}(C,1)$ | Pearson $r$ | $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`global_alpha_logpower`** | **Stable Neural Trait** | **$0.877$** $[0.76, 0.94]$ | $0.879$ | $0.884$ | $1.92 \times 10^{-11}$ |
| **`posterior_alpha_periodic_power`** | **Stable Neural Trait** | **$0.816$** $[0.66, 0.91]$ | $0.822$ | $0.828$ | $5.10 \times 10^{-9}$ |
| **`global_beta_logpower`** | **Stable Neural Trait** | **$0.792$** $[0.62, 0.89]$ | $0.787$ | $0.787$ | $8.99 \times 10^{-8}$ |
| **`posterior_alpha_logpower`** | **Trait + State Modulated** | **$0.767$** $[0.56, 0.88]$ | $0.791$ | $0.798$ | $4.61 \times 10^{-8}$ |
| **`global_spectral_exponent`** | **Stable Aperiodic Trait** | **$0.717$** $[0.50, 0.85]$ | $0.723$ | $0.724$ | $2.81 \times 10^{-6}$ |
| **`global_hjorth_mobility`** | **Moderately Stable Trait** | **$0.651$** $[0.39, 0.81]$ | $0.675$ | $0.679$ | $1.92 \times 10^{-5}$ |
| **`frontal_theta_periodic_power`** | **Moderately Stable** | **$0.592$** $[0.29, 0.78]$ | $0.626$ | $0.630$ | $0.00011$ |
| `frontal_theta_logpower` | **State-Reactive Marker** | $0.333$ $[0.00, 0.60]$ | $0.390$ | $0.391$ | $0.0271$ |
| `global_delta_logpower` | **State-Reactive Marker** | $0.295$ $[-0.03, 0.57]$ | $0.369$ | $0.382$ | $0.0309$ |

---

## 4. Aim 4: Machine Learning Benchmark (LOSO Classification & Nested-CV Regression)

### Table 3A: Leave-One-Subject-Out (LOSO) State Decoding (NS vs. SD, $N=32$ subjects, 64 sessions)
| Classifier | Validation Protocol | AUC-ROC | Subject-Pair AUC | Accuracy | F1-Score | Sensitivity | Specificity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | LOSO | **$0.862$** | $0.781$ | **$78.1\%$** | **$0.781$** | $78.1\%$ | $78.1\%$ |
| **Linear SVM** | LOSO | **$0.835$** | $0.781$ | **$78.1\%$** | **$0.781$** | $78.1\%$ | $78.1\%$ |
| **Random Forest** | LOSO | **$0.812$** | $0.688$ | $68.8\%$ | $0.688$ | $68.8\%$ | $68.8\%$ |
| **RBF SVM** | LOSO | $0.769$ | $0.719$ | $71.9\%$ | $0.719$ | $71.9\%$ | $71.9\%$ |
| **Gradient Boosting** | LOSO | $0.739$ | $0.656$ | $67.2\%$ | $0.677$ | $68.8\%$ | $65.6\%$ |

### Table 3B: Nested Cross-Validation Continuous Vulnerability Prediction ($\Delta\text{KSS}$, $N=32$)
| Regression Model | Out-of-Fold Pearson $r$ | $p$-value | Out-of-Fold $R^2$ | Mean Absolute Error (MAE) |
| :--- | :---: | :---: | :---: | :---: |
| **Linear SVR** | **$0.482$** | **$0.0052$** | **$0.231$** | **$1.289$** |
| **Ridge Regression (L2)** | **$0.474$** | **$0.0061$** | **$0.224$** | $1.320$ |
| **ElasticNet (L1+L2)** | **$0.434$** | **$0.0130$** | $0.188$ | $1.365$ |
| Dummy Regressor (Mean) | $-0.129$ | $0.4828$ | $-0.009$ | $1.582$ |

---

## 5. Aim 3 & Aim 5: Brain–Behavior Coupling and Incremental Validity
* **Specificity of Periodic Alpha to Subjective Sleepiness ($\Delta\text{KSS}$)**:
  * While canonical posterior alpha log-power change showed a marginal association with $\Delta\text{KSS}$ ($r = 0.276, p = 0.087$), **aperiodic-adjusted pure periodic posterior alpha power change (`posterior_alpha_periodic_power_change`)** exhibited a strong positive coupling with subjective sleepiness amplification ($r = \mathbf{0.417}$, baseline-adjusted $\beta = \mathbf{0.384}, p = \mathbf{0.0125}$).
  * **Incremental Validity (Hypothesis H4)**: Adding `posterior_alpha_periodic_power_change` to a baseline clinical model (`KSS_NS + PSQI_GlobalScore`) increased explained variance in $\Delta\text{KSS}$ from $R^2 = 24.5\%$ to **$R^2 = 40.3\%$ ($\Delta R^2 = \mathbf{+15.8\%}, F(1,26) = 6.89, p = \mathbf{0.0143}$)**.
* **Dissociable Slow-Wave Coupling with State Anxiety ($\Delta\text{SAI}$)**:
  * Unlike subjective sleepiness (which coupled with posterior periodic alpha and frontal theta), acute increases in state anxiety ($\Delta\text{SAI}$) correlated selectively with global delta log-power elevation ($r = \mathbf{0.358}$) and Hjorth complexity shifts ($r = \mathbf{0.347}$), reflected in the first PLS latent variable where delta change ($\text{BSR} = 2.99$) and Hjorth complexity ($\text{BSR} = 3.40$) loaded stably onto affective-anxious reactivity.
