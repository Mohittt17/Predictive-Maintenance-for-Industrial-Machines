# Predictive Maintenance & RUL Optimization Methodology

## 1. Problem Formulation

Industrial equipment degradation follows non-linear physical fatigue trajectories (e.g., Paris-Erdogan law for crack growth). Unexpected catastrophic failures incur massive financial losses through unannounced assembly line downtime, secondary structural damage, and emergency repair overtime.

We formulate predictive maintenance as a joint multi-task learning problem:
1. **Health State Estimation**: continuous score $HI \in [0, 100]$.
2. **72-Hour Failure Risk Classification**: binary probability $P(Y_{t+72}=1 | X_{\le t})$.
3. **Probabilistic RUL Regression**: interval estimate $[RUL_{p10}, RUL_{p90}]$.
4. **Prescriptive Action Optimization**: cost-minimizing intervention window $t^*$.

---

## 2. Feature Extraction & Engineering

High-frequency sensor telemetry undergoes multi-domain feature extraction across sliding temporal windows ($N=30$ samples):

### Time Domain Statistics
- **Root Mean Square (RMS)**: Measures total energy content in vibration signals.
  $$\text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}$$
- **Kurtosis**: Detects impulsive impact shocks characteristic of bearing race pitting.
  $$\text{Kurtosis} = \frac{\frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^4}{\sigma^4}$$
- **Crest Factor**: Peak severity relative to RMS energy.
  $$CF = \frac{\|x\|_{\infty}}{\text{RMS}}$$

### Frequency Domain Statistics
- **Fast Fourier Transform (FFT)**: Converts time series to frequency power spectrum.
- **Spectral Entropy**: Quantifies spectral irregularity and structural looseness.
- **Harmonic Band Energies**: Integrates power around characteristic defect frequencies (BPFO/BPFI).

### Wavelet Domain (DWT)
- 5-level Daubechies-4 (`db4`) decomposition isolates high-frequency transient impact energy at specific detail scales $D_1 \dots D_5$.

---

## 3. Machine Learning Algorithms Evaluated

| Model | Architecture Description | Hyperparameters |
|---|---|---|
| **XGBoost Classifier** | Gradient boosted decision trees optimized for PR-AUC on imbalanced labels | `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`, `scale_pos_weight=auto` |
| **Gradient Boosting Classifier** | Scikit-learn gradient boosting ensemble | `n_estimators=200`, `max_depth=5`, `learning_rate=0.05` |
| **Random Forest Classifier** | Bagged decision forest with balanced sub-sampling | `n_estimators=200`, `class_weight=balanced_subsample` |
| **Logistic Regression** | L2-regularized linear baseline with standardized scaling | `C=1.0`, `class_weight=balanced` |

---

## 4. Empirical Evaluation Protocol

Models are evaluated on realistic imbalanced sensor telemetry (10:1 class ratio) across non-overlapping train/test splits.

### Core Metrics:
- **Precision**: Ratio of true failure alarms to total failure alarms.
- **Recall**: Ratio of caught failures to total actual failures.
- **F1 Score**: Harmonic mean of Precision and Recall.
- **PR-AUC**: Area under Precision-Recall Curve (the gold standard for imbalanced industrial data).
- **ROC-AUC**: Receiver Operating Characteristic Area.

---

## 5. Explainability & Prescriptive Optimization

- **SHAP (SHapley Additive exPlanations)**: Computes fair marginal contribution of each sensor variable to the predicted failure probability.
- **Degradation Diagnosis Mapping**: Rules map top SHAP features to physical fault mechanisms (e.g. `vib_rms` $\to$ *Bearing race spalling*, `temp_max` $\to$ *Thermal overheating*).
- **Cost Minimization**: Computes $t^*$ balancing planned maintenance ($\approx \$3,200$) vs unplanned failure ($\ge \$185,000$).
