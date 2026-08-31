# Data Schema Documentation

This document is the authoritative reference for all data types flowing through
the Predictive Maintenance pipeline.

---

## 1. `SensorReading` — Raw ingestion schema

The atomic unit produced by every dataset loader and the sensor simulator.

| Field | Type | Unit | Description |
|---|---|---|---|
| `machine_id` | `str` | — | Unique machine identifier (e.g. `"M-07"`, `"CMAPSS-FD001-0001"`) |
| `timestamp` | `datetime` (UTC) | — | Observation time |
| `vibration_x` | `float` | m/s² | Horizontal vibration acceleration |
| `vibration_y` | `float` | m/s² | Lateral vibration acceleration |
| `vibration_z` | `float` | m/s² | Axial vibration acceleration |
| `temperature` | `float` | °C | Component / oil temperature |
| `current` | `float` | A | Motor phase current (RMS) |
| `pressure` | `float` | bar | System / lubricant pressure |
| `acoustic` | `float` | dB | Acoustic emission RMS envelope |
| `rpm` | `float` | rev/min | Shaft speed |
| `load` | `float` | % | Percentage of rated load |
| `rul_hours` | `Optional[float]` | hours | Remaining useful life (None when unknown) |
| `health_label` | `Optional[int]` | — | 0=Healthy, 1=Minor, 2=Moderate, 3=Severe, 4=Critical/Failure |
| `dataset` | `str` | — | Source dataset tag (for traceability) |

### Health label mapping

| Label | Meaning | RUL range (approximate) |
|---|---|---|
| 0 | Healthy | RUL > 80 cycles/hours |
| 1 | Minor degradation | 40 < RUL ≤ 80 |
| 2 | Moderate degradation | 20 < RUL ≤ 40 |
| 3 | Severe degradation | 5 < RUL ≤ 20 |
| 4 | Critical / Failure | RUL ≤ 5 |

---

## 2. `FeatureVector` — Engineered feature vector

Computed by the signal processing + feature engineering pipeline over a
sliding window of `SensorReading` objects.

### Time-domain features

| Feature | Formula | Sensor(s) |
|---|---|---|
| `vib_rms` | √(Σxᵢ²/N) | vibration |
| `vib_kurtosis` | E[(X-μ)⁴]/σ⁴ | vibration |
| `vib_crest_factor` | peak / RMS | vibration |
| `vib_skewness` | E[(X-μ)³]/σ³ | vibration |
| `vib_peak_to_peak` | max(x) - min(x) | vibration |
| `vib_variance` | σ² | vibration |
| `vib_std` | σ | vibration |
| `temp_mean`, `temp_std`, `temp_max` | — | temperature |
| `curr_mean`, `curr_std`, `curr_rms` | — | current |
| `pres_mean`, `pres_std` | — | pressure |
| `acou_rms`, `acou_kurtosis`, `acou_peak` | — | acoustic |
| `rpm_mean`, `load_mean` | — | operational |

### Frequency-domain features (FFT)

| Feature | Description |
|---|---|
| `vib_dominant_freq` | Frequency bin with highest power |
| `vib_spectral_entropy` | Shannon entropy of normalised PSD |
| `vib_spectral_centroid` | Weighted mean frequency |
| `bearing_band1_energy` | Integrated PSD in BPFO/BPFI band (100–200 Hz) |
| `bearing_band2_energy` | Integrated PSD in cage+roller band (200–500 Hz) |
| `imbalance_1x_energy` | Energy at 1× RPM harmonic |
| `imbalance_2x_energy` | Energy at 2× RPM harmonic |

### Wavelet features (DWT — Daubechies 4, 5 levels)

| Feature | Description |
|---|---|
| `vib_wavelet_d1_energy` | Energy of detail level 1 (highest freq) |
| `vib_wavelet_d2_energy` | Energy of detail level 2 |
| `vib_wavelet_d3_energy` | Energy of detail level 3 |
| `vib_wavelet_d4_energy` | Energy of detail level 4 |
| `vib_wavelet_d5_energy` | Energy of detail level 5 |
| `vib_wavelet_approx_energy` | Energy of approximation (lowest freq) |

---

## 3. `MaintenanceRecommendation` — Final pipeline output

The structured object shown to the maintenance engineer.

```
Machine M-07       Health Score: 38/100
Failure probability in next 72h: 81%
Estimated RUL: 56–78 hours
Primary suspected degradation: Bearing degradation    Confidence: 87%
Recommended: Schedule intervention within the next 24–36 hours.
Reason: Waiting beyond 48h has higher expected cost (₹1,51,000) than
        planned maintenance now (₹42,000).
```

| Field | Type | Description |
|---|---|---|
| `machine_id` | `str` | Machine identifier |
| `evaluated_at` | `datetime` | Prediction timestamp |
| `health_score` | `float` (0–100) | Current health (100=perfect, 0=failure) |
| `failure_prob_72h` | `float` (0–1) | P(failure within 72 hours) |
| `rul_median_hours` | `float` | Median RUL (point estimate) |
| `rul_p10_hours` | `float` | 10th-percentile RUL (pessimistic) |
| `rul_p90_hours` | `float` | 90th-percentile RUL (optimistic) |
| `degradation_mode` | `str` | Inferred failure mode from SHAP analysis |
| `confidence` | `float` (0–1) | Overall model confidence |
| `top_shap_features` | `List[(str, float)]` | Top-5 SHAP feature contributions |
| `recommended_action` | `str` | Human-readable action recommendation |
| `action_reasoning` | `str` | Cost-optimization rationale |
| `optimal_window_hours` | `float` | Hours from now to optimal maintenance |
| `expected_cost_now` | `float` | Expected cost (₹) if maintenance done now |
| `expected_cost_optimal` | `float` | Expected cost (₹) at optimal window |
| `anomaly_score` | `float` | Current anomaly score (higher = more abnormal) |
| `is_anomaly` | `bool` | Whether current state is flagged as anomalous |

---

## 4. Dataset-to-schema mapping

| Dataset | vibration_x | vibration_y | vibration_z | temperature | current | pressure | acoustic |
|---|---|---|---|---|---|---|---|
| C-MAPSS | sensor_11 | sensor_12 | sensor_14 | sensor_2 | sensor_9 | sensor_7 | sensor_21 |
| IMS | bearing RMS | kurtosis | peak | N/A (0) | N/A (0) | N/A (0) | bearing RMS |
| XJTU-SY | horiz. RMS | vert. RMS | horiz. kurtosis | N/A (0) | N/A (0) | N/A (0) | crest factor |
| Simulator | direct | direct | direct | direct | direct | direct | direct |
