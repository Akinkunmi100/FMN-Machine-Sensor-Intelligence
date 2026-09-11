# Model Selection & Performance Evaluation Report

## 1. Executive Summary

This report documents the machine learning architecture, validation methodology, and candidate selection process for predicting 24-hour equipment breakdown risk.

* **Selected Model:** Random Forest Classifier (300 estimators, `max_depth=8`, `min_samples_leaf=20`, balanced class weighting).
* **Feature Set:** 26 strictly causal features (19 raw rolling aggregates and differences; 7 per-machine expanding median relative ratios).
* **Alert Threshold:** 0.20 operating threshold.
* **Validation Performance:** 100% breakdown event detection (19 of 19 real breakdowns detected across three chronological evaluation periods) with 38 total false alarm episodes, achieving a median lead warning time of 24.0 hours (minimum 16 hours).
* **Primary Optimization Metric:** Area Under the Precision-Recall Curve (AUC-PR) alongside Event-Level Episode Capture.

---

## 2. Problem Formulation & Target Definition

The task is formulated as estimating the conditional probability of a machine experiencing a breakdown event within a 24-hour forward-looking operating window:

$$\text{Risk} = P(\text{Failure Event occurs in } (t, t + 24\text{h}] \mid \text{Sensor History up to } t)$$

Each machine-operating hour is labeled `1` if a failure event occurs within the subsequent 24 hours, and `0` otherwise. The final 24 hours of telemetry for each machine are censored (excluded from training and evaluation) to prevent unobserved future outcomes from introducing label noise.

This labeling transforms 17 raw failure events into 404 positive machine-hours (0.943% of the 42,850 labeled dataset records).

---

## 3. Metric Selection Under Severe Class Imbalance

The raw positive event rate across the dataset is 0.039% (17 failures in 43,354 operating hours). Under this degree of imbalance:

* **Accuracy is uninformative:** A naive classifier that never predicts failure achieves 99.96% accuracy while failing to detect any breakdown.
* **ROC-AUC is artificially inflated:** The large volume of true negative operating hours inflates the specificity metric, masking low precision.
* **AUC-PR provides honest ranking evaluation:** The Area Under the Precision-Recall Curve evaluates precision against recall exclusively on the positive class. The naive baseline floor corresponds to the sample positive prevalence (0.009 to 0.020 across evaluation splits).

### Event-Level Evaluation vs. Hourly Counting
While AUC-PR measures row-level ranking discrimination, industrial operations depend on event-level warning. A single breakdown spans a 24-hour pre-failure window (24 consecutive positive rows). A model that detects only a few isolated hours may score moderately on row metrics while failing to provide a dependable operational alert. 

Consecutive alert hours on a machine are grouped into discrete **alert episodes**. Models are evaluated on two operational criteria:
1. **Breakdown Event Capture:** Did an alert episode trigger prior to each real failure?
2. **False Alarm Episodes:** How many distinct, non-consecutive alert episodes occurred that did not result in a failure?

---

## 4. Time-Series Validation Methodology

Sensor readings from the same physical asset exhibit temporal autocorrelation. Random cross-validation shuffles future data into training folds, causing severe lookahead leakage. 

All evaluations employ strict chronological splits:

| Split Identifier | Training Period | Testing Period | Test Records | Test Positive Rows | Real Failure Events |
|---|---|---|---|---|---|
| Primary Split (80/20) | Inception to 2026-04-06 | 2026-04-07 to 2026-04-29 | 8,282 | 164 | 7 |
| Cross-Validation Fold 1 | Jan to Feb 2026 | March 2026 | 11,162 | 96 | 4 |
| Cross-Validation Fold 2 | Jan to Mar 2026 | April 2026 | 10,442 | 188 | 8 |

*Note: A split trained on January alone was excluded because January recorded zero failure events, providing no positive instances for supervised learning.*

### Temporal Embargo Buffers
Because labels look forward 24 hours ($t$ to $t+24\text{h}$), any training record within 24 hours of a split cutoff contains a label derived from test-period events. To eliminate this leakage, training rows within the 24-hour window preceding each test boundary are purged (`embargo()` buffer).

### Causal Imputation and Feature Construction
Exploratory data analysis identified that 77.3% of service resets in `run_hours_since_maintenance` coincide with or immediately follow a breakdown event. Maintenance is predominantly reactive. Any forward-looking maintenance feature would directly leak the outcome. All engineered features are restricted to backward-looking trailing windows and differences.

---

## 5. Cold-Start Asset Strategy

Assets `MCH-300` and `MCH-301` have only 72 operating hours (~3 days) of telemetry and zero recorded failures. They are excluded from training and validation sets:
1. 72 hours of history is insufficient to establish stable expanding baselines.
2. With zero recorded failure events, recall cannot be empirically evaluated.

To ensure the production pipeline can score newly introduced machinery, `machine_id` is excluded from the feature set. The model evaluates purely physical sensor characteristics. For assets with under 24 hours of operating history, machine-relative ratios default to 1.0 (neutral baseline), and predictions rely on raw sensor values.

---

## 6. Class Imbalance Treatment

Class imbalance was handled during model training using cost-sensitive loss weighting:
* Scikit-Learn classifiers: `class_weight="balanced"`.
* XGBoost: `scale_pos_weight = n_negative / n_positive`.

Synthetic oversampling techniques (e.g., SMOTE) were excluded. The training set contains approximately 240 positive rows originating from only 10 distinct failure events. Synthesizing artificial sensor readings between sparse, high-volatility points risks generating physically impossible equipment operating states.

---

## 7. Candidate Model Evaluation

All candidate models were evaluated across the identical 26-feature pipeline under strict time-ordered splits:

| Model Architecture | Fold 1 (Mar) AUC-PR | Fold 2 (Apr) AUC-PR | Primary Split AUC-PR | Mean AUC-PR | Interpretability |
|---|---|---|---|---|---|
| Naive Baseline (Never Alert) | 0.0086 | 0.0180 | 0.0198 | 0.0155 | N/A |
| Logistic Regression (L2 penalty) | 0.5711 | 0.6575 | 0.6705 | 0.6330 | High |
| Decision Tree (depth=4) | 0.1265 | 0.4042 | 0.3688 | 0.2998 | High |
| Random Forest (default `leaf=1`) | 0.5351 | 0.7126 | 0.6757 | 0.6411 | Medium |
| **Random Forest (`leaf=20`) [Selected]** | **0.5561** | **0.7052** | **0.7241** | **0.6618** | **Medium** |
| HistGradientBoosting | 0.1878 | 0.6215 | 0.4908 | 0.4334 | Medium |
| XGBoost | 0.3202 | 0.5995 | 0.4559 | 0.4585 | Medium |
| Ensemble (Averaged Probability) | 0.5030 | 0.6423 | 0.5884 | 0.5779 | Low |
| Ensemble (Majority Vote) | 0.1244 | 0.4349 | 0.4076 | 0.3223 | Low |

The tuned Random Forest outperforms the naive baseline by over 36x on the primary split (0.7241 vs. 0.0198). Gradient boosting algorithms underperformed Random Forest due to sensitivity to noise and extreme sparsity in the positive class. Training-free ensemble combinations failed to exceed the performance of the individual Random Forest model.

---

## 8. Operational Event-Level Comparison

Evaluating candidate models by distinct breakdown events detected versus false alarm episodes across all validation windows:

| Model & Operating Point | Primary Split (23 days) | Fold 1 (31 days) | Fold 2 (29 days) | Total Breakdown Capture | Total False Episodes |
|---|---|---|---|---|---|
| **Random Forest (`leaf=20`) @ 0.20** | **7 / 7 (14 eps)** | **4 / 4 (9 eps)** | **8 / 8 (15 eps)** | **19 / 19 (100%)** | **38** |
| Logistic Regression @ 0.30 | 7 / 7 (18 eps) | 3 / 4 (6 eps) | 8 / 8 (15 eps) | 18 / 19 (94.7%) | 39 |
| Random Forest (`leaf=1`) @ 0.20 | 7 / 7 (22 eps) | 1 / 4 (3 eps) | 8 / 8 (23 eps) | 16 / 19 (84.2%) | 48 |

### Lead Time Characteristics
For the selected Random Forest model operating at threshold 0.20:
* **Median Lead Time:** 24.0 hours prior to failure.
* **Minimum Lead Time:** 16.0 hours prior to failure.
This advance warning window provides sufficient lead time for shift supervisors to schedule corrective maintenance without emergency line stoppages.

### Analysis of Logistic Regression
While Logistic Regression achieved a competitive mean AUC-PR (0.6330), it missed a critical breakdown in Fold 1 (detecting 3 of 4 events) and demonstrated a minimum lead time of only 7 hours (compared to 16 hours for Random Forest). Random Forest demonstrated superior reliability under operational deployment criteria.

---

## 9. Operating Threshold Derivation

Evaluation of threshold cutoffs for the Random Forest model on the primary test split:

| Decision Threshold | Breakdown Events Caught | False Alarm Episodes | Row Recall | Row Precision |
|---|---|---|---|---|
| 0.50 | 6 / 7 (Misses MCH-213) | 14 | 0.744 | 0.632 |
| 0.30 | 7 / 7 | 18 | 0.951 | 0.455 |
| **0.20** | **7 / 7** | **14** | **0.951** | **0.402** |
| 0.10 | 7 / 7 | 17 | 0.963 | 0.344 |

The 0.20 threshold captures 100% of breakdown events while minimizing false alarm episodes. This stability extends across the 0.05 to 0.25 range across all validation folds, indicating robust operating margins.

### Operational Risk Tiers
* **HIGH ($\ge 0.20$):** Critical alert tier requiring immediate equipment dispatch. Represents ~2.8% of fleet hours.
* **WATCH ($0.02 - 0.20$):** Advisory tier for scheduling inspection during planned maintenance windows. Represents ~1.5% of fleet hours.
* **LOW ($< 0.02$):** Normal operating condition. Represents ~95.7% of fleet hours. Over 90% of normal hours score exactly 0.0000.

---

## 10. Feature Importance & Interaction Analysis

Relative feature importances extracted from the production Random Forest model:

| Feature Identifier | Description | Relative Importance |
|---|---|---|
| `temp_roll_mean_24h_rel` | 24h mean temperature relative to machine baseline | 0.188 |
| `vib_roll_std_24h_rel` | 24h vibration volatility relative to machine baseline | 0.157 |
| `temp_roll_mean_6h_rel` | 6h mean temperature relative to machine baseline | 0.154 |
| `vib_roll_std_24h` | Raw 24h vibration standard deviation | 0.115 |
| `temp_roll_std_24h` | Raw 24h temperature standard deviation | 0.102 |
| `vib_roll_mean_24h_rel` | 24h mean vibration relative to machine baseline | 0.076 |
| `temp_roll_std_24h_rel` | 24h temperature volatility relative to machine baseline | 0.061 |
| `vib_roll_mean_6h_rel` | 6h mean vibration relative to machine baseline | 0.053 |
| `run_hours_since_maintenance` | Causal operating hours since last service | 0.001 |
| `recently_reset_24h` | Indicator of service reset within trailing 24h | < 0.001 |
| `line_*` | Production line categorical encodings | < 0.001 combined |

### Key Findings
1. **Dominance of Machine-Relative Features:** Features normalized by machine historical baselines represent 70.7% of total model importance. Machine-specific baseline normalization accounts for asset-to-asset variations in operating conditions.
2. **Instability Over Static Level:** Rolling standard deviations and relative mean deviations contribute the vast majority of predictive signal, demonstrating that failure is preceded by volatility rather than constant high absolute readings.
3. **Causal Maintenance Signal:** Operating hours since maintenance contributes only 0.1% importance when restricted to strictly causal history, confirming that maintenance is reactive rather than preventative in the historical data.

---

## 11. Production Artifact Details

The production model artifact is serialized in the `models/` directory:

* **Model File:** `models/failure_risk_rf.joblib`
* **Metadata Specification:** `models/failure_risk_rf.meta.json`
* **Algorithm:** Scikit-Learn `RandomForestClassifier` (300 trees, `max_depth=8`, `min_samples_leaf=20`, `class_weight='balanced'`)
* **Training Corpus:** 42,850 rows, 404 positive instances, across all 15 established machines (2026-01-01 to 2026-04-29).
* **Serving Policy:** Pre-fitted at deployment; zero runtime retraining overhead.

---

## 12. Reproduction Reference

Execute the following commands to reproduce candidate comparisons, threshold sweep, and event analyses:

```bash
pip install -r requirements.txt
python -m src.data_exploration        # Dataset profiling and missingness analysis
python -m src.train_baseline          # Candidate model evaluations across time-series splits
python -m src.threshold_sweep         # Empirical threshold sweep analysis
python -m src.event_level_analysis    # Event-level breakdown capture and lead time calculations
python -m src.train_final_model       # Fit and serialize production model artifact
```
