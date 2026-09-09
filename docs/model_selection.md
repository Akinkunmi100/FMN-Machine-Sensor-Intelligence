# Model Selection — Predictive Maintenance

**Decision: Random Forest (balanced class weights), alert threshold 0.20.**

Every number below was produced by code in this repo and can be reproduced with
the commands in [How to reproduce](#how-to-reproduce). Nothing here is asserted
without the run that produced it.

---

## 1. What the model predicts

> **P(a failure event occurs on this machine within the next 24 hours)**

For every machine-hour `t`, the label is `1` if `failure_event = 1` occurs in the
window `(t, t+24h]`, otherwise `0`. The final 24 hours of each machine's history
are dropped rather than labelled `0` — we don't know their outcome, and calling
an unknown a negative would quietly poison the training set.

This expands 17 raw failure events into 404 positive machine-hours (0.943% of
42,850 labelled rows). **Both framings matter and they are not interchangeable:**
row-level metrics measure ranking quality, event-level metrics measure whether the
plant actually gets warned. Section 6 shows how badly the two can disagree.

## 2. Metric: AUC-PR

The raw positive rate is 0.039%. A model that predicts "no failure" forever scores
**99.96% accuracy** and is worthless. ROC-AUC is nearly as misleading here, because
a huge true-negative pool flatters the false-positive rate.

AUC-PR ignores true negatives entirely and answers the question that matters: of the
hours we flag, how many are real, across every possible threshold. The naive floor
is the positive rate itself (0.009–0.020 depending on split), which gives an honest
baseline to beat rather than a 99.96% number that means nothing.

## 3. Validation design

Hourly readings within a machine are strongly autocorrelated, so random k-fold
would leak neighbouring hours across the split and inflate every score. All
validation is time-ordered:

| Split | Train | Test | Test rows | Test positive rows | Real events |
|---|---|---|---|---|---|
| Primary 80/20 | → 2026-04-06 | 2026-04-07 → 04-29 | 8,282 | 164 | 7 |
| CV fold 1 | Jan–Feb | March | 11,162 | 96 | 4 |
| CV fold 2 | Jan–Mar | April | 10,442 | 188 | 8 |

A third fold trained on January alone was deliberately **not** used: January
contains zero failures, so there would be no positive class to fit against.

### 24-hour embargo

The label at time `t` depends on data up to `t+24h`. A training row within 24 hours
of the cutoff therefore has a label partly determined by test-side data — leakage
through the *target*, which no amount of careful feature engineering would catch.
Every split purges training rows whose horizon window crosses its cutoff
(`embargo()` in `src/train_baseline.py`).

### Leakage guard on maintenance resets

Phase 0 found that **77.3% of the 22 maintenance resets coincide with or immediately
follow the failure they would appear to predict** — resets are largely *reactive*.
Any forward-looking feature built on `run_hours_since_maintenance` ("hours until next
reset") would be a near-copy of the label.

Only backward-looking use is permitted. This is enforced, not just intended:
`src/test_no_leakage.py` recomputes every feature on truncated data and asserts that
values for already-past rows are unchanged. A feature that peeks forward changes when
the future is removed; these do not.

```
OK: all 19 features are causal (unchanged across 25926 rows when future data
is truncated).
```

### Cold-start machines

MCH-300 and MCH-301 (72 rows / 2.96 days each, zero failures) are **excluded from
training and evaluation** — three days of history cannot support a 24-hour horizon
label, and zero positives means nothing to learn or score against.

So that they remain *scoreable*, `machine_id` is deliberately **not** a feature. The
model reads only physically-grounded signals, so it generalises to machines it has
never seen. Their scores are real predictions, but **unvalidated** — see Limitations.

## 4. Class imbalance

Handled at fit time, not only through metric choice: `class_weight="balanced"` for
Logistic Regression, Decision Tree, Random Forest and HistGradientBoosting;
`scale_pos_weight = n_neg/n_pos` for XGBoost.

SMOTE was considered and rejected. After the primary split the training set holds
~240 positive rows drawn from only 10 real events; interpolating synthetic sensor
readings between so few genuine failures risks inventing machine states that cannot
physically occur.

## 5. Candidate comparison (AUC-PR)

| Model | Fold 1 (Mar) | Fold 2 (Apr) | Primary | Interpretability |
|---|---|---|---|---|
| Naive baseline (never alert) | 0.0086 | 0.0180 | 0.0198 | — |
| Logistic Regression | 0.0301 | 0.3438 | 0.3957 | High — readable coefficients |
| Decision Tree (depth 4) | 0.2227 | 0.3397 | 0.3136 | High — visualisable rules |
| **Random Forest** | **0.4016** | 0.5773 | **0.6411** | Medium — feature importances |
| HistGradientBoosting | 0.3499 | 0.5730 | 0.5516 | Medium |
| XGBoost | 0.3492 | 0.5334 | 0.4758 | Medium |
| Ensemble — averaged probability | 0.3896 | **0.6059** | 0.5834 | Low |
| Ensemble — majority vote | 0.2538 | 0.3906 | 0.4341 | Low |

**Random Forest beats the naive baseline by 32x on the primary split** (0.641 vs
0.020) and leads every split except fold 2, where the averaged ensemble edges ahead.

Three findings worth stating plainly:

- **Boosting did not help.** HistGradientBoosting and XGBoost both trail Random
  Forest on all three splits. Further variants (LightGBM, CatBoost, AdaBoost) were
  deliberately not added — with 164 positive test rows, more candidates buy noise,
  not signal.
- **No stacked ensemble was built.** With 17 real events, a trained meta-learner
  would fit which model got lucky on which failure, not a real pattern. The two
  training-free combinations tested don't beat the best single model on 2 of 3
  splits, and majority vote is clearly worst — binarising at 0.5 before combining
  discards the ranking information AUC-PR rewards.
- **Logistic Regression is unstable.** At 0.0301 on fold 1 it is barely above the
  0.0086 naive floor. Trained on the least data, the interpretable model essentially
  stops working — which matters, because the system will be retrained as history
  accumulates.

## 6. Why row-level metrics chose the wrong model

At threshold 0.5 on the primary split, Logistic Regression looks *safer*:

| Model | TP | FP | FN | Recall | Precision |
|---|---|---|---|---|---|
| Logistic Regression @ 0.5 | 162 | 557 | **2** | 0.988 | 0.225 |
| Random Forest @ 0.5 | 115 | 119 | 49 | 0.701 | 0.491 |
| Random Forest @ 0.2 | 140 | 230 | 24 | 0.854 | 0.378 |

Two missed hours versus forty-nine looks decisive. **It isn't** — those are *hours*,
not breakdowns. The 164 positive rows are 7 real failures, each inflated into a
24-hour window. Missing 49 hours of a 24-hour warning window says nothing about
whether the plant got warned.

Re-scored on what operations actually experiences — did *any* alert fire before each
real failure, and how many separate false alarms did operators have to chase
(consecutive alert hours on one machine = **one** episode, not 24):

| Split | Random Forest @ 0.2 | Logistic Regression @ 0.3 |
|---|---|---|
| Primary (23 d) | **7/7 caught**, 22 false episodes | 7/7 caught, 34 false episodes |
| Fold 1 (31 d) | **4/4 caught**, 6 false episodes | **3/4 caught**, 27 false episodes |
| Fold 2 (29 d) | **8/8 caught**, 26 false episodes | 8/8 caught, 70 false episodes |
| **Total** | **19/19 events, 54 false episodes** | 18/19 events, 131 false episodes |

Random Forest catches **every failure in all three windows with 59% fewer false
alarms**. Logistic Regression — the supposedly safer model — is the one that misses
an event, on the same fold where its AUC-PR collapsed.

**Median warning time: 24.0 hours; minimum observed: 22 hours.** A full shift of
notice, enough to move the repair into a planned window instead of reacting to a
breakdown.

## 7. Threshold: 0.20

Selected on the primary split, then **verified on both CV folds before adoption** —
tuning a threshold on the window you then report it on is soft leakage of the same
family this document has been guarding against.

| Threshold | Events caught (primary) | False episodes | Note |
|---|---|---|---|
| 0.5 | 6/7 — misses MCH-213 (2026-04-08) | 22 | |
| 0.3 | 6/7 — misses MCH-213 | 23 | |
| **0.20** | **7/7** | **22** | Catches MCH-213 at no extra alarm cost |

Dropping from 0.3 to 0.2 converts a missed breakdown into a caught one while the
false-alarm episode count stays flat (23 → 22). The extra alert *hours* land inside
warning windows that were already firing, so operators see no additional noise.

Held up out of sample: 4/4 on fold 1 (6 false episodes) and 8/8 on fold 2 (26).

## 8. Why this decision does not depend on cost assumptions

Normally this call requires pricing unplanned downtime against an unnecessary
inspection — a ratio this dataset does not contain and which is not invented here.

**It isn't needed.** Random Forest @ 0.2 is better on *both* axes simultaneously —
more failures caught *and* fewer false alarms. When one option dominates, the
exchange rate between the two costs is irrelevant; every ratio gives the same answer.

The supporting argument is about trust rather than arithmetic. At ~0.45 false
episodes per machine per week, the plant sees roughly one false alarm per day across
15 machines. Logistic Regression's 1.13/machine/week on fold 2 is where operators
begin dismissing alerts — and a model whose warnings get ignored has an effective
recall of zero regardless of its metrics.

### On the interpretability preference

The project rule is to prefer interpretable models *where performance is comparable*.
At 0.641 vs 0.396 AUC-PR, with a missed event and a near-baseline fold, performance
is not comparable. Operator trust is instead delivered by the Phase 3 explanation
layer, grounded in each machine's real sensor values, plus the feature importances
below — not by reading regression coefficients, which no one on a plant floor does.

## 9. What the model actually keys on

| Feature | Importance |
|---|---|
| `vib_roll_std_24h` | 0.258 |
| `temp_roll_std_24h` | 0.204 |
| `vib_roll_std_6h` | 0.122 |
| `vib_roll_mean_24h` | 0.106 |
| `vib_roll_mean_6h` | 0.072 |
| `vibration_mm_s` (raw) | 0.060 |
| `temp_roll_mean_24h` | 0.057 |
| `temp_roll_std_6h` | 0.051 |
| `temp_roll_mean_6h` | 0.037 |
| `temperature_c` (raw) | 0.017 |
| `run_hours_since_maintenance` | 0.007 |
| `recently_reset_24h` | 0.0004 |
| `line_*` (all three) | 0.0006 combined |

**Instability predicts failure, not absolute level.** Rolling standard deviations
alone carry ~63% of total importance; raw instantaneous readings contribute under 8%
combined. Machines don't fail because they are hot — they fail because they start
running *erratically*. Vibration features outweigh temperature roughly 62% to 37%.

Two results worth flagging honestly:

- **`run_hours_since_maintenance` is nearly irrelevant (0.007).** The intuitive
  "wear since last service" hypothesis is not what drives these predictions. This is
  also a quiet vindication of the leakage guard: restricted to causal use the feature
  is almost worthless, whereas a forward-looking version would have looked
  spectacularly predictive by encoding the reactive reset that *follows* the failure.
- **`line` contributes ~0.0006.** Production line carries no failure signal, which
  strengthens the cold-start case: the model is not leaning on plant topology it
  cannot verify for a new machine.

## 10. Limitations

- **19 events is a small sample.** 19/19 is a real result over ~83 days of held-out
  testing, not a guarantee. It should be described to the plant team as
  "caught every failure in testing", never as "never misses".
- **Roughly 3 of 4 alert episodes are false** (row-level precision 0.378). Acceptable
  given the cost asymmetry, but the plant team must be told upfront so the
  expectation is calibrated from day one.
- **Cold-start machines are unvalidated, not validated-good.** MCH-300/301 have zero
  failures, so their event-level recall is *unmeasurable*. Their dashboard scores
  should carry a visible lower-confidence marker.
- **Historical scores in the app will be in-sample.** The shipped artifact is refit
  on all established history, so re-scoring that same history for the trend view
  produces optimistic risk curves. **Phase 4 should generate walk-forward
  out-of-sample scores for the trend chart** rather than scoring history with the
  final model.
- **Missingness treatment is forward-fill**, justified by ~99% of gaps being isolated
  single hours (max run 2). A future feed with longer outages would need revisiting.
- **Failures are near-uniform across machines** (1–2 each), so the model cannot learn
  machine-specific failure modes — by design, since that is also what lets it score
  new machines.

## 11. The shipped artifact

| | |
|---|---|
| File | `models/failure_risk_rf.joblib` |
| Metadata | `models/failure_risk_rf.meta.json` |
| Trained on | 42,850 rows, 404 positive (0.943%), 15 established machines |
| Date range | 2026-01-01 → 2026-04-29 |
| Threshold | 0.20 |
| Excluded | MCH-300, MCH-301 (scoreable, not trained on) |

Refit on all established history after validation — same hyperparameters, more data.
The app loads this file; nothing is retrained at request time.

## How to reproduce

```bash
pip install -r requirements.txt
python src/data_exploration.py      # Phase 0 findings
python src/test_no_leakage.py       # causality guard (must pass)
python src/train_baseline.py        # candidate comparison table
python src/threshold_sweep.py       # threshold sweep
python src/event_level_analysis.py  # event-level analysis + generalisation check
python src/train_final_model.py     # fit and persist the selected model
```
