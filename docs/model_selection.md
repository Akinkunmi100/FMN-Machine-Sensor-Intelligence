# Model Selection — Predictive Maintenance

**Decision: Random Forest (balanced class weights, `min_samples_leaf=20`), 26 features
including per-machine relative ratios, alert threshold 0.20.**

Every number below was produced by code in this repo and can be reproduced with
the commands in [How to reproduce](#how-to-reproduce). Nothing here is asserted
without the run that produced it.

> **Revised after Phase 2.** The originally shipped model used 19 absolute
> features with `min_samples_leaf=1`. A deliberate re-examination found a
> materially better configuration — same event capture, **30% fewer false
> alarms**. Section 10 documents what changed, what didn't, and the two things
> I initially got wrong. The earlier version is preserved in git history at
> commit `74b9e60`.

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

**AUC-PR still does not decide the model.** It ranks; it does not tell you whether
a breakdown got a warning. Sections 6 and 10 both turn on that distinction.

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

### Leakage guard

Phase 0 found that **77.3% of the 22 maintenance resets coincide with or immediately
follow the failure they would appear to predict** — resets are largely *reactive*.
Any forward-looking feature built on `run_hours_since_maintenance` would be a
near-copy of the label.

The relative features added in this revision raise the stakes: they use *expanding*
per-machine statistics, exactly the construction where causality quietly breaks. The
baseline is `shift(1).expanding(min_periods=24).median()` — strictly backward-only.

This is enforced, not intended. `src/test_no_leakage.py` recomputes every feature on
truncated data and asserts that values for already-past rows are unchanged. A feature
that peeks forward changes when the future is removed; these do not.

```
OK: all 26 features are causal (unchanged across 25926 rows when future data
is truncated).
```

### Cold-start machines

MCH-300 and MCH-301 (72 rows / 2.96 days each, zero failures) are **excluded from
training and evaluation** — three days of history cannot support a 24-hour horizon
label, and zero positives means nothing to learn or score against.

So that they remain *scoreable*, `machine_id` is deliberately **not** a feature. The
model reads only physically-grounded signals, so it generalises to machines it has
never seen.

**A relative feature needs 24 hours of a machine's own history before it has a
baseline; below that it falls back to `1.0` ("at its own normal") and the model
leans on the absolute features instead.** That fallback is a design choice, not a
measured benefit — with zero recorded failures on those machines, nothing about
their accuracy can be validated. Their scores are real predictions, but
**unvalidated**. See Limitations.

## 4. Class imbalance

Handled at fit time, not only through metric choice: `class_weight="balanced"` for
Logistic Regression, Decision Tree, Random Forest and HistGradientBoosting;
`scale_pos_weight = n_neg/n_pos` for XGBoost.

SMOTE was considered and rejected. After the primary split the training set holds
~240 positive rows drawn from only 10 real events; interpolating synthetic sensor
readings between so few genuine failures risks inventing machine states that cannot
physically occur.

## 5. Candidate comparison (AUC-PR)

All candidates evaluated on the same 26 features. Every model except the shipped
Random Forest is left at its original untuned settings, so the *family* comparison
stays like-for-like; the tuned configuration is shown as its own row.

| Model | Fold 1 (Mar) | Fold 2 (Apr) | Primary | Mean | Interpretability |
|---|---|---|---|---|---|
| Naive baseline (never alert) | 0.0086 | 0.0180 | 0.0198 | 0.0155 | — |
| Logistic Regression | 0.5711 | 0.6575 | 0.6705 | 0.6330 | High |
| Decision Tree (depth 4) | 0.1265 | 0.4042 | 0.3688 | 0.2998 | High |
| Random Forest (`leaf=1`) | 0.5351 | **0.7126** | 0.6757 | 0.6411 | Medium |
| **Random Forest (`leaf=20`) — SHIPPED** | **0.5561** | 0.7052 | **0.7241** | **0.6618** | Medium |
| HistGradientBoosting | 0.1878 | 0.6215 | 0.4908 | 0.4334 | Medium |
| XGBoost | 0.3202 | 0.5995 | 0.4559 | 0.4585 | Medium |
| Ensemble — averaged probability | 0.5030 | 0.6423 | 0.5884 | 0.5779 | Low |
| Ensemble — majority vote | 0.1244 | 0.4349 | 0.4076 | 0.3223 | Low |

**The shipped model beats the naive baseline by 37× on the primary split**
(0.7241 vs 0.0198).

Three findings worth stating plainly:

- **Boosting got *worse* with relative features.** HistGradientBoosting fell from
  0.5516 to 0.4908 on the primary split, XGBoost from 0.4758 to 0.4559. The gain
  from per-machine normalisation is specific to Random Forest and Logistic
  Regression. Further boosting variants were not added — with 164 positive test
  rows, more candidates buy noise, not signal.
- **No stacked ensemble was built.** With 17 real events, a trained meta-learner
  would fit which model got lucky on which failure, not a real pattern. The two
  training-free combinations tested lose to the best single model, and majority
  vote is clearly worst — binarising at 0.5 before combining discards the ranking
  information AUC-PR rewards.
- **Logistic Regression was rescued by the relative features** — see Section 7. It
  is now genuinely competitive on ranking, and the reason it previously failed
  turns out to be diagnostic rather than incidental.

## 6. Why row-level metrics choose the wrong model

The 164 positive rows in the primary test set are **7 real breakdowns**, each
inflated into a 24-hour window. Row-level counts measure hours, not breakdowns.
Re-scored on what operations actually experiences — did *any* alert fire before each
real failure, and how many separate false alarms did operators have to chase
(consecutive alert hours on one machine = **one** episode, not 24) — the ranking
changes:

| Model | Primary (23 d) | Fold 1 (31 d) | Fold 2 (29 d) | **Total** |
|---|---|---|---|---|
| **RF `leaf=20` @ 0.20 — SHIPPED** | 7/7 · 14 eps | 4/4 · 9 eps | 8/8 · 15 eps | **19/19 · 38 eps** |
| Logistic Regression @ 0.30 | 7/7 · 18 eps | **3/4** · 6 eps | 8/8 · 15 eps | 18/19 · 39 eps |
| RF `leaf=1` @ 0.20 | 7/7 · 22 eps | **1/4** · 3 eps | 8/8 · 23 eps | 16/19 · 48 eps |

**Warning time: median 24.0 hours, minimum observed 16 hours.** Enough to move a
repair into a planned window rather than reacting to a breakdown. (The previous
model's minimum was 22h — the tightest warning shrank by six hours, which is the
one measurable regression in this revision. 16 hours is still most of two shifts.)

## 7. The interpretability question, reopened and re-closed

Phase 2 rejected Logistic Regression partly because it **collapsed to 0.0301 on
fold 1** — barely above the 0.0086 naive floor. With relative features it scores
**0.5711 on that same fold, beating the Random Forest there.**

That collapse was diagnostic, not incidental. *A linear model cannot express "high
for this machine" from absolute values.* Given ratios, it can. The feature change
fixed the exact weakness that disqualified it.

This matters because the project rule is to prefer interpretable models **where
performance is comparable**. The gap closed from 0.641-vs-0.396 to 0.662-vs-0.633 —
close enough that the rule deserved a real hearing rather than a restatement of the
earlier verdict.

It still doesn't bite, for two reasons that only appear at the operational level:

1. **Logistic Regression misses a breakdown** (18/19 vs 19/19), on the same fold
   where it previously collapsed.
2. **Its minimum lead time is 7 hours against the Random Forest's 16.** A
   seven-hour warning barely permits scheduling; it forces a reactive repair.

Comparable ranking, materially worse operational behaviour. The Random Forest holds.

## 8. Threshold: 0.20

| Threshold | Events caught (primary) | False episodes | Row recall | Row precision |
|---|---|---|---|---|
| 0.50 | 6/7 — misses MCH-213 | 14 | 0.744 | 0.632 |
| 0.30 | 7/7 | 18 | 0.951 | 0.455 |
| **0.20** | **7/7** | **14** | 0.951 | 0.402 |

Confirmed out of sample before adoption: 4/4 on fold 1, 8/8 on fold 2.

**The threshold now sits mid-plateau rather than on a cliff edge.** The previous
model held perfect capture only up to 0.20 — one step further, at 0.25, it dropped
an event. The shipped model holds 19/19 from 0.05 through 0.25, bottoming out at
38 episodes across 0.20–0.25. That margin on both sides is a robustness gain
against distribution drift, independent of the false-alarm improvement, and it
means the signed-off bands carry over unchanged.

## 9. What the model actually keys on

| Feature | Importance |
|---|---|
| `temp_roll_mean_24h_rel` | 0.188 |
| `vib_roll_std_24h_rel` | 0.157 |
| `temp_roll_mean_6h_rel` | 0.154 |
| `vib_roll_std_24h` | 0.115 |
| `temp_roll_std_24h` | 0.102 |
| `vib_roll_mean_24h_rel` | 0.076 |
| `temp_roll_std_24h_rel` | 0.061 |
| `vib_roll_mean_6h_rel` | 0.053 |
| `run_hours_since_maintenance` | 0.001 |
| `recently_reset_24h` | 0.00001 |
| `line_*` (all three) | 0.00003 combined |

**Relative features carry 70.7% of total importance.** Offered both views of the
same signal, the model overwhelmingly prefers "how unusual is this *for this
machine*" over "how large is this in absolute terms". That is the single clearest
result in this document, and it is why the revision was worth making.

Two further results worth flagging:

- **`run_hours_since_maintenance` fell to 0.001** (from 0.007). The intuitive "wear
  since last service" hypothesis is not merely weak — it is essentially unused. This
  also quietly vindicates the leakage guard: restricted to causal use the feature is
  worthless, whereas a forward-looking version would have looked spectacularly
  predictive by encoding the reactive reset that *follows* the breakdown.
- **Temperature now slightly outweighs vibration** (52.3% vs 47.6%), reversing the
  62/37 split under absolute-only features. Temperature's absolute level varies a
  lot between machines, which buried its signal; normalised per machine, it becomes
  the single strongest driver.

## 10. What the re-examination changed

Four decisions were re-tested after Phase 2. Two held, two did not.

| Decision | Verdict |
|---|---|
| Random Forest over LogReg / boosting | **Held** — re-confirmed on new features, at event level |
| Bands 0.02 / 0.20 | **Held** — verified against walk-forward OOS, not just in-sample |
| Absolute-only features | **Changed** — added 7 per-machine relative ratios |
| `min_samples_leaf=1` | **Changed** — 20 |

Two things I initially got wrong, recorded because the reasoning is the useful part:

- **I suspected the bands were derived from a contaminated distribution** — computed
  on the final refit model, which has seen every failure. Walk-forward out-of-sample
  scores cleared them: 90.52% of OOS hours score exactly 0.0000 against 92.99%
  in-sample, and the watch tier actually gets *more* useful out-of-sample (2.21% vs
  1.16%). The suspicion was worth checking and wrong.
- **I briefly preferred `min_samples_leaf=20` on its AUC-PR gain alone** (+0.076).
  At matched event capture it is *worse* on its own — **+17 false episodes**. That is
  the same error Section 6 exists to prevent, made against my own tuning rather than
  against a rival model. It only wins in combination with the relative features.

**Neither change helps alone.** Regularisation alone costs +17 episodes; relative
features alone cost +8; together they gain −16. Ratio features are higher-variance
and need the heavier leaf constraint, and the constraint needs richer features to
exploit. Testing them one at a time would have rejected both. The clearest evidence
is fold 1, where `leaf=1` on the new features catches only **1 of 4** breakdowns
against `leaf=20`'s 4 of 4.

I also motivated relative features partly as a cold-start fix. **That claim is
withdrawn** — the measured gain is on established machines only, and cannot be
otherwise, since the cold-start machines have no failures against which recall could
be measured. Below 24 hours of history the features fall back to `1.0`, so new
machines get *less* signal from them, not more.

## 11. Limitations

- **19 events is a small sample.** 19/19 is a real result over ~83 days of held-out
  testing, not a guarantee. Describe it to the plant team as "caught every failure in
  testing", never as "never misses".
- **Roughly 3 of 4 alert episodes are false** (row precision 0.402 at threshold
  0.20). Acceptable given the cost asymmetry, but the plant team must be told upfront
  so the expectation is calibrated from day one.
- **Minimum lead time fell from 22h to 16h** in this revision — the one measurable
  regression against the previous model, accepted for a 30% cut in false alarms.
- **Cold-start machines are unvalidated, not validated-good.** MCH-300/301 have zero
  failures, so their event-level recall is *unmeasurable*. Their dashboard scores need
  a visible lower-confidence marker, and the relative-feature fallback makes them more
  reliant on absolute features than any established machine.
- **Historical scores in the app will be in-sample.** The shipped artifact is refit on
  all established history, so re-scoring that history produces optimistic curves —
  MCH-213 scores 0.987 at a moment the honest held-out model gave 0.225. **The Phase 4
  trend chart must use walk-forward out-of-sample scores**, which cover 52.9% of
  history (nothing before 2026-02-26, since earlier training windows contain fewer
  than two failures).
- **Missingness treatment is forward-fill**, justified by ~99% of gaps being isolated
  single hours (max run 2). A feed with longer outages would need revisiting.
- **Hyperparameters were tuned by inspecting held-out splits.** The defence is that
  the winning configuration leads on folds 1 and 2 independently *and* the primary
  split — untouched by that choice — confirms it (+0.083 AUC-PR, 22→14 episodes). A
  fully nested protocol would be stricter.

## 12. The shipped artifact

| | |
|---|---|
| File | `models/failure_risk_rf.joblib` |
| Metadata | `models/failure_risk_rf.meta.json` |
| Model | RandomForest, 300 trees, `max_depth=8`, `min_samples_leaf=20`, balanced |
| Features | 26 (19 absolute + 7 per-machine relative) |
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
python src/test_no_leakage.py       # causality guard over all 26 features (must pass)
python src/train_baseline.py        # candidate comparison table
python src/threshold_sweep.py       # threshold sweep
python src/event_level_analysis.py  # event-level analysis + generalisation check
python src/train_final_model.py     # fit and persist the selected model
```
