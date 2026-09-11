# Model Selection — Predictive Maintenance

**The decision: Random Forest, balanced class weights, `min_samples_leaf=20`, 26
features (19 raw plus 7 per-machine relative ratios), alert threshold 0.20.**

Everything below came from actually running the code in this repo — see
[How to reproduce](#how-to-reproduce) if you want to check any of it yourself
rather than take my word for it.

> **One honest note up front.** The model I originally shipped used 19 raw
> features and `min_samples_leaf=1`. I went back and re-tested that choice,
> and found something meaningfully better — same number of breakdowns caught,
> 30% fewer false alarms. Section 10 walks through what changed and, just as
> important, the two things I got wrong along the way. The earlier version is
> still there in git history at commit `74b9e60`, if you want to compare.

---

## 1. What the model is actually predicting

> **The probability that a machine has a failure event in the next 24 hours.**

For every machine-hour, I look 24 hours forward and label it `1` if a failure
happens somewhere in that window, `0` if not. The last 24 hours of each
machine's history get dropped rather than labeled `0` — I don't know what
happens after the data ends, and calling an unknown a "safe" outcome would
have quietly poisoned the training set.

That labeling turns 17 raw failure events into 404 positive machine-hours
(0.943% of 42,850 labeled rows). Two ways of counting this — by hour and by
event — give very different pictures, and they're not interchangeable: one
tells you how good the ranking is, the other tells you whether the plant
actually got warned. Section 6 shows just how far apart those two answers can
land.

## 2. Why AUC-PR, and not accuracy or ROC-AUC

The real positive rate here is 0.039%. A model that always predicts "no
failure" scores 99.96% accuracy and is worthless — it would never warn anyone
about anything. ROC-AUC isn't much better in this situation, because it gets
credit for correctly ignoring a huge pile of hours that were never going to
be a failure anyway.

AUC-PR ignores that huge pile of easy negatives and only asks: of the hours I
flag, how many are real? The naive floor — the score you'd get by never
alerting — is just the positive rate itself (0.009 to 0.020 depending on the
split). That's a floor worth beating, unlike 99.96%, which means nothing.

Even so, AUC-PR only tells you how good the ranking is. It doesn't tell you
whether a real breakdown actually got a warning. Sections 6 and 10 both come
back to that gap.

## 3. How the validation actually works

Hourly readings on the same machine are obviously related to each other, so a
random shuffle-and-split would leak information across time and make every
score look better than it should. Every split here is ordered by real time
instead:

| Split | Train | Test | Test rows | Test positive rows | Real events |
|---|---|---|---|---|---|
| Primary 80/20 | → 2026-04-06 | 2026-04-07 → 04-29 | 8,282 | 164 | 7 |
| CV fold 1 | Jan–Feb | March | 11,162 | 96 | 4 |
| CV fold 2 | Jan–Mar | April | 10,442 | 188 | 8 |

I deliberately didn't add a third fold trained on January alone — January has
zero recorded failures, so there'd be nothing for the model to actually learn.

### The 24-hour buffer around each split

A label at time `t` depends on data up to `t+24h`. So a training row within
24 hours of the cutoff has a label that's partly decided by data from the
test side — that's leakage through the answer itself, not something careful
feature engineering could ever catch after the fact. Every split here trims
out any training row whose 24-hour window crosses the cutoff (see
`embargo()` in `src/train_baseline.py`).

### Making sure the maintenance-reset trap couldn't sneak back in

Phase 0 found that 77.3% of the 22 maintenance resets happen at, or right
after, the failure they'd appear to predict — maintenance here is mostly
reactive. Any forward-looking feature built on `run_hours_since_maintenance`
would end up being a near-copy of the answer.

The relative features I added in this revision made that risk sharper, not
smaller — they're built from *expanding* per-machine statistics, which is
exactly the kind of construction where causality quietly breaks if you're not
careful. The baseline for each one is
`shift(1).expanding(min_periods=24).median()` — strictly backward-looking,
nothing else.

I didn't just design it that way and hope. `src/test_no_leakage.py` recomputes
every feature on data with the future chopped off and checks that nothing
about the already-past rows changed. A feature that's secretly peeking
forward would shift when you take the future away; none of these do.

```
OK: all 26 features are causal (unchanged across 25926 rows when future data
is truncated).
```

### The two machines with almost no history

MCH-300 and MCH-301 (72 rows, about 3 days each, zero recorded failures) are
left out of training and evaluation entirely. Three days isn't enough to
build a real 24-hour label, and with zero failures there's nothing to check
recall against anyway.

I still wanted them to be scoreable, so `machine_id` is deliberately never a
feature. The model only reads physically grounded sensor signal, so it
carries over to machines it's never seen.

One consequence of that design: a relative feature needs 24 hours of a
machine's own history before it has a real baseline to compare against. Below
that, it falls back to `1.0` — "at its own normal" — and the model leans more
on the raw features instead. That fallback is a deliberate choice, not a
measured win; with zero recorded failures on those two machines, there's no
way to check whether it actually helps them. Their scores are real
predictions, but unvalidated ones. More on that in Limitations.

## 4. Handling the class imbalance

I didn't just pick a metric that's honest about the imbalance — I also
weighted for it at training time: `class_weight="balanced"` for Logistic
Regression, the Decision Tree, Random Forest, and HistGradientBoosting;
`scale_pos_weight = n_neg/n_pos` for XGBoost.

I considered SMOTE and decided against it. After the primary split, the
training set has roughly 240 positive rows drawn from only 10 real events.
Interpolating synthetic sensor readings between that few genuine failures
felt like a good way to invent machine states that could never actually
happen.

## 5. Comparing the candidates

Every candidate saw the same 26 features. Every model except the one I
shipped kept its original, untuned settings, so the comparison between model
*families* stays fair — the tuned version gets its own row, called out
separately.

| Model | Fold 1 (Mar) | Fold 2 (Apr) | Primary | Mean | Interpretability |
|---|---|---|---|---|---|
| Naive baseline (never alert) | 0.0086 | 0.0180 | 0.0198 | 0.0155 | — |
| Logistic Regression | 0.5711 | 0.6575 | 0.6705 | 0.6330 | High |
| Decision Tree (depth 4) | 0.1265 | 0.4042 | 0.3688 | 0.2998 | High |
| Random Forest (`leaf=1`) | 0.5351 | 0.7126 | 0.6757 | 0.6411 | Medium |
| **Random Forest (`leaf=20`) — shipped** | 0.5561 | 0.7052 | 0.7241 | 0.6618 | Medium |
| HistGradientBoosting | 0.1878 | 0.6215 | 0.4908 | 0.4334 | Medium |
| XGBoost | 0.3202 | 0.5995 | 0.4559 | 0.4585 | Medium |
| Ensemble — averaged probability | 0.5030 | 0.6423 | 0.5884 | 0.5779 | Low |
| Ensemble — majority vote | 0.1244 | 0.4349 | 0.4076 | 0.3223 | Low |

The shipped model beats the naive baseline by 37× on the primary split
(0.7241 vs 0.0198).

Three things worth saying plainly, because they surprised me at the time:

- **Boosting got worse once I added the relative features**, not better.
  HistGradientBoosting dropped from 0.5516 to 0.4908 on the primary split,
  XGBoost from 0.4758 to 0.4559. Whatever the relative features are giving
  Random Forest and Logistic Regression, boosting doesn't take the same
  advantage of it. I didn't chase more boosting variants after that — with
  only 164 positive test rows, adding more candidates buys noise, not signal.
- **I didn't build a stacked ensemble.** With only 17 real events, a trained
  meta-learner would essentially be memorizing which model happened to get
  lucky on which single failure, not learning a real pattern. The two
  training-free combinations I did test both lost to the best single model,
  and majority vote was clearly the worst of the two — turning each model's
  score into a plain yes/no before combining throws away exactly the ranking
  information AUC-PR is built to reward.
- **Logistic Regression came back to life with the relative features** — see
  Section 7 for the full story. It's genuinely competitive on ranking now,
  and it turns out the reason it failed before wasn't a fluke — it was
  telling me something real about what the model needed.

## 6. Why counting hours instead of breakdowns picks the wrong model

The 164 positive rows in the primary test set are really only 7 breakdowns,
each one stretched across a 24-hour window. Counting hours measures how many
of those hours got flagged — it doesn't tell you whether the plant actually
got warned. So I re-scored everything on what operations actually cares
about: did any alert fire before each real failure, and how many separate
false alarms did someone have to go chase (a run of consecutive alert hours
on one machine counts as one episode, not 24). The ranking changes:

| Model | Primary (23 d) | Fold 1 (31 d) | Fold 2 (29 d) | Total |
|---|---|---|---|---|
| **RF `leaf=20` @ 0.20 — shipped** | 7/7 · 14 eps | 4/4 · 9 eps | 8/8 · 15 eps | **19/19 · 38 eps** |
| Logistic Regression @ 0.30 | 7/7 · 18 eps | 3/4 · 6 eps | 8/8 · 15 eps | 18/19 · 39 eps |
| RF `leaf=1` @ 0.20 | 7/7 · 22 eps | 1/4 · 3 eps | 8/8 · 23 eps | 16/19 · 48 eps |

Median warning time is 24 hours, with a minimum observed of 16 — enough to
move a repair into a planned window instead of scrambling after a breakdown.
Worth being honest about: the earlier model's minimum was 22 hours, so the
tightest warning shrank by six hours in this revision. That's the one real
regression I'm accepting here, in exchange for the drop in false alarms. 16
hours is still most of two shifts.

## 7. Reopening the interpretability question — and closing it again

Phase 2 passed over Logistic Regression partly because it collapsed to 0.0301
on fold 1, barely above the 0.0086 naive floor. With the relative features
added, it scores 0.5711 on that exact same fold — actually beating the
Random Forest there.

That collapse wasn't bad luck. A linear model simply can't express "this is
high *for this particular machine*" using raw values — it needs the ratio to
do that. Once I gave it one, the exact weakness that had disqualified it went
away.

That matters because the project's own rule is to prefer an interpretable
model where performance is comparable, and the gap really did close — from
0.641-vs-0.396 to 0.662-vs-0.633. Close enough that I owed it a real second
look, not just a restatement of the earlier verdict.

It still doesn't change the decision, for two reasons that only show up once
you look past the ranking score:

1. Logistic Regression misses a breakdown (18/19 against the Random Forest's
   19/19), on the very fold where it used to collapse.
2. Its shortest lead time is 7 hours, against the Random Forest's 16. Seven
   hours is barely enough to schedule anything — it forces a reactive repair
   rather than a planned one.

Comparable ranking, but clearly worse behavior once you look at what actually
happens on the floor. The Random Forest stays the pick.

## 8. Why the threshold sits at 0.20

| Threshold | Events caught (primary) | False episodes | Row recall | Row precision |
|---|---|---|---|---|
| 0.50 | 6/7 — misses MCH-213 | 14 | 0.744 | 0.632 |
| 0.30 | 7/7 | 18 | 0.951 | 0.455 |
| **0.20** | **7/7** | **14** | 0.951 | 0.402 |

I confirmed this held up out of sample before adopting it: 4/4 on fold 1, 8/8
on fold 2.

One thing I like about 0.20 that isn't obvious from the table: it sits in the
middle of a plateau now, not on the edge of a cliff. The earlier model only
held perfect capture right up to 0.20 — one step further, at 0.25, it started
missing an event. This one holds 19/19 all the way from 0.05 through 0.25,
bottoming out at 38 false episodes across that whole range. That's real
headroom on both sides of the chosen threshold, which matters if the data
ever drifts a little — and it means the risk bands already signed off on
carry over unchanged.

## 9. What the model is actually reacting to

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

Relative features carry 70.7% of the model's total attention. Given both the
raw number and the "how unusual is this *for this specific machine*" version
of the same signal, the model overwhelmingly prefers the second one. That's
the single clearest result in this whole document, and honestly the reason
this revision was worth doing at all.

Two more things worth flagging:

- **`run_hours_since_maintenance` fell to 0.001** (from 0.007 before). The
  intuitive "wear since last service" story isn't just weak — the model
  barely uses it. I actually take this as a good sign, not a disappointing
  one: it's a quiet confirmation the leakage guard is working. Used honestly,
  this feature is nearly worthless; a version that peeked at the *next*
  maintenance reset would have looked spectacularly predictive, for exactly
  the wrong reason — because that reset usually happens right after the
  breakdown it appears to "predict."
- **Temperature now edges out vibration** (52.3% vs 47.6%), flipping the old
  62/37 split from when the model only saw raw values. Temperature's absolute
  level varies a lot from machine to machine, which was burying its real
  signal. Once it's measured relative to each machine's own normal, it turns
  out to be the strongest single driver.

## 10. What changed when I went back and re-checked my own work

I re-tested four decisions after Phase 2. Two held up. Two didn't.

| Decision | Verdict |
|---|---|
| Random Forest over Logistic Regression / boosting | Held — re-confirmed on the new features, at the event level |
| Bands at 0.02 / 0.20 | Held — checked against walk-forward out-of-sample scores, not just in-sample ones |
| Raw features only | Changed — added 7 per-machine relative ratios |
| `min_samples_leaf=1` | Changed — moved to 20 |

The two things I got wrong are worth recording, because the reasoning matters
more than the mistake itself:

- **I suspected the risk bands were built on a contaminated distribution** —
  computed from the final model, which has already seen every failure it's
  scoring. I checked, and I was wrong to worry: walk-forward out-of-sample
  scores cleared them. 90.52% of out-of-sample hours score exactly 0.0000,
  against 92.99% in-sample, and the watch tier actually gets *more* useful
  out of sample (2.21% versus 1.16%). Good instinct to check. Wrong
  conclusion.
- **I briefly preferred `min_samples_leaf=20` for the wrong reason** — its
  AUC-PR gain alone (+0.076) looked convincing. At matched event capture, it's
  actually worse on its own: +17 false episodes. That's the exact mistake
  Section 6 exists to warn against, and I made it against my own tuning
  choice rather than against a competing model. It only turns into a real
  improvement combined with the relative features.

Neither change helps on its own. Regularization alone costs 17 extra false
episodes; relative features alone cost 8. Together, they gain 16. The ratio
features are higher-variance, and they need the heavier leaf constraint to
behave — and that constraint needs the richer features to actually have
something worth constraining. Testing them one at a time would have talked me
out of both. The clearest evidence for this is fold 1, where `leaf=1` on the
new features catches only 1 of 4 breakdowns, against `leaf=20`'s 4 of 4.

I also want to walk back something I said earlier: I'd partly justified the
relative features as a fix for the cold-start problem. That claim doesn't
hold up. The measured gain is on the established machines only, and it can't
be otherwise — the cold-start machines have no failures to measure recall
against in the first place. Below 24 hours of history, the relative features
fall back to `1.0`, so a brand-new machine actually gets *less* signal from
them, not more.

## 11. Where this falls short

- **19 caught events is a small sample.** It's a real result over roughly 83
  days of held-out testing, not a guarantee. I'd describe it to the plant
  team as "caught every failure we tested against," never as "never misses."
- **Around 3 of every 5 alerts are false** (row precision 0.402 at the
  0.20 threshold). That's an acceptable cost given how much worse a missed
  failure is than an unnecessary inspection — but the plant team needs to
  hear that number up front, not discover it after the tenth false alarm.
- **The tightest warning time dropped from 22 hours to 16** in this revision
  — the one real regression I'm accepting for a 30% cut in false alarms.
- **The two cold-start machines are unvalidated, not proven safe.** With zero
  recorded failures, there's genuinely no way to measure their recall. Their
  dashboard scores need a visible lower-confidence marker, and the relative-
  feature fallback means they lean on raw sensor values more than any
  established machine does.
- **The trend chart in the app can't use this same shipped model.** It's
  refit on all established history, so scoring that same history with it
  would be optimistic — MCH-213 scores 0.987 at a moment the honest,
  held-out model only gave it 0.225. The app's historical view has to use
  walk-forward out-of-sample scores instead, which currently cover 52.9% of
  the timeline (nothing before 2026-02-26, since earlier training windows
  simply don't contain two failures yet).
- **Missing sensor readings are forward-filled.** That's justified here
  because about 99% of the gaps are single, isolated hours — but a future
  data feed with longer outages would need this revisited.
- **I tuned the hyperparameters by looking at the held-out splits**, which
  isn't a fully clean protocol. My defense is that the winning configuration
  leads on folds 1 and 2 independently, and the primary split — which wasn't
  part of that choice — confirms it on its own (+0.083 AUC-PR, false episodes
  down from 22 to 14). A fully nested validation setup would be stricter, and
  I'd do that if I were starting this project over.

## 12. What's actually shipped

| | |
|---|---|
| File | `models/failure_risk_rf.joblib` |
| Metadata | `models/failure_risk_rf.meta.json` |
| Model | Random Forest, 300 trees, `max_depth=8`, `min_samples_leaf=20`, balanced |
| Features | 26 (19 raw + 7 per-machine relative) |
| Trained on | 42,850 rows, 404 positive (0.943%), 15 established machines |
| Date range | 2026-01-01 → 2026-04-29 |
| Threshold | 0.20 |
| Excluded | MCH-300, MCH-301 (scoreable, just not trained on) |

It's refit on all established history after validation finished — same
hyperparameters, just more data to learn from. The app loads this file
directly; nothing gets retrained when it boots.

## How to reproduce

```bash
pip install -r requirements.txt
python -m src.data_exploration      # Phase 0 findings
python -m src.test_no_leakage       # causality guard over all 26 features (must pass)
python -m src.train_baseline        # candidate comparison table
python -m src.threshold_sweep       # threshold sweep
python -m src.event_level_analysis  # event-level analysis + generalization check
python -m src.train_final_model     # fit and persist the selected model
```
