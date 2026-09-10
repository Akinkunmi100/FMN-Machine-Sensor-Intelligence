# Predictive Maintenance — Plant Failure Risk

An end-to-end system for a manufacturing plant: predicts which machines are at
risk of failure in the next 24 hours, explains why using a live LLM call
grounded in that machine's real sensor data, shows how risk has moved over
time, and answers free-text questions from a plant manager.

Every number in this README is reproducible from this repo — see
[How to run](#how-to-run).

---

## Problem understanding

The plant has 15 machines with 120 days of hourly sensor history and 2 new
machines with only ~3 days each. Failures are rare — **17 events in 43,354
machine-hours (0.039%)** — so this is a severe class-imbalance problem, not a
routine classification task. A model that never predicts failure scores
99.96% accuracy and is worthless; the real questions are *how much warning
does a real alert give*, and *how many false alarms does that warning cost*.

The dataset also contains a trap: `run_hours_since_maintenance` resets to 0
after maintenance, and **77.3% of those resets coincide with or immediately
follow the failure they would appear to predict** (Phase 0 finding). Resets
are reactive, not preventive. Any feature built carelessly from this column
would leak the answer.

## Approach

### Risk definition and horizon

> **Risk = P(a failure event occurs on this machine within the next 24 hours)**

For every machine-hour `t`, the label is `1` if `failure_event = 1` occurs in
`(t, t+24h]`. The last 24 hours of each machine's history are dropped rather
than labelled `0` — the true outcome there is unknown, and calling an unknown
a negative would poison training.

### Metric

**AUC-PR**, not accuracy or ROC-AUC. At a 0.039% positive rate, ROC-AUC is
flattered by a huge true-negative pool; AUC-PR ignores true negatives
entirely and answers the question that matters — of the hours flagged, how
many are real. The naive floor (always predict no-failure) is the positive
rate itself, giving an honest baseline instead of a meaningless 99.96%.

AUC-PR still only measures ranking, not warnings. The real model choice was
made on **event-level** metrics instead — see below.

### Model selection

Six candidates were compared on identical time-ordered splits (a primary
80/20 split plus two expanding-window CV folds, each protected by a 24-hour
embargo so no training row's label depends on test-side data). Full
comparison, event-level reasoning, and two rounds of re-examination are in
[docs/model_selection.md](docs/model_selection.md) — the short version:

**Random Forest** (300 trees, `max_depth=8`, `min_samples_leaf=20`, balanced
class weights) was chosen because it **dominates on both axes at once**:
across all three held-out windows it catches **19 of 19 real breakdowns**
with **38 total false-alarm episodes**, against Logistic Regression's 18/19
with 39. When one candidate is better on both missed failures *and* false
alarms, no cost ratio between "missed failure" and "unnecessary inspection"
needs to be assumed — every ratio gives the same answer. Boosting variants
(XGBoost, HistGradientBoosting) were tested and rejected; a training-free
probability-averaged ensemble was tested and didn't beat the single best
model on 2 of 3 splits, so no ensemble or stacked meta-learner was built.

The row-level metrics (which count *hours*, not breakdowns) initially
favoured Logistic Regression — 164 "positive" test rows are only 7 real
events, each inflated into a 24-hour window, so hour-level recall rewarded a
model that later turned out to miss an actual breakdown. This mismatch is
documented in detail in the model-selection writeup because it's the single
easiest way to pick the wrong model from a metrics table that looks
convincing.

### Threshold and risk bands

The dashboard uses three bands, chosen from the model's actual score
distribution rather than round numbers:

| Band | Cutoff | Share of machine-hours |
|---|---|---|
| **HIGH** | ≥ 0.20 | 2.8% |
| **MEDIUM** (watch) | ≥ 0.02 | 1.1–2.2% |
| **LOW** | < 0.02 | ~96% |

0.20 is the validated alert threshold — the highest cutoff that still catches
every recoverable breakdown in every held-out window, confirmed against
**walk-forward out-of-sample scores**, not the in-sample distribution (see
below), so it isn't inflated by the model grading its own homework. The score
distribution is strongly bimodal (over 90% of hours score exactly 0.0000),
which is why the watch band is thin no matter where it's cut — that's
reported as a property of the data, not hidden.

### What actually predicts failure

| Feature | Importance |
|---|---|
| `temp_roll_mean_24h_rel` (24h mean temp, relative to this machine's own median) | 0.188 |
| `vib_roll_std_24h_rel` (24h vibration volatility, relative) | 0.157 |
| `temp_roll_mean_6h_rel` | 0.154 |
| `vib_roll_std_24h` (absolute) | 0.115 |
| `temp_roll_std_24h` (absolute) | 0.102 |
| `run_hours_since_maintenance` | 0.001 |

**Per-machine relative features carry 70.7% of total importance.** Every
rolling statistic has a counterpart divided by that machine's own running
median (`shift(1).expanding(min_periods=24).median()` — strictly backward
in time, verified by `src/test_no_leakage.py`). A reading that's high in
absolute terms but normal *for that specific machine* isn't a warning sign;
one that's only modestly elevated but far above *that machine's own* history
is. Offered both views, the model overwhelmingly prefers the relative one —
adding these features and retuning around them cut false alarms by 30% at
identical event capture (see the model-selection doc, §10, for the full
before/after and the two wrong turns taken to get there).

`run_hours_since_maintenance` used causally (no forward-looking version) is
nearly irrelevant to the model. This is a quiet confirmation the leakage
guard works: the intuitive "wear since service" signal is genuinely weak,
while a version that leaked the reactive reset would have looked
spectacularly — and falsely — predictive.

### Cold-start strategy

MCH-300 and MCH-301 have 72 hours of history each and zero recorded
failures. They are **excluded from training and evaluation** — three days
can't support a 24-hour horizon label, and zero positives means nothing to
score recall against.

They remain **scoreable**: `machine_id` is deliberately never a feature, so
the model reads only physically-grounded sensor signal and generalizes to
machines it has never trained on. Their relative features fall back to
`1.0` ("at its own normal") until 24 hours of history accumulate, so early
on they lean more heavily on absolute features than an established machine
would. Their dashboard scores carry a visible "new" flag and lower-confidence
language everywhere they appear (fleet table, drill-down, LLM explanations)
— these are real predictions, but literally unvalidated, since there is no
recorded failure to check them against.

### How the LLM stays grounded

`src/llm_explain.py` sends the model **only** a JSON snapshot built by
`src/risk_context.py` — current readings, each driver's percentile against
that machine's own history, and the `times_own_normal` ratio the risk model
itself keys on. The prompt requires the explanation to name the standout
driver using that ratio, to say plainly when nothing is anomalous rather
than manufacture concern, and forbids raw field names or bare-decimal risk
("0.94" reads as a probability; "94% chance" reads as a warning).

**Verification, not assertion**: `src/test_llm_grounded.py` calls the same
machine with two different synthetic sensor readings and checks the output
*wording* differs — not just the embedded numbers — by masking every
numeric token before comparing. A same-input control call proves the
difference is driven by the sensor picture, not sampling noise: across
repeated runs, two different readings consistently score **lower** wording
similarity (0.15–0.42 observed) than the same reading asked twice (0.43–0.63
observed) — the test asserts that ordering, not a fixed number, since the
LLM call is not deterministic. Escalation language ("inspect", "stop",
"immediate") appears only in the erratic case; reassurance language only in
the steady one, every run.

### How Q&A retrieval works

`src/llm_qa.py` is deliberately two stages:

1. **Retrieve** — the question is parsed with plain regex into filters —
   machine ID, line, **risk band** ("high risk," "watchlist," "safe"),
   **date** (both relative, "last 3 days," and absolute, "April 8th" or
   "2026-04-08"), and "top N" — then real rows are pulled from the scored
   dataset via `risk_context.query()`. No LLM call happens in this stage, so
   it cannot hallucinate a filter. (An earlier version computed whether a
   question was risk-related but never actually filtered by band before
   handing evidence to the model — a real gap against the "filtered by
   machine, date range, or risk level" requirement, caught in review and
   fixed: band and absolute-date filters are now applied at retrieval, not
   left for the LLM to sift out of the full fleet itself.)
2. **Answer** — the retrieved rows (and only the retrieved rows) are handed
   to the LLM with instructions to answer strictly from that evidence, say
   plainly when something isn't in it, summarize a uniform result rather than
   enumerating every machine, and never use markdown (the UI displays plain
   text, so `**bold**` would show as literal asterisks).

Asked *"Which machine has the highest oil pressure?"* — a column that
doesn't exist in this dataset — the system responds that no oil-pressure
reading is available rather than inventing one. Asked *"which machines are
high risk right now?"* when none are, it says so in one sentence instead of
dumping the full fleet list. Counts (e.g. "all 17 machines") are computed by
code and handed to the model directly rather than left for it to count a
list itself — an LLM miscounting a 17-item array is a real failure mode that
was observed and fixed during review, not a hypothetical one. Every answer
ships with the retrieved evidence attached, viewable in the UI ("show the
records this used"), so a wrong answer can always be traced back to what was
retrieved.

### The trend chart is walk-forward out-of-sample — deliberately not the shipped model

The shipped model is refit on **all** established history, which is correct
for scoring the *present* but would make the *historical* trend chart grade
its own homework. `src/build_historical_scores.py` instead does a
rolling-origin weekly refit: each week is scored only by a model trained on
data strictly before it (minus the 24h embargo). Concretely, at one real
moment the shipped model reports MCH-213 at **98.7%** risk; the honest
walk-forward score for that same hour is **42.4%** — both true, answering
different questions ("what do we know now, with hindsight" vs. "what would
we have known then"). The trend chart always shows the latter.

Coverage is 53.5% of history — nothing is scored before 2026-02-26, because
earlier weeks' training windows contain fewer than two recorded failures and
there was no model to speak of yet. Unscored hours are returned as an
explicit gap (`status: "insufficient_history"`), never a flat line implying
the model was quiet.

## App

- **Dashboard** — every machine ranked by current risk, colour-coded by
  band. Visual weight follows severity: LOW rows carry no fill and an
  outlined pill; HIGH rows get a red stripe, a tinted background, and a
  filled pill — so the seventeen quiet machines don't compete for attention
  with the one that matters.
- **Drill-down** — sensor history (temperature and vibration on separate
  charts, never a dual axis), the driver table with each reading's ratio to
  that machine's own normal, a live "explain this risk" button, and the
  walk-forward trend chart.
- **Ask box** — free-text question, grounded answer, retrieved evidence
  visible on request.
- **Time control** — since the dataset's final hour has every machine at
  LOW (the last recorded failure was 30 April 04:00), the dashboard includes
  a jump-to-any-recorded-breakdown picker so the model can be seen doing its
  actual job rather than always landing on an all-green fleet.

Stack: **FastAPI** (backend, holds no modelling logic of its own — it
imports `src/` unchanged) + **React/Vite/Recharts** (frontend). In
production both are served from one origin by the same FastAPI process
(`frontend/dist` mounted as static files), so there's no separate static
site and no CORS in production.

## How to run

### Locally

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in GROQ_API_KEY

# one-time: build the model artifacts (already committed to the repo, but
# reproducible from scratch)
python src/train_final_model.py
python src/build_current_scores.py
python src/build_historical_scores.py

# backend
uvicorn backend.main:app --reload --port 8000

# frontend (separate terminal)
cd frontend
npm install
npm run dev      # http://localhost:5173, proxies /api to :8000
```

Or the production shape (single origin, no separate dev servers):

```bash
cd frontend && npm install && npm run build && cd ..
uvicorn backend.main:app --port 8000    # serves the app AND the API at :8000
```

### Docker / Render

```bash
docker build -t predictive-maintenance .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key predictive-maintenance
```

`render.yaml` deploys this as a single Docker web service — push to GitHub,
then in Render choose **New → Blueprint** and point it at the repo.
`GROQ_API_KEY` is requested by Render's dashboard at deploy time (marked
`sync: false` in the blueprint), never committed to the repo.

**Deployed URL**: not yet live — deployment is pending the Render service
being created from this GitHub repository. The Dockerfile and render.yaml are
prepared and their individual build stages verified.

### Reproducing every claim in this README and the model-selection doc

```bash
python src/data_exploration.py       # Phase 0 findings
python src/test_no_leakage.py        # causality guard over all 26 features
python src/test_retrieval.py         # deterministic Q&A filter regression checks
python src/train_baseline.py         # candidate comparison table
python src/threshold_sweep.py        # threshold sweep
python src/event_level_analysis.py   # event-level analysis + generalization check
python src/test_llm_grounded.py      # groundedness verification (needs GROQ_API_KEY)
```

## Limitations & next steps

- **Small sample.** 19 caught events over ~83 days of held-out testing is a
  real result, not a guarantee — describe it as "caught every failure in
  testing," never "never misses."
- **~3 of 4 alerts are false** at the shipped threshold (row-level precision
  0.402). Acceptable given the cost asymmetry, but the plant team should be
  told this upfront so the expectation is set correctly from day one.
- **Minimum warning time is 16 hours**, down from 22h in an earlier
  configuration — a real, accepted trade for a 30% cut in false alarms
  (documented in model_selection.md §10).
- **Cold-start machines are unvalidated, not validated-good.** Their
  event-level recall is literally unmeasurable with zero recorded failures.
- **The trend chart covers 53.5% of history** by design — see above. A
  finer-grained (e.g. daily) walk-forward refit could extend coverage
  slightly but not past the first two failures in Feb 2026.
- **10 duplicate `(machine_id, timestamp)` rows exist in the raw CSV**, all
  on MCH-200, all exact duplicates (same readings, no failures involved).
  They were not removed — doing so now would shift row counts already
  reported throughout Phase 0–2 for a change too small to affect any
  conclusion (MCH-200 gets 10 hours of harmless double weight in training).
  Flagged here rather than silently fixed or silently ignored.
- **Explanation caching** is per `(machine_id, as_of)` to avoid re-billing
  Groq on UI re-renders — a cache miss always makes a live call, and it is
  never precomputed. Bounded to 500 entries (simple FIFO eviction) so a
  long-lived server process can't grow this without limit.
- **LLM endpoints are rate-limited in-process** to protect the Groq budget from
  refresh loops and casual abuse. The limit is per client IP and resets every
  minute; a multi-instance deployment should move this policy to an edge or
  shared store.
- **Groq rate limits are real and were hit during testing** (8,000 tokens/min
  on the on-demand tier used here). Back-to-back explanation or Q&A requests
  can return a 429 from the provider. The backend never surfaces that raw
  error to the UI — it logs the real cause server-side and returns a plain
  "temporarily unavailable, try again shortly" message — but under
  concurrent stakeholder usage this is a genuine capacity ceiling, not just
  an edge case. A paid Groq tier or simple request queuing would remove it.
- **If `GROQ_API_KEY` is missing or invalid, the app still boots and the
  dashboard/trend chart work normally** — only "explain this risk" and the
  ask box fail, with the same plain error message above. This is by design
  (the risk model doesn't need an LLM to function) but is worth knowing
  before assuming a blank explanation means the whole app is broken.
- **A malformed or out-of-range `as_of` value returns a clear 400/404**
  rather than a misleading error (an earlier version reported "no data for
  machine X" when the actual problem was an unparseable timestamp — fixed).
  A view-in-progress error no longer fails silently either: the last
  successfully loaded data stays visible with a plain-language banner
  explaining what didn't update.
- **Next steps**: extend walk-forward coverage as more plant history
  accumulates; revisit the missingness treatment (forward-fill, justified
  today by ~99% of gaps being isolated single hours) if a future feed has
  longer sensor outages. Leading gaps use fixed causal fallbacks rather than
  future observations. Consider a small held-out validation set for the
  two cold-start machines once they've run long enough to have failures of
  their own; deploy and put a real Render URL here.

## Repository structure

```
src/                     data prep, feature engineering, training, LLM logic
  features.py              causal feature engineering + labelling
  train_baseline.py        candidate model comparison (Phase 1)
  train_final_model.py     fits and persists the shipped artifact
  build_historical_scores.py   walk-forward OOS scores for the trend chart
  build_current_scores.py      persisted current-state scores for fast API boot
  risk_context.py          retrieval layer: scoring, snapshots, Q&A queries
  llm_explain.py            live Groq explanation calls
  llm_qa.py                 two-stage grounded Q&A
  test_no_leakage.py        causality guard (must pass)
  test_llm_grounded.py      groundedness verification (must pass)
backend/main.py          FastAPI — imports src/ unchanged, shapes JSON
frontend/                React app (Vite + Recharts)
models/                  committed artifacts: trained model + current/OOS scores
docs/model_selection.md  full model-selection record with every metric shown
Dockerfile, render.yaml  deployment config
```
