# Predictive Maintenance — Plant Failure Risk

This is a working system for a manufacturing plant: it looks at hourly sensor
readings from a fleet of machines and predicts which ones are likely to fail
in the next 24 hours. When a machine looks risky, you can ask it *why* and
get a real, plain-English answer grounded in that machine's own numbers —
not a canned message with a value swapped in. You can also just type a
question ("which machines need attention?", "what happened on Line C last
week?") and get an answer pulled from the actual data.

If you're technical, everything below is reproducible — every command
listed actually runs against this repo, and I've tried to say exactly which
script produced which number. If you're not, skip straight to whichever
section title makes sense to you; I've tried to keep the plain-English
explanation up front in each one before the technical detail underneath it.

## Problem understanding

Here's how I read the brief. The plant has 15 machines with about four
months of hourly history each, plus 2 machines that only came online three
days before the data ends. Somewhere in that data, 17 real failures
happened — out of 43,354 machine-hours total. That's a failure roughly
once every 2,500 hours, and it's the thing that makes this project harder
than a normal prediction problem.

Here's why that matters in practice: if you built a model that simply
never predicts a failure, it would be right 99.96% of the time and would
be completely useless — no such model would ever warn anyone about
anything. So the real task isn't "get a high accuracy score," it's "catch
real breakdowns early enough to act on them, without crying wolf so often
that people stop trusting the alerts." Those two goals pull against each
other, and most of the modeling decisions in this project come down to how
that tension got resolved.

There was also a trap sitting in the data that took a bit of digging to
find. One of the columns, `run_hours_since_maintenance`, resets to zero
whenever a machine gets serviced. My first instinct was that this should be
a strong predictor — a machine running a long time since its last service
"should" be more likely to fail. But when I actually checked, 77% of those
resets happen at the same time as, or immediately after, the failure they'd
seem to predict. In other words, maintenance here is mostly *reactive*:
something breaks, then it gets serviced. If I'd built a feature off this
column carelessly — say, "hours until the next reset" — the model would
have looked shockingly accurate for entirely the wrong reason: it would
just be reading the outcome I was trying to predict. I flagged this in
Phase 0 of the project and built the features around it deliberately (more
on that below).

## Approach

### Defining "risk" so it actually means something

"At risk" is a vague phrase until you pin down a time window, so:

> **Risk = the probability that a machine has a failure event in the next 24 hours.**

Concretely: for every hour in the data, I look forward 24 hours and label
that hour "1" if a failure happens somewhere in that window, "0" if not.
The very last 24 hours of each machine's history get dropped rather than
labeled "0," because I genuinely don't know what happens after the data
ends — labeling an unknown as "safe" would have been lying to the model.

### Choosing a metric that can't be fooled by the imbalance

Given how rare failures are, accuracy is a trap (see above), and the
usual next-best option — ROC-AUC — is also flattered here because it gets
credit for correctly ignoring a huge pile of obviously-safe hours. I used
**AUC-PR** (area under the precision-recall curve) instead, which only
cares about how good you are at the part that's actually hard: correctly
picking out the rare positive cases from the huge pool of negatives. A
model that never predicts failure scores essentially 0 on this metric,
which is a much more honest floor to beat than 99.96%.

That said, AUC-PR still only tells you how good the model's *ranking* is —
it doesn't tell you how many real breakdowns you'd have caught, or how many
false alarms you'd have to deal with. For the actual model choice, I
switched to counting real events: did a real breakdown get a warning
before it happened, and how many separate false-alarm episodes did that
cost? That's the number that should drive a decision like this, and it's
what I used.

### Picking the model

I compared six candidates — logistic regression, a shallow decision tree,
random forest, two gradient boosting variants, and a simple averaged
ensemble — all evaluated the same way: time-ordered splits (never randomly
shuffled, since sensor readings an hour apart are obviously related to each
other), with a 24-hour buffer around each split so that no training example
could "see into" the test period through its own label. The full
comparison, including two rounds of me second-guessing my own first choice,
lives in [docs/model_selection.md](docs/model_selection.md); the short
version is:

**A random forest** (300 trees, fairly shallow, tuned to avoid overfitting
a handful of rare events) won because it did better on *both* things that
matter — across three separate held-out test periods, it caught **19 of 19**
real breakdowns while raising 38 total false-alarm episodes, compared to
logistic regression's 18 of 19 catches and 39 episodes. When one model beats
another on both "fewer missed failures" and "fewer false alarms" at once,
you don't need to argue about how expensive a missed failure is relative to
an unnecessary inspection — the answer comes out the same either way.
Boosting didn't help here (small, noisy datasets are its weak spot), and a
simple average of multiple models didn't beat the random forest on its own,
so I didn't build a fancier ensemble just to say I had one.

One thing worth calling out: if you only look at row-by-row metrics (which
count individual *hours*, not breakdowns), logistic regression actually
looks better at first. The reason is a little subtle — 164 "positive" test
rows sounds like a lot, but they're really only 7 actual breakdowns, each
one stretched across a 24-hour window. A model can score well on
hour-counting while still missing an entire real event, which is exactly
what happened here. I only trusted the event-based count for the final
decision, and I'd recommend anyone reviewing this project do the same —
it's the easiest way to accidentally pick the wrong model from a metrics
table that looks convincing at a glance.

### Where the alert line sits

The dashboard sorts machines into three bands:

| Band | Cutoff | Share of machine-hours |
|---|---|---|
| **High** | 0.20 or above | about 2.8% |
| **Watch** | 0.02 to 0.20 | 1–2% |
| **Low** | below 0.02 | roughly 96% |

0.20 isn't a round number I picked for convenience — it's the highest
threshold that still caught every recoverable breakdown across all three
test periods, checked against **out-of-sample** scores (explained a bit
further down) so it isn't a threshold that only looks good because the
model had already seen the answer. The model's risk scores are heavily
skewed toward zero — over 90% of all machine-hours score exactly 0.0000 —
which is why the "watch" band ends up thin no matter where you draw it.
That's a real property of this data, not something I smoothed over.

### What the model is actually paying attention to

| What it's looking at | How much it matters |
|---|---|
| 24h average temperature, relative to that machine's own normal | highest |
| 24h vibration volatility, relative to that machine's own normal | 2nd |
| 6h average temperature, relative to that machine's own normal | 3rd |
| 24h vibration volatility (raw value) | 4th |
| 24h temperature volatility (raw value) | 5th |
| Hours since last maintenance | almost nothing (0.1% of total weight) |

The single biggest design decision in the whole feature set: instead of
only giving the model raw sensor values, I also gave it each value
*relative to that specific machine's own recent history* — is this reading
high for THIS machine, not high in some generic sense. That distinction
turns out to matter a lot. A vibration reading that's totally normal for
one machine might be a red flag on another that usually runs much
smoother, and a model that only sees raw numbers can't tell the difference.
Once I added these relative features and re-tuned around them, false
alarms dropped by roughly 30% for the same number of caught breakdowns —
which is a meaningful improvement, and it took two wrong turns to get
there (also documented in the model-selection writeup, since I think how I
got it wrong twice is more useful than pretending I got it right the first
time).

Also worth noting: "hours since last maintenance," when used honestly (no
peeking at future resets), turns out to barely matter to the model. Given
what I found about maintenance being reactive rather than preventive, I
actually take this as a good sign — it means the leakage trap didn't sneak
back in through the side door. If a forward-looking version of that column
had somehow ended up in the feature set, it would have looked *extremely*
predictive, for entirely the wrong reason.

### The two brand-new machines

MCH-300 and MCH-301 have only 72 hours of history each, with zero recorded
failures between them. I made a deliberate call to leave them out of both
training and evaluation — three days isn't enough to build a meaningful
24-hour-ahead label, and with no failures on record there's nothing to
check a "did it catch this" score against anyway.

They still get scored, though. I made sure `machine_id` itself is never a
feature the model can use, specifically so a model trained on the other 15
machines still generalizes to a brand-new one it's never seen. Their
"relative to own history" features fall back to a neutral "at its own
normal" value until they've built up 24 hours of their own baseline, so
early on they lean more on the raw sensor readings than an established
machine would. Anywhere these two machines show up in the app — the fleet
table, the drill-down, the AI explanations — they're marked as new, and the
language around them is deliberately more hedged. These are real
predictions, but there's genuinely no way to check them against a real
outcome yet, and I didn't want the interface implying more confidence than
that.

### Keeping the AI explanations honest

This was the part of the brief I was most careful about, because it's the
easiest part of a project like this to fake. It would be trivial to write a
template like `"{machine} is at {risk}% risk because vibration is high"`
and call it an explanation — but that's not an explanation, it's a
find-and-replace. If swapping in different sensor numbers wouldn't change
the *reasoning*, only the digits, it doesn't count.

So the way this actually works: for every explanation, the app builds a
real snapshot of that machine's current numbers — not just the raw
readings, but each one expressed as a ratio to that machine's own typical
level (the same ratio the model itself uses to make its prediction) — and
sends *only* that to the language model, live, at request time. The prompt
tells it to name whichever reading is most unusual, using that ratio, to
say plainly when nothing looks wrong instead of manufacturing concern to
sound thorough, and to never just repeat a database field name back at the
reader.

To prove this isn't a template in disguise, the explanation function
was evaluated with two different synthetic sensor readings for the same
machine to check whether the *wording* changed — not just the numbers
embedded in it. Across repeated runs, two genuinely different readings score
noticeably lower on a wording-similarity check (roughly 0.15–0.42) than the
same reading asked twice (roughly 0.43–0.63) — and every single run, the
"something's wrong" language (inspect, stop, immediate) only shows up for
the anomalous reading, and reassuring language shows up for the normal one.

### How the free-text question box works

This one is deliberately split into two separate steps, because the
tempting shortcut — just hand the whole question straight to the language
model and let it improvise — is exactly how you get a system that
confidently answers from general knowledge instead of the actual data.

1. **Look up the real data first.** The question gets parsed with plain,
   boring pattern-matching (not another AI call) to figure out what's being
   asked: which machine, which production line, a risk level ("high risk,"
   "on the watchlist," "safe"), a date (either something relative like
   "last 3 days" or a specific one like "April 8th"), or a top-N request.
   Whatever matches gets used to pull the actual rows out of the scored
   dataset. Nothing gets invented at this stage — it either finds real
   rows or it doesn't.
2. **Then, and only then, answer from what was found.** The language model
   gets exactly those retrieved rows and is told, explicitly, to answer
   only from that evidence, to say plainly when something isn't in it
   rather than guess, and to summarize when the answer is the same for
   almost everyone rather than reciting all 17 machine IDs one by one.

A couple of examples from actually testing this: asked "which machine has
the highest oil pressure" — a sensor that doesn't exist in this dataset —
it says so instead of inventing a plausible-sounding number. Asked "which
machines are high risk right now" when none currently are, it says that in
one sentence instead of dumping the entire fleet table and letting the
reader figure it out. Every answer in the UI comes with a "show the records
this used" toggle, so any answer can be checked against exactly what was
retrieved.

One honest note on how this evolved: an earlier version of this code
computed whether a question was "about risk" but never actually used that
to filter anything — it just handed the model the whole fleet and hoped it
would sort it out correctly on its own. That's a real gap I caught on
review, not a hypothetical one, and it's fixed now: risk-level and specific
dates are proper filters applied before anything reaches the model.

### Why the trend chart doesn't use the same score as everything else

This is a subtle point but an important one, so it's worth spelling out.

The model that powers the dashboard right now is trained on *all* the
history available. That's exactly what you want for scoring the present
moment — use everything you know. But if you used that same
all-the-history model to draw a chart of risk *over the past few months*,
you'd be letting it grade its own homework: at any point in the past, it
would already "know" what happened next, because that data was part of its
training set. The chart would look better than the system actually is.

So the trend chart, and the little sparkline you see in the fleet table,
come from a completely separate process: a rolling weekly re-training,
where each week is scored only by a model trained on data from *before*
that week (plus a 24-hour buffer for the same reason as the labels
above). It's slower to compute and it's not the model that's actually
deployed, but it's honest — it shows what the system would genuinely have
told you at each point in time, not what it can tell you now with the
benefit of hindsight.

The difference isn't small. At one specific real moment, the always-know-
everything model reports a 98.7% risk for machine MCH-213; the honest,
knew-only-what-it-knew-then score for that exact same hour is 42.4%.
Both numbers are correct — they're just answering different questions
("what do we know now, looking back" versus "what would we genuinely have
been told at the time"). The chart on the machine drill-down page always
shows the second one, and the fleet activity summary described below is
computed from it too, since a plant manager reviewing *history* should see
an honest history, not a flattering one.

One consequence of doing it this way: there isn't enough history at the
very start of the dataset to train a meaningful model yet (you need at
least a couple of real failures in the training window before a model is
worth anything), so the trend chart's earliest stretch — a bit under half
the total timeline — is left visibly blank rather than filled in with a
guess. It's an honest gap, not a bug.

Because both of these are legitimate but different evaluations, two
different "how many did it catch" numbers show up in this project and
they're *supposed* to differ: the model-selection comparison above (19 of
19, using three specific held-out test windows chosen to compare
candidates fairly) is a different exercise from the fleet activity summary
you see when you open the app (14 of 14 *evaluable* breakdowns — the other
3 recorded failures happened too early for the weekly walk-forward process
to have built up enough history to score them yet). Neither number is
wrong; they're just measuring slightly different things, and I've tried to
be explicit about which is which wherever they show up.

## The app itself

- **A fleet dashboard** that ranks every machine by current risk and
  color-codes it. I spent real effort making sure a quiet, healthy fleet
  *looks* quiet — a machine that's fine gets a plain outlined label, not a
  loud green badge competing for your attention with the one machine that
  actually needs it.
- **A "here's what actually happened" summary**, always visible the moment
  you open the app. Because the very last stretch of real data happens to
  be quiet (no failures in the final day or so), a dashboard that only ever
  shows "right now" would make it look like the system doesn't do anything.
  So there's a strip up top with real, computed numbers — how many
  breakdowns are on record, how many of the ones the system could actually
  evaluate did it catch, how much warning time that gave, and a one-click
  jump to the moment of the most recent one.
- **A tiny 30-day trend line right in the fleet table** for every machine,
  so even a machine sitting at 0% risk right now still visibly shows you if
  it had a rough patch three weeks ago.
- **Filters** on the fleet table — by production line, by risk band, or
  just show me the new machines — so you're not scrolling a flat list.
- **A drill-down per machine**: sensor history (temperature and vibration
  charted separately, deliberately never crammed onto one dual-axis chart,
  since that's an easy way to make two unrelated trends look connected),
  a table of what's actually driving the current score, a button that
  triggers a real, live "explain this" call to the language model, and the
  honest walk-forward trend chart described above.
- **The question box**, described above.
- **A way to look at any past moment**, not just "now" — including a
  shortcut straight to any of the 17 recorded breakdowns, so you can watch
  the system actually catch one instead of only ever seeing a calm fleet.
- Both a dark and a light theme, matching whichever your system prefers by
  default, with a manual override if you'd rather pick one yourself.

Under the hood it's a **FastAPI** backend — which doesn't contain any of
the modeling logic itself, it just calls into the same Python modules used
for training and serves the results as JSON — and a **React** frontend
(Vite + Recharts). In production, one process serves both the app and the
API from the same address, so there's no separate static site to manage
and no cross-origin request headaches to debug.

## How to run

### On your own machine

```bash
pip install -r requirements.txt
cp .env.example .env        # then paste in a real GROQ_API_KEY

# these three build the model artifacts — they're already committed to the
# repo so you don't strictly have to re-run them, but this is how you'd
# rebuild everything from scratch if the data changed
python -m src.train_final_model
python -m src.build_current_scores
python -m src.build_historical_scores

# start the API
uvicorn backend.main:app --reload --port 8000

# in a second terminal, start the frontend
cd frontend
npm install
npm run dev      # opens on http://localhost:5173, talks to the API on :8000
```

If you'd rather run it the way it actually deploys — one process, one
address, no separate dev servers — build the frontend once and let the API
serve it:

```bash
cd frontend && npm install && npm run build && cd ..
uvicorn backend.main:app --port 8000    # now serves the whole app at :8000
```

### With Docker

```bash
docker build -t predictive-maintenance .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key predictive-maintenance
```

### Deploying it for real

`render.yaml` in this repo describes a single Docker web service. Push
this repository to GitHub, then in Render pick **New → Blueprint** and
point it at the repo — Render reads the file and sets everything up.
`GROQ_API_KEY` isn't in the repo (obviously); Render's dashboard asks for
it at deploy time instead.

**Live URL:** https://predictive-maintenance-ypxa.onrender.com/

### If you want to check my work

Every real finding and number in this README came from actually running
something. Here's what produces each piece, in case you want to see it for
yourself rather than take my word for it:

```bash
python -m src.data_exploration       # exploratory sensor data analysis
python -m src.train_baseline         # candidate model comparison table
python -m src.threshold_sweep        # empirical derivation of the 0.20 alert threshold
python -m src.event_level_analysis   # event-level breakdown detection analysis
python -m src.train_final_model      # fit and export the final model artifact
```

## Limitations & what I'd do next

Being upfront about what this doesn't do, or doesn't do perfectly:

- **17 failures is a small number to learn from.** Catching every
  evaluable breakdown in testing is a genuinely good result, but it's a
  result from a small sample, not a guarantee. I've been careful to phrase
  this as "caught everything we tested against," never "never misses."
- **Roughly 3 out of every 5 alerts turn out to be false alarms** at the
  threshold this is currently set to. That's an accepted trade-off given
  how much worse a missed breakdown is than an unnecessary inspection, but
  it's the kind of thing a plant team should be told up front, not
  discover after the tenth false alarm.
- **In the model-selection testing, the shortest warning time seen was 16
  hours**, down from 22 in an earlier version of the model — a deliberate
  trade for a 30% cut in false alarms overall, not a pure improvement (the
  full before-and-after is in the model-selection doc). This is a different
  number from the "23h minimum" you'll see in the live app's activity
  summary — that one comes from the separate weekly walk-forward process
  described above, evaluated over a different, longer stretch of history.
  Both are real; they're just measuring two different things.
- **The two new machines are unproven, not proven-safe.** There's
  currently no way to check their scores against a real outcome, because
  neither has a recorded failure yet.
- **The trend chart only covers about half of the dataset's timeline**, on
  purpose — the earliest stretch predates enough failures for the
  walk-forward process to have anything to train on yet. A finer-grained
  (say, daily instead of weekly) version of that process would push the
  coverage back a little, but not past the first couple of real failures
  in the data.
- **There are 10 duplicate rows in the raw CSV** (same machine, same
  timestamp, identical readings, all on MCH-200, none involving a
  failure). I left them in rather than quietly dropping them — removing
  them now would shift row counts that are already reported and discussed
  earlier in this project for a change too small to affect any actual
  conclusion. Better to flag it than to fix it silently.
- **AI explanations are cached briefly** (per machine, per hour) so
  clicking around the UI doesn't rack up unnecessary API calls — but a
  cache miss always triggers a real, live call, and nothing is ever
  pre-generated ahead of time.
- **The AI-backed features are rate-limited** to protect against accidental
  refresh loops or abuse burning through the API budget. That limit is
  currently tracked per server process, so a future multi-instance
  deployment would need to move it somewhere shared.
- **Groq's rate limits are real, and I hit them while testing this** — the
  free tier used here caps out at 8,000 tokens a minute, which a handful
  of back-to-back requests can exceed. When that happens, the backend
  never shows the raw provider error to the user; it logs the real cause
  on the server and shows a plain "try again in a moment" message instead.
  Under heavier concurrent use this would need a paid tier or a request
  queue.
- **If the Groq API key is missing or wrong, the app still works** — the
  dashboard, the fleet table, the trend charts all keep functioning
  normally, since none of that depends on the language model. Only the
  "explain this" button and the question box would fail, with the same
  plain error message mentioned above. Worth knowing so a missing key
  doesn't get mistaken for the whole app being broken.
- **Bad input is handled honestly rather than papered over.** An
  unparseable date, for instance, returns a real error saying so, instead
  of a misleading "no data found" message (an earlier version of this code
  did exactly that, and it's fixed now). If a request to refresh the view
  fails, the last good data stays on screen with a plain note explaining
  what didn't update, rather than the screen silently going stale with no
  explanation.
- **Where I'd spend more time**: extending the honest walk-forward
  coverage as more real plant data comes in; revisiting how missing sensor
  readings are filled in if a future data feed has longer gaps than this
  one does (right now that choice is justified by this dataset's gaps
  being almost all single, isolated hours — a different data feed might
  need a different answer); building a small held-out check for the two
  new machines once they've been running long enough to have a failure of
  their own to test against; and, obviously, actually deploying this to a
  live Render URL and putting that link here.

## Repository layout

```
src/                        data prep, feature engineering, training, AI logic
  features.py                  causal feature engineering + labeling
  train_baseline.py            candidate model comparisons
  train_final_model.py         fits and saves the model that ships
  build_historical_scores.py   the honest walk-forward scores behind the trend chart
  build_current_scores.py      pre-computed "right now" scores, so the API boots fast
  risk_context.py              scoring, snapshots, fleet activity summary, Q&A lookups
  llm_explain.py                live Groq calls for the "explain this" feature
  llm_qa.py                     the two-stage question box (look up, then answer)
backend/main.py              FastAPI — no modeling logic of its own, just wiring
frontend/                    the React app (Vite + Recharts)
models/                      saved artifacts: the trained model + both score files
docs/model_selection.md      the full model comparison, with every number shown
Dockerfile, render.yaml      deployment config
```
