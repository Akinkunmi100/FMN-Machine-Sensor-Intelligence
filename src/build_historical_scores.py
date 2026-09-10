"""
Phase 4 — walk-forward out-of-sample risk scores for the trend chart.

Why this exists: the shipped artifact is refit on ALL established history, so
re-scoring that same history with it grades its own homework. MCH-213 scores
0.987 at a moment the honest held-out model gave 0.225. A trend chart built on
the final model would show the system catching things it could not have caught
in real time.

Method — rolling-origin weekly refit. For each calendar week, train on every
labelled row strictly before that week (minus a 24h embargo, since the label at
time t reads up to t+24h) and score the week with that model. Adjacent windows
share almost all their training data, so scores stay comparable across the seam
in a way that two monthly CV-fold models would not.

Coverage limit, stated rather than hidden: a week is only scored once its
training window holds at least 2 real failure events. The first failure is
2026-02-04, so nothing before 2026-02-26 is scoreable — roughly half the
history. Those rows are emitted with status="insufficient_history" and no
score, so the app can render the gap honestly instead of implying a flat line.

All 17 machines are scored, including the two cold-start ones: the model uses
no machine_id feature, so a model trained on established machines generalises
to them. Their rows land in the final weeks and do get genuine out-of-sample
scores — but with under 24h of prior history their relative features sit at the
1.0 fallback, so those scores lean on absolute features. Flagged, not fixed.

Run:  python src/build_historical_scores.py
Out:  models/historical_scores_oos.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from features import (
    FEATURE_COLUMNS,
    NEW_MACHINES,
    build_dataset,
    engineer_features,
    impute_sensors,
    load_data,
)
from train_final_model import ALERT_THRESHOLD, HORIZON_HOURS, MODEL_PARAMS

CSV_PATH = "project2_manufacturing_sensors.csv"
OUT_PATH = Path("models/historical_scores_oos.parquet")

MIN_TRAIN_EVENTS = 2   # a model trained on 0-1 failures has nothing to learn
WEEK = pd.Timedelta(days=7)


def build_frames():
    """(labelled established rows for training, all-machine rows for scoring)."""
    train_pool, _ = build_dataset(CSV_PATH, horizon_hours=HORIZON_HOURS)
    scoring = engineer_features(impute_sensors(load_data(CSV_PATH)))
    return train_pool, scoring


def main() -> int:
    train_pool, scoring = build_frames()

    start = scoring["timestamp"].min().normalize()
    end = scoring["timestamp"].max()
    weeks = pd.date_range(start, end, freq="7D")

    parts, log = [], []
    for week_start in weeks:
        week_end = week_start + WEEK
        block = scoring[(scoring["timestamp"] >= week_start)
                        & (scoring["timestamp"] < week_end)]
        if block.empty:
            continue

        # strictly-prior training rows, minus the embargo
        cutoff = week_start - pd.Timedelta(hours=HORIZON_HOURS)
        train = train_pool[train_pool["timestamp"] < cutoff]
        n_events = int(train["failure_event"].sum()) if len(train) else 0

        if n_events < MIN_TRAIN_EVENTS or train["label"].sum() == 0:
            part = block[["machine_id", "timestamp", "line"]].copy()
            part["risk"] = np.nan
            part["status"] = "insufficient_history"
            part["train_events"] = n_events
            parts.append(part)
            log.append((week_start.date(), len(block), n_events, "skipped"))
            continue

        model = RandomForestClassifier(**MODEL_PARAMS)
        model.fit(train[FEATURE_COLUMNS], train["label"])

        part = block[["machine_id", "timestamp", "line"]].copy()
        part["risk"] = model.predict_proba(block[FEATURE_COLUMNS])[:, 1]
        part["status"] = "out_of_sample"
        part["train_events"] = n_events
        parts.append(part)
        log.append((week_start.date(), len(block), n_events, "scored"))

    out = pd.concat(parts).sort_values(["machine_id", "timestamp"])
    out["cold_start"] = out["machine_id"].isin(NEW_MACHINES)
    # A cold-start machine's relative features sit at the 1.0 fallback for its
    # first 24h, so mark those rows rather than letting them look equivalent.
    out.loc[out["cold_start"] & (out["status"] == "out_of_sample"),
            "status"] = "out_of_sample_cold_start"
    out["alert"] = out["risk"] >= ALERT_THRESHOLD

    OUT_PATH.parent.mkdir(exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    scored = out["status"].str.startswith("out_of_sample")
    print(f"Saved -> {OUT_PATH}")
    print(f"\n{len(out):,} rows total, {int(scored.sum()):,} with an "
          f"out-of-sample score ({100 * scored.mean():.1f}%)")
    print(f"Scored range: {out.loc[scored, 'timestamp'].min()} -> "
          f"{out.loc[scored, 'timestamp'].max()}")
    print("\nWeekly log:")
    for wk, n, ev, status in log:
        print(f"   {wk}  {n:5,} rows  train_events={ev:<3} {status}")

    print("\nStatus breakdown:")
    for status, n in out["status"].value_counts().items():
        print(f"   {status:<28} {n:6,}")

    print("\nHow often the honest score would have alerted, by machine:")
    per = (out[scored].groupby("machine_id")
           .agg(hours=("risk", "size"), alert_hours=("alert", "sum"),
                peak=("risk", "max")))
    per["alert_pct"] = (100 * per["alert_hours"] / per["hours"]).round(2)
    print(per.sort_values("peak", ascending=False).to_string())
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
