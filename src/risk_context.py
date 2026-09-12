"""
Retrieval layer: turns the dataset + saved model into real, numeric context.

No LLM calls happen here. This module exists so that explanations and Q&A are
built on retrieved facts rather than on the model's general knowledge — every
number an explanation cites has to come from a snapshot produced here.

Scoring covers ALL 17 machines, including the two cold-start ones. The model
uses no machine_id feature, so it generalises to unseen machines; their
snapshots carry cold_start=True so downstream text can flag lower confidence.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .config import (
    CURRENT_SCORES_PATH,
    DATA_PATH,
    HISTORICAL_SCORES_PATH,
    METADATA_PATH,
    MODEL_PATH,
    RISK_BANDS,
)
from .features import (
    FEATURE_COLUMNS,
    NEW_MACHINES,
    engineer_features,
    impute_sensors,
    load_data,
)

CSV_PATH = str(DATA_PATH)
OOS_PATH = HISTORICAL_SCORES_PATH

# Drivers worth reporting, ordered by the fitted model's combined importance
# (absolute + relative counterpart). See docs/model_selection.md §9.
#
# Each driver reports its absolute value, that machine's own median, its
# percentile against its own history, AND — where the model uses one — the
# `_rel` ratio the model actually keys on. The explanation layer and the model
# now reason over the same quantity instead of merely adjacent ones.
DRIVER_FEATURES = [
    ("vib_roll_std_24h", "24h vibration volatility", "mm/s"),
    ("temp_roll_mean_24h", "24h mean temperature", "°C"),
    ("temp_roll_std_24h", "24h temperature volatility", "°C"),
    ("temp_roll_mean_6h", "6h mean temperature", "°C"),
    ("vib_roll_mean_24h", "24h mean vibration", "mm/s"),
    ("vib_roll_mean_6h", "6h mean vibration", "mm/s"),
    ("vib_roll_std_6h", "6h vibration volatility", "mm/s"),
    ("run_hours_since_maintenance", "hours since last maintenance", "h"),
]


def band_for(score: float) -> str:
    if score >= RISK_BANDS["high"]:
        return "HIGH"
    if score >= RISK_BANDS["medium"]:
        return "MEDIUM"
    return "LOW"


def build_scoring_frame(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """All machines, imputed and featured, with no label/censoring applied —
    so the most recent hour of every machine is present and scoreable."""
    raw = load_data(csv_path)
    return engineer_features(impute_sensors(raw))


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found — run `python -m src.train_final_model` first."
        )
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return joblib.load(MODEL_PATH), meta


def _scores_artifact_is_fresh(csv_path: str) -> bool:
    """Return whether the persisted score frame matches the source inputs."""
    if Path(csv_path).resolve() != Path(CSV_PATH).resolve():
        return False
    if not CURRENT_SCORES_PATH.exists():
        return False
    source_mtime = Path(csv_path).stat().st_mtime
    newest_model_input = max(MODEL_PATH.stat().st_mtime,
                             METADATA_PATH.stat().st_mtime)
    return CURRENT_SCORES_PATH.stat().st_mtime >= max(source_mtime,
                                                     newest_model_input)


def score_frame(df: pd.DataFrame = None, csv_path: str = CSV_PATH,
                use_cached: bool = True) -> pd.DataFrame:
    """Adds a `risk` column (P(failure within 24h)) and a `risk_band`.

    The application uses a committed, source-checked score artifact so boot
    does not rebuild 43k rows of rolling features on every restart. Training
    and data changes automatically invalidate it; ``use_cached=False`` is
    used by the artifact build script.
    """
    if df is None and use_cached and _scores_artifact_is_fresh(csv_path):
        return pd.read_parquet(CURRENT_SCORES_PATH)
    if df is None:
        df = build_scoring_frame(csv_path)
    model, _ = load_model()
    df = df.copy()
    df["risk"] = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    df["risk_band"] = [band_for(v) for v in df["risk"]]
    df["cold_start"] = df["machine_id"].isin(NEW_MACHINES)
    return df


def _pct_rank(series: pd.Series, value: float) -> float:
    """Percentile of `value` within that machine's own history (0-100)."""
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    return float((clean <= value).mean() * 100)


def machine_snapshot(scored: pd.DataFrame, machine_id: str, at=None) -> dict:
    """Real, retrieved numbers for one machine at one timestamp.

    Everything an explanation is allowed to cite lives in here. Percentiles are
    computed against that machine's OWN history, which is what makes the
    reasoning comparative rather than a fixed template: the same absolute
    vibration reading means different things on different machines.
    """
    g = scored[scored["machine_id"] == machine_id].sort_values("timestamp")
    if g.empty:
        raise ValueError(f"unknown machine_id: {machine_id}")
    row = g.iloc[-1] if at is None else g[g["timestamp"] <= at].iloc[-1]
    hist = g[g["timestamp"] <= row["timestamp"]]

    drivers = []
    for col, label, unit in DRIVER_FEATURES:
        entry = {
            "feature": col,
            "label": label,
            "unit": unit,
            "value": round(float(row[col]), 3),
            "machine_median": round(float(hist[col].median()), 3),
            "percentile_vs_own_history": round(_pct_rank(hist[col], row[col]), 1),
        }
        rel_col = f"{col}_rel"
        if rel_col in row.index:
            # The ratio the model itself uses: 1.0 = at this machine's own
            # normal, 2.0 = twice its usual level.
            entry["times_own_normal"] = round(float(row[rel_col]), 2)
        drivers.append(entry)

    def risk_at(hours_back):
        past = hist[hist["timestamp"] <= row["timestamp"] - pd.Timedelta(hours=hours_back)]
        return round(float(past.iloc[-1]["risk"]), 4) if len(past) else None

    prior_failures = int(hist["failure_event"].sum())
    resets = int((hist["run_hours_since_maintenance"] == 0).sum())

    return {
        "machine_id": machine_id,
        "line": str(row["line"]),
        "as_of": str(row["timestamp"]),
        "cold_start": bool(row["cold_start"]),
        "hours_of_history": int(len(hist)),
        "risk": round(float(row["risk"]), 4),
        "risk_band": row["risk_band"],
        "alert_threshold": RISK_BANDS["high"],
        "risk_24h_ago": risk_at(24),
        "risk_7d_ago": risk_at(24 * 7),
        "current_readings": {
            "temperature_c": round(float(row["temperature_c"]), 2),
            "vibration_mm_s": round(float(row["vibration_mm_s"]), 3),
            "run_hours_since_maintenance": int(row["run_hours_since_maintenance"]),
        },
        "drivers": drivers,
        "prior_failures_on_record": prior_failures,
        "maintenance_resets_on_record": resets,
    }


def fleet_snapshot(scored: pd.DataFrame) -> list:
    """Latest scored hour for every machine, ranked by risk (dashboard view)."""
    latest = scored.sort_values("timestamp").groupby("machine_id").tail(1)
    latest = latest.sort_values("risk", ascending=False)
    return [
        {
            "machine_id": r["machine_id"],
            "line": r["line"],
            "as_of": str(r["timestamp"]),
            "risk": round(float(r["risk"]), 4),
            "risk_band": r["risk_band"],
            "cold_start": bool(r["cold_start"]),
            "vibration_mm_s": round(float(r["vibration_mm_s"]), 3),
            "temperature_c": round(float(r["temperature_c"]), 2),
            "run_hours_since_maintenance": int(r["run_hours_since_maintenance"]),
        }
        for _, r in latest.iterrows()
    ]


def query(scored: pd.DataFrame, machines=None, line=None, band=None,
          start=None, end=None, end_exclusive=None, last_hours=None,
          limit=400) -> pd.DataFrame:
    """Deterministic filtered retrieval — the evidence step for Q&A."""
    out = scored
    if machines:
        out = out[out["machine_id"].isin(machines)]
    if line:
        out = out[out["line"].str.lower() == str(line).lower()]
    if band:
        out = out[out["risk_band"] == band.upper()]
    if last_hours:
        cutoff = out["timestamp"].max() - pd.Timedelta(hours=last_hours)
        out = out[out["timestamp"] >= cutoff]
    if start:
        out = out[out["timestamp"] >= pd.Timestamp(start)]
    if end_exclusive:
        out = out[out["timestamp"] < pd.Timestamp(end_exclusive)]
    if end:
        out = out[out["timestamp"] <= pd.Timestamp(end)]
    return out.sort_values("timestamp").tail(limit)


def fleet_activity_summary(oos: pd.DataFrame, failures: pd.DataFrame,
                           horizon_hours: int = 24) -> dict:
    """Real, computed history for the dashboard's landing view.

    At the dataset's final hour every machine reads LOW (the last failure was
    30 Apr 04:00), so a viewer who only sees the current snapshot sees an
    all-calm fleet and has no way to tell the system does anything. This
    reuses the SAME catch/lead-time method validated in
    src/event_level_analysis.py -- run here, live, against the committed
    walk-forward artifact, not copied as a number from a report -- so the
    landing view can honestly say what the model has actually caught.
    """
    threshold = RISK_BANDS["high"]
    scored_mask = oos["status"].str.startswith("out_of_sample")
    oos_scored = oos[scored_mask]

    events = []
    for _, f in failures.iterrows():
        window = oos_scored[
            (oos_scored["machine_id"] == f["machine_id"])
            & (oos_scored["timestamp"] >= f["timestamp"] - pd.Timedelta(hours=horizon_hours))
            & (oos_scored["timestamp"] < f["timestamp"])
        ]
        if window.empty:
            continue  # outside walk-forward coverage — not countable either way
        alerts = window[window["risk"] >= threshold]
        caught = not alerts.empty
        lead_hours = (
            round((f["timestamp"] - alerts["timestamp"].min()).total_seconds() / 3600, 1)
            if caught else None
        )
        peak_risk = float(window["risk"].max())
        events.append({
            "machine_id": f["machine_id"], "line": f["line"],
            "timestamp": str(f["timestamp"]), "caught": caught,
            "lead_hours": lead_hours,
            "peak_risk": round(peak_risk, 4),
            "peak_band": band_for(peak_risk),
        })

    caught_events = [e for e in events if e["caught"]]
    lead_times = [e["lead_hours"] for e in caught_events]

    most_recent = None
    if len(failures):
        r = failures.sort_values("timestamp").iloc[-1]
        most_recent = {"machine_id": r["machine_id"], "line": r["line"],
                       "timestamp": str(r["timestamp"])}

    return {
        "total_recorded_failures": int(len(failures)),
        "evaluable_in_walk_forward": len(events),
        "caught_in_walk_forward": len(caught_events),
        "median_lead_hours": round(float(np.median(lead_times)), 1) if lead_times else None,
        "min_lead_hours": round(float(np.min(lead_times)), 1) if lead_times else None,
        "most_recent_failure": most_recent,
        "events": events,
    }


def machine_sparkline(oos: pd.DataFrame, machine_id: str, points: int = 30) -> list:
    """Compact daily-max out-of-sample risk for one machine, oldest first.

    Daily MAX (not mean) is used deliberately: a single bad hour is the
    signal that matters for "did anything happen that day," and averaging
    would wash out a short-lived spike into invisibility.
    """
    g = oos[(oos["machine_id"] == machine_id)
            & oos["status"].str.startswith("out_of_sample")].copy()
    if g.empty:
        return []
    g["day"] = g["timestamp"].dt.floor("D")
    daily = g.groupby("day")["risk"].max().reset_index()
    daily = daily.tail(points)
    return [
        {"day": str(r["day"].date()), "risk": round(float(r["risk"]), 4)}
        for _, r in daily.iterrows()
    ]


def band_report(scored: pd.DataFrame) -> dict:
    """What the provisional bands actually cover — the numbers needed to sign
    off (or change) the HIGH/MEDIUM cuts in Phase 4."""
    latest = scored.sort_values("timestamp").groupby("machine_id").tail(1)
    counts = latest["risk_band"].value_counts().to_dict()
    return {
        "bands": RISK_BANDS,
        "machines_per_band_now": {b: int(counts.get(b, 0)) for b in ["HIGH", "MEDIUM", "LOW"]},
        "all_hours_pct_per_band": {
            b: round(float((scored["risk_band"] == b).mean() * 100), 2)
            for b in ["HIGH", "MEDIUM", "LOW"]
        },
    }


if __name__ == "__main__":
    scored = score_frame()
    print(f"Scored {len(scored):,} machine-hours across "
          f"{scored['machine_id'].nunique()} machines\n")
    print("Fleet, ranked by current risk:")
    for m in fleet_snapshot(scored):
        flag = "  [COLD START]" if m["cold_start"] else ""
        print(f"  {m['machine_id']}  {m['line']:<7} risk={m['risk']:.4f}  "
              f"{m['risk_band']:<6} vib={m['vibration_mm_s']:.3f} "
              f"temp={m['temperature_c']:.1f}{flag}")
    print("\nBand coverage:")
    print(json.dumps(band_report(scored), indent=2))
