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

from features import (
    FEATURE_COLUMNS,
    NEW_MACHINES,
    engineer_features,
    impute_sensors,
    load_data,
)

CSV_PATH = "project2_manufacturing_sensors.csv"
MODEL_PATH = Path("models/failure_risk_rf.joblib")
METADATA_PATH = Path("models/failure_risk_rf.meta.json")

# HIGH is the validated operating point from Phase 2: 0.20 caught 19/19
# breakdowns across every held-out window.
#
# MEDIUM ("watch") is 0.02, signed off after seeing the score distribution.
# The model is bimodal — 92.95% of machine-hours score exactly 0.0000 and the
# 97th percentile is already 0.1367 — so a middle band is thin wherever it is
# cut. 0.02 was chosen over a rounder 0.10 because it is the widest useful
# watch tier (1.14% of hours vs 0.33%), and anything above it already sits in
# the top 5% of all hours for that fleet.
RISK_BANDS = {"high": 0.20, "medium": 0.02}

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
            f"{MODEL_PATH} not found — run `python src/train_final_model.py` first."
        )
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return joblib.load(MODEL_PATH), meta


def score_frame(df: pd.DataFrame = None, csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Adds a `risk` column (P(failure within 24h)) and a `risk_band`."""
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
          start=None, end=None, last_hours=None, limit=400) -> pd.DataFrame:
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
    if end:
        out = out[out["timestamp"] <= pd.Timestamp(end)]
    return out.sort_values("timestamp").tail(limit)


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
