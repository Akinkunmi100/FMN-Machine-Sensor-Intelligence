"""Shared project configuration.

Paths, model settings, and operational thresholds live here so training,
scoring, and the API cannot silently drift apart.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "project2_manufacturing_sensors.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "failure_risk_rf.joblib"
METADATA_PATH = MODEL_DIR / "failure_risk_rf.meta.json"
CURRENT_SCORES_PATH = MODEL_DIR / "current_scores.parquet"
HISTORICAL_SCORES_PATH = MODEL_DIR / "historical_scores_oos.parquet"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

PREDICTION_HORIZON_HOURS = 24
ALERT_THRESHOLD = 0.20
COLD_START_MACHINES = ("MCH-300", "MCH-301")

# HIGH (ALERT_THRESHOLD) is the validated operating point from Phase 2: 0.20
# caught 19/19 breakdowns across every held-out window.
#
# MEDIUM ("watch") is 0.02, signed off after seeing the score distribution.
# The model is bimodal — 92.95% of machine-hours score exactly 0.0000 and the
# 97th percentile is already 0.1367 — so a middle band is thin wherever it is
# cut. 0.02 was chosen over a rounder 0.10 because it is the widest useful
# watch tier (1.14% of hours vs 0.33%), and anything above it already sits in
# the top 5% of all hours for that fleet.
RISK_BANDS = {"high": ALERT_THRESHOLD, "medium": 0.02}

# min_samples_leaf=20 was adopted after a post-Phase-2 re-examination. On its
# own it is WORSE operationally (+17 false episodes at equal event capture);
# combined with the per-machine relative features it is better (-16). Neither
# change helps alone — ratio features are higher-variance and need the heavier
# leaf constraint, and the constraint needs the richer features to exploit.
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}
