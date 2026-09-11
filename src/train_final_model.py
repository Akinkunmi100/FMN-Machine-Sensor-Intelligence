"""
Phase 2 — fit and persist the selected model.

Selected: Random Forest (balanced class weights), alert threshold 0.20.
Selection evidence lives in docs/model_selection.md; this script only
produces the artifact the app loads at boot so nothing is retrained at
request time.

Refit policy: the metrics in the writeup come from held-out time-ordered
evaluation (see train_baseline.py / event_level_analysis.py). The shipped
artifact is then refit on ALL established-machine history, which is
standard practice once an approach is validated — more history, same
hyperparameters. The saved metadata records exactly what it saw.
"""

import json

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from .config import (
    ALERT_THRESHOLD,
    DATA_PATH,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PARAMS,
    MODEL_PATH,
    PREDICTION_HORIZON_HOURS,
)
from .features import NEW_MACHINES, build_dataset

CSV_PATH = str(DATA_PATH)
HORIZON_HOURS = PREDICTION_HORIZON_HOURS


def main():
    df, feature_cols = build_dataset(CSV_PATH, horizon_hours=HORIZON_HOURS)

    X, y = df[feature_cols], df["label"]
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X, y)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_type": "RandomForestClassifier",
        "model_params": MODEL_PARAMS,
        "alert_threshold": ALERT_THRESHOLD,
        "prediction_horizon_hours": HORIZON_HOURS,
        "target": "P(failure_event occurs in (t, t+24h] for this machine)",
        "feature_columns": feature_cols,
        "trained_on": {
            "machines": sorted(df["machine_id"].unique().tolist()),
            "excluded_machines": NEW_MACHINES,
            "n_rows": int(len(df)),
            "n_positive_rows": int(y.sum()),
            "positive_rate": round(float(y.mean()), 6),
            "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        },
        "sklearn_version": sklearn.__version__,
        "notes": (
            "Refit on all established-machine history after time-ordered "
            "validation. Cold-start machines MCH-300/301 are excluded from "
            "training but CAN be scored: no machine_id feature is used, so "
            "the model generalizes to unseen machines. Their scores are "
            "unvalidated (those machines have zero recorded failures)."
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    importances = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
    )

    print(f"Saved model    -> {MODEL_PATH}")
    print(f"Saved metadata -> {METADATA_PATH}")
    print(f"\nTrained on {len(df):,} rows ({int(y.sum())} positive, "
          f"{100 * y.mean():.3f}%) from {df['machine_id'].nunique()} machines")
    print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"\nFeature importances (top 12 of {len(feature_cols)}):")
    for name, value in importances.head(12).items():
        bar = "#" * int(round(value * 120))
        print(f"  {name:<32} {value:.4f}  {bar}")

    print(f"\nAll {len(feature_cols)} importances as JSON-ready dict:")
    print(json.dumps({k: round(float(v), 5) for k, v in importances.items()}, indent=2))


if __name__ == "__main__":
    main()
