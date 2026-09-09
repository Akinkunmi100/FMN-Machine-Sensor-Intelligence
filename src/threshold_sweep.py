"""
Threshold sweep for Random Forest and Logistic Regression on the primary
80/20 split, so confusion counts can be compared at matched thresholds
before the final model choice. No new training beyond the two models
already in the Phase 1 comparison — same fit, just scored at more cutoffs.
"""

import pandas as pd
from sklearn.metrics import confusion_matrix

from features import build_dataset
from train_baseline import CSV_PATH, HORIZON_HOURS, build_models, fit_predict, make_splits

THRESHOLDS = [0.5, 0.3, 0.2]
MODELS_TO_SWEEP = [
    "Random Forest (balanced, leaf=20) [SHIPPED]",
    "Random Forest (balanced)",
    "Logistic Regression (balanced)",
]


def confusion_at(y_test, proba, threshold):
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    return tp, fp, fn, tn


def main():
    df, feature_cols = build_dataset(CSV_PATH, horizon_hours=HORIZON_HOURS)
    splits = make_splits(df)
    train, test = splits["Primary 80/20 (cutoff 2026-04-06)"]
    y_test = test["label"]

    models = build_models(train["label"])

    print("=" * 90)
    print("Threshold sweep — primary split (test: 8,282 rows, 164 positives)")
    print("=" * 90)

    for model_name in MODELS_TO_SWEEP:
        proba = fit_predict(models[model_name], train, test, feature_cols)
        print(f"\n{model_name}")
        print(f"  {'threshold':<10} {'TP':>4} {'FP':>6} {'FN':>4} {'TN':>6} {'recall':>8} {'precision':>10}")
        for t in THRESHOLDS:
            tp, fp, fn, tn = confusion_at(y_test, proba, t)
            recall = tp / (tp + fn) if (tp + fn) else float("nan")
            precision = tp / (tp + fp) if (tp + fp) else float("nan")
            print(f"  {t:<10} {tp:>4} {fp:>6} {fn:>4} {tn:>6} {recall:>8.3f} {precision:>10.3f}")


if __name__ == "__main__":
    main()
