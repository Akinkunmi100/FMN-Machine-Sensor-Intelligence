"""
Phase 1 — baseline + candidate models for 24h-horizon failure prediction,
with time-ordered, embargoed validation.

Produces a real comparison table (AUC-PR + confusion counts) across:
  - a primary 80/20 time-ordered split
  - two expanding-window CV folds (train=Jan-Feb/test=Mar, train=Jan-Mar/test=Apr)
No model is chosen here — that call is the user's, once they see this table.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from features import build_dataset

CSV_PATH = "project2_manufacturing_sensors.csv"
HORIZON_HOURS = 24
THRESHOLD = 0.5


def embargo(train_df: pd.DataFrame, cutoff: pd.Timestamp, horizon_hours: int) -> pd.DataFrame:
    """Drop training rows whose horizon window (t, t+horizon] crosses the
    cutoff — their label was determined partly by data on the test side."""
    boundary = cutoff - pd.Timedelta(hours=horizon_hours)
    return train_df[train_df["timestamp"] <= boundary]


def make_splits(df: pd.DataFrame):
    t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
    # primary cutoff computed the same way validated with the user: 80% of
    # the way through the established machines' raw calendar range.
    raw_t_min, raw_t_max = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-30 23:00:00")
    primary_cutoff = raw_t_min + 0.8 * (raw_t_max - raw_t_min)

    fold1_cutoff = pd.Timestamp("2026-03-01")   # train=Jan+Feb, test=Mar
    fold2_cutoff = pd.Timestamp("2026-04-01")   # train=Jan-Mar, test=Apr

    splits = {}

    train = embargo(df[df["timestamp"] <= primary_cutoff], primary_cutoff, HORIZON_HOURS)
    test = df[df["timestamp"] > primary_cutoff]
    splits["Primary 80/20 (cutoff 2026-04-06)"] = (train, test)

    train = embargo(df[df["timestamp"] < fold1_cutoff], fold1_cutoff, HORIZON_HOURS)
    test = df[(df["timestamp"] >= fold1_cutoff) & (df["timestamp"] < fold2_cutoff)]
    splits["CV fold 1 (train Jan-Feb, test Mar)"] = (train, test)

    train = embargo(df[df["timestamp"] < fold2_cutoff], fold2_cutoff, HORIZON_HOURS)
    test = df[df["timestamp"] >= fold2_cutoff]
    splits["CV fold 2 (train Jan-Mar, test Apr)"] = (train, test)

    return splits


def build_models(y_train: pd.Series):
    """Fresh, unfitted model instances. XGBoost has no class_weight param,
    so its balanced weighting is computed from this split's own train
    labels via scale_pos_weight = n_negative / n_positive."""
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / n_pos if n_pos else 1.0

    return {
        "Logistic Regression (balanced)": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ]),
        "Decision Tree depth=4 (balanced)": DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=42
        ),
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "HistGradientBoosting (balanced)": HistGradientBoostingClassifier(
            max_depth=6, class_weight="balanced", random_state=42
        ),
        "XGBoost (balanced)": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            random_state=42, n_jobs=-1,
        ),
    }


def fit_predict(model, train, test, feature_cols):
    """Fit on train, return predicted probabilities for test (positive class)."""
    X_train, y_train = train[feature_cols], train["label"]
    X_test = test[feature_cols]
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def score(y_test, proba):
    auc_pr = average_precision_score(y_test, proba)
    preds = (proba >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    return auc_pr, tp, fp, fn, tn


ENSEMBLE_MEMBERS = [
    "Random Forest (balanced)",
    "HistGradientBoosting (balanced)",
    "XGBoost (balanced)",
]


def main():
    df, feature_cols = build_dataset(CSV_PATH, horizon_hours=HORIZON_HOURS)
    splits = make_splits(df)

    print("=" * 100)
    print("Split sizes (after embargo purge) and positive counts")
    print("=" * 100)
    for split_name, (train, test) in splits.items():
        print(f"\n{split_name}")
        print(f"  train: {len(train):,} rows, {train['label'].sum()} positives")
        print(f"  test:  {len(test):,} rows, {test['label'].sum()} positives")

    print("\n" + "=" * 100)
    print("AUC-PR comparison table (naive baseline + 5 candidates + 2 ensembles x 3 splits)")
    print("=" * 100)

    rows = []
    model_names_seen = []
    for split_name, (train, test) in splits.items():
        y_test = test["label"]
        naive_auc_pr = y_test.mean()  # no-skill AUC-PR = prevalence
        rows.append({
            "split": split_name, "model": "Naive baseline (predict no-failure)",
            "auc_pr": naive_auc_pr, "tp": 0, "fp": 0,
            "fn": int(y_test.sum()), "tn": int((y_test == 0).sum()),
        })

        models = build_models(train["label"])
        proba_by_model = {}
        for model_name, model in models.items():
            proba = fit_predict(model, train, test, feature_cols)
            proba_by_model[model_name] = proba
            auc_pr, tp, fp, fn, tn = score(y_test, proba)
            rows.append({
                "split": split_name, "model": model_name, "auc_pr": auc_pr,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })
            if model_name not in model_names_seen:
                model_names_seen.append(model_name)

        # Ensembles: no new training, just combine the fitted models' own
        # test-set probabilities already computed above.
        members = [m for m in ENSEMBLE_MEMBERS if m in proba_by_model]
        member_probas = np.column_stack([proba_by_model[m] for m in members])

        avg_proba = member_probas.mean(axis=1)
        auc_pr, tp, fp, fn, tn = score(y_test, avg_proba)
        rows.append({
            "split": split_name, "model": "Ensemble: averaged probability",
            "auc_pr": auc_pr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

        vote_share = (member_probas >= THRESHOLD).mean(axis=1)  # fraction of members voting positive
        auc_pr, tp, fp, fn, tn = score(y_test, vote_share)
        rows.append({
            "split": split_name, "model": "Ensemble: majority vote",
            "auc_pr": auc_pr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    print(f"\nEnsemble members ({len(ENSEMBLE_MEMBERS)}): {', '.join(ENSEMBLE_MEMBERS)}")

    results = pd.DataFrame(rows)
    ordered_models = [
        "Naive baseline (predict no-failure)", *model_names_seen,
        "Ensemble: averaged probability", "Ensemble: majority vote",
    ]
    pivot = results.pivot(index="model", columns="split", values="auc_pr")
    pivot = pivot.reindex(ordered_models)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 10)
    print("\nAUC-PR by model x split:")
    print(pivot.round(4))

    print(f"\nConfusion counts at threshold={THRESHOLD} (TP / FP / FN / TN):")
    for split_name in splits:
        print(f"\n{split_name}")
        sub = results[results["split"] == split_name].set_index("model").reindex(ordered_models)
        for model_name, r in sub.iterrows():
            print(f"  {model_name:<38} TP={r['tp']:>3.0f}  FP={r['fp']:>4.0f}  "
                  f"FN={r['fn']:>3.0f}  TN={r['tn']:>5.0f}")


if __name__ == "__main__":
    main()
