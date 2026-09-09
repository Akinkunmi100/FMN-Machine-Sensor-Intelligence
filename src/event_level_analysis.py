"""
Event-level (operational) evaluation for the primary split.

Row-level confusion counts overstate the problem size: 164 positive rows in
the primary test set are 7 real failure events, each expanded into a 24h
warning window. What the plant actually cares about:
  - Did we raise ANY alert in the 24h before each real failure? (events caught)
  - How much lead time did that first alert give? (hours to act)
  - How many DISTINCT false alarm episodes did operators have to chase?
    (consecutive alert hours on one machine = one episode, not 24 alarms)

This is the view the final model choice should be made on.
"""

import numpy as np
import pandas as pd

from features import NEW_MACHINES, load_data
from train_baseline import CSV_PATH, HORIZON_HOURS, build_models, fit_predict, make_splits
from features import build_dataset

SWEEP = {
    "Random Forest (balanced, leaf=20) [SHIPPED]": [0.5, 0.3, 0.2],
    "Random Forest (balanced)": [0.5, 0.3, 0.2],
    "Logistic Regression (balanced)": [0.5, 0.3, 0.2],
}


def get_failure_events(csv_path):
    raw = load_data(csv_path)
    established = raw[~raw["machine_id"].isin(NEW_MACHINES)]
    fails = established[established["failure_event"] == 1][["machine_id", "timestamp"]]
    return fails.sort_values("timestamp").reset_index(drop=True)


def analyze(test, proba, threshold, failures, horizon_hours=HORIZON_HOURS):
    """Returns event-level catch stats and distinct false-alarm episode count."""
    test = test.copy()
    test["proba"] = proba
    test["alert"] = (test["proba"] >= threshold).astype(int)

    # --- which real failures fall in the test period, and were they caught? ---
    caught, missed, lead_times = [], [], []
    true_window_index = set()

    for _, f in failures.iterrows():
        machine, T = f["machine_id"], f["timestamp"]
        window = test[
            (test["machine_id"] == machine)
            & (test["timestamp"] >= T - pd.Timedelta(hours=horizon_hours))
            & (test["timestamp"] < T)
        ]
        if window.empty:
            continue  # this failure's warning window isn't in the test period

        true_window_index.update(window.index.tolist())

        alerts = window[window["alert"] == 1]
        if len(alerts):
            first_alert = alerts["timestamp"].min()
            lead = (T - first_alert).total_seconds() / 3600
            caught.append((machine, T, lead, len(alerts)))
            lead_times.append(lead)
        else:
            missed.append((machine, T))

    # --- distinct false-alarm episodes (consecutive alert hours = 1 episode) ---
    false_alert_rows = test[(test["alert"] == 1) & (~test.index.isin(true_window_index))]
    n_false_rows = len(false_alert_rows)

    episodes = 0
    for machine, g in false_alert_rows.groupby("machine_id"):
        ts = g["timestamp"].sort_values()
        # a gap of more than 1 hour starts a new episode
        gaps = ts.diff() > pd.Timedelta(hours=1)
        episodes += 1 + int(gaps.sum()) if len(ts) else 0

    return {
        "threshold": threshold,
        "events_in_test": len(caught) + len(missed),
        "events_caught": len(caught),
        "events_missed": len(missed),
        "median_lead_hrs": float(np.median(lead_times)) if lead_times else float("nan"),
        "min_lead_hrs": float(np.min(lead_times)) if lead_times else float("nan"),
        "false_alarm_rows": n_false_rows,
        "false_alarm_episodes": episodes,
        "missed_detail": missed,
        "caught_detail": caught,
    }


def main():
    df, feature_cols = build_dataset(CSV_PATH, horizon_hours=HORIZON_HOURS)
    splits = make_splits(df)
    train, test = splits["Primary 80/20 (cutoff 2026-04-06)"]
    failures = get_failure_events(CSV_PATH)

    test_period_start = test["timestamp"].min()
    test_period_end = test["timestamp"].max()
    n_machines = test["machine_id"].nunique()
    n_days = (test_period_end - test_period_start).total_seconds() / 86400

    print("=" * 96)
    print("EVENT-LEVEL ANALYSIS - primary split")
    print("=" * 96)
    print(f"Test period: {test_period_start} -> {test_period_end} "
          f"({n_days:.1f} days, {n_machines} machines)")
    print(f"Test rows: {len(test):,} | positive ROWS: {int(test['label'].sum())} "
          f"| real failure EVENTS in window: "
          f"{sum(1 for _, f in failures.iterrows() if not test[(test['machine_id'] == f['machine_id']) & (test['timestamp'] >= f['timestamp'] - pd.Timedelta(hours=HORIZON_HOURS)) & (test['timestamp'] < f['timestamp'])].empty)}")

    models = build_models(train["label"])

    all_results = {}
    for model_name, thresholds in SWEEP.items():
        proba = fit_predict(models[model_name], train, test, feature_cols)
        rows = []
        for t in thresholds:
            rows.append(analyze(test, proba, t, failures))
        all_results[model_name] = rows

    print("\n" + "=" * 96)
    print("Events caught vs. false-alarm burden")
    print("=" * 96)
    for model_name, rows in all_results.items():
        print(f"\n{model_name}")
        print(f"  {'thresh':<8} {'caught':>10} {'missed':>7} {'med lead':>9} "
              f"{'min lead':>9} {'FA rows':>8} {'FA episodes':>12} {'FA/machine/wk':>14}")
        for r in rows:
            fa_per_machine_week = r["false_alarm_episodes"] / n_machines / (n_days / 7)
            caught_str = f"{r['events_caught']}/{r['events_in_test']}"
            print(f"  {r['threshold']:<8} {caught_str:>10} {r['events_missed']:>7} "
                  f"{r['median_lead_hrs']:>8.1f}h {r['min_lead_hrs']:>8.1f}h "
                  f"{r['false_alarm_rows']:>8} {r['false_alarm_episodes']:>12} "
                  f"{fa_per_machine_week:>14.2f}")

    print("\n" + "=" * 96)
    print("Missed events detail (which failures slipped through)")
    print("=" * 96)
    for model_name, rows in all_results.items():
        for r in rows:
            if r["missed_detail"]:
                print(f"\n{model_name} @ threshold {r['threshold']}: "
                      f"{len(r['missed_detail'])} missed")
                for machine, T in r["missed_detail"]:
                    print(f"    {machine} at {T}")

    # --- generalization check: does the chosen operating point hold up on the
    # CV folds, or was 0.2 just tuned to this one test window? ---
    print("\n" + "=" * 96)
    print("Operating-point generalization check across all splits")
    print("=" * 96)
    operating_points = [
        ("Random Forest (balanced, leaf=20) [SHIPPED]", 0.2),
        ("Random Forest (balanced)", 0.2),
        ("Logistic Regression (balanced)", 0.3),
    ]
    for split_name, (sp_train, sp_test) in splits.items():
        sp_models = build_models(sp_train["label"])
        sp_days = (sp_test["timestamp"].max() - sp_test["timestamp"].min()).total_seconds() / 86400
        sp_machines = sp_test["machine_id"].nunique()
        print(f"\n{split_name}  ({sp_days:.1f} days, {sp_machines} machines)")
        for model_name, t in operating_points:
            sp_proba = fit_predict(sp_models[model_name], sp_train, sp_test, feature_cols)
            r = analyze(sp_test, sp_proba, t, failures)
            fa_rate = r["false_alarm_episodes"] / sp_machines / (sp_days / 7)
            print(f"  {model_name:<34} @ {t}: "
                  f"caught {r['events_caught']}/{r['events_in_test']}, "
                  f"median lead {r['median_lead_hrs']:.1f}h, "
                  f"{r['false_alarm_episodes']} false episodes "
                  f"({fa_rate:.2f} per machine per week)")


if __name__ == "__main__":
    main()
