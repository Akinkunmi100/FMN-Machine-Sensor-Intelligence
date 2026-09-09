"""
Phase 0 — Data Exploration for project2_manufacturing_sensors.csv

Reports real findings only: missingness, class balance, cold-start sparsity
for the 2 new machines, and maintenance-reset behavior. No imputation, no
feature engineering, no modeling happens here — see CLAUDE.md Build Phases.
"""

import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

CSV_PATH = "project2_manufacturing_sensors.csv"
pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 30)
pd.set_option("display.max_columns", 20)


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_data():
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    return df


def report_shape(df):
    section("0. Dataset shape & overview")
    print(f"Rows: {len(df):,}")
    print(f"Machines: {df['machine_id'].nunique()}")
    print(f"Lines: {sorted(df['line'].unique())}")
    print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print("\nColumn dtypes:")
    print(df.dtypes)


def report_missingness(df):
    section("1. Missingness — temperature_c and vibration_mm_s")

    for col in ["temperature_c", "vibration_mm_s"]:
        n_missing = df[col].isna().sum()
        pct_missing = 100 * n_missing / len(df)
        print(f"\n{col}: {n_missing:,} missing / {len(df):,} rows ({pct_missing:.2f}%)")

    print("\n--- Missing % by machine_id ---")
    by_machine = df.groupby("machine_id")[["temperature_c", "vibration_mm_s"]].apply(
        lambda g: pd.Series({
            "temp_missing_pct": 100 * g["temperature_c"].isna().mean(),
            "vib_missing_pct": 100 * g["vibration_mm_s"].isna().mean(),
            "n_rows": len(g),
        })
    )
    print(by_machine.round(2))

    print("\n--- Missing-run analysis (contiguous gap lengths, per machine, per column) ---")
    for col in ["temperature_c", "vibration_mm_s"]:
        run_lengths = []
        for machine_id, g in df.groupby("machine_id"):
            is_na = g[col].isna().to_numpy()
            run = 0
            for v in is_na:
                if v:
                    run += 1
                elif run > 0:
                    run_lengths.append(run)
                    run = 0
            if run > 0:
                run_lengths.append(run)
        if run_lengths:
            s = pd.Series(run_lengths)
            print(f"\n{col}: {len(s)} missing runs across all machines")
            print(f"  run length -> min={s.min()}, median={s.median()}, "
                  f"mean={s.mean():.2f}, max={s.max()}")
            print(f"  runs of length 1 (isolated single points): {(s == 1).sum()} "
                  f"({100 * (s == 1).mean():.1f}% of runs)")
        else:
            print(f"\n{col}: no missing values, no runs to report")


def report_class_balance(df):
    section("2. Class balance — failure_event")

    counts = df["failure_event"].value_counts().sort_index()
    pct = 100 * df["failure_event"].value_counts(normalize=True).sort_index()
    print("Overall counts:")
    for label in counts.index:
        print(f"  failure_event={label}: {counts[label]:,} rows ({pct[label]:.3f}%)")

    print("\n--- Failure rate by machine_id (failures / machine-hours, and total failures) ---")
    by_machine = df.groupby("machine_id").agg(
        n_rows=("failure_event", "size"),
        n_failures=("failure_event", "sum"),
    )
    by_machine["failure_rate_pct"] = 100 * by_machine["n_failures"] / by_machine["n_rows"]
    print(by_machine.sort_values("failure_rate_pct", ascending=False).round(4))

    print("\n--- Failure rate by line ---")
    by_line = df.groupby("line").agg(
        n_rows=("failure_event", "size"),
        n_failures=("failure_event", "sum"),
    )
    by_line["failure_rate_pct"] = 100 * by_line["n_failures"] / by_line["n_rows"]
    print(by_line.sort_values("failure_rate_pct", ascending=False).round(4))


def report_new_machine_sparsity(df):
    section("3. Cold-start sparsity — identifying and sizing the 2 new machines")

    span = df.groupby("machine_id").agg(
        n_rows=("timestamp", "size"),
        first_ts=("timestamp", "min"),
        last_ts=("timestamp", "max"),
    )
    span["span_days"] = (span["last_ts"] - span["first_ts"]).dt.total_seconds() / 86400
    span = span.sort_values("span_days")
    span["span_days"] = span["span_days"].round(2)
    print("All machines, sorted by history span (shortest first):")
    print(span)

    n_new = 2
    new_machines = span.index[:n_new].tolist()
    print(f"\nShortest-history machines (candidate 'new machines'): {new_machines}")

    print("\n--- Detail for candidate new machines ---")
    for m in new_machines:
        sub = df[df["machine_id"] == m]
        n_fail = sub["failure_event"].sum()
        print(f"\n{m}: {len(sub)} rows, span={span.loc[m, 'span_days']:.2f} days, "
              f"failures={n_fail} ({100 * n_fail / len(sub):.3f}%)")
        print(f"  temp missing: {sub['temperature_c'].isna().sum()}, "
              f"vib missing: {sub['vibration_mm_s'].isna().sum()}")

    print("\n--- Compare to the established machines (remaining machines) ---")
    established = span.index[n_new:]
    est_span = span.loc[established, "span_days"]
    print(f"Established machines: n={len(established)}, "
          f"span_days min={est_span.min():.2f}, median={est_span.median():.2f}, "
          f"max={est_span.max():.2f}")


def report_maintenance_resets(df):
    section("4. Maintenance resets — run_hours_since_maintenance")

    reset_summary = []
    boundary_rows_for_spotcheck = []

    for machine_id, g in df.groupby("machine_id"):
        g = g.reset_index(drop=True)
        rh = g["run_hours_since_maintenance"]
        # a reset = value drops to 0 (or drops relative to previous row) after having been >0
        is_reset = (rh == 0) & (rh.shift(1).fillna(0) != 0)
        n_resets = is_reset.sum()

        # run lengths between resets = consecutive hours before each reset
        reset_positions = g.index[is_reset].tolist()
        run_lengths = []
        prev = 0
        for pos in reset_positions:
            run_lengths.append(pos - prev)
            prev = pos
        run_lengths.append(len(g) - prev)  # trailing run to end of data

        # does a reset follow a failure_event within the reset row or the row before it?
        resets_after_failure = 0
        for pos in reset_positions:
            window_start = max(0, pos - 1)
            if g.loc[window_start:pos, "failure_event"].sum() > 0:
                resets_after_failure += 1

        reset_summary.append({
            "machine_id": machine_id,
            "n_resets": n_resets,
            "median_run_len_hrs": pd.Series(run_lengths).median() if run_lengths else None,
            "min_run_len_hrs": min(run_lengths) if run_lengths else None,
            "max_run_len_hrs": max(run_lengths) if run_lengths else None,
            "resets_following_failure": resets_after_failure,
        })

        if reset_positions:
            pos = reset_positions[0]
            boundary_rows_for_spotcheck.append((machine_id, g.loc[max(0, pos - 2):pos + 2]))

    summary_df = pd.DataFrame(reset_summary).set_index("machine_id")
    print("Reset summary by machine:")
    print(summary_df)

    total_resets = summary_df["n_resets"].sum()
    total_after_failure = summary_df["resets_following_failure"].sum()
    print(f"\nTotal resets across all machines: {total_resets}")
    print(f"Resets where failure_event=1 occurred at or immediately before the reset: "
          f"{total_after_failure} ({100 * total_after_failure / total_resets:.1f}% of resets)"
          if total_resets else "No resets found.")

    print("\n--- Spot-check: raw rows around the first reset for a couple of machines ---")
    for machine_id, window in boundary_rows_for_spotcheck[:3]:
        print(f"\n{machine_id}:")
        print(window[["timestamp", "run_hours_since_maintenance", "failure_event"]])


def main():
    df = load_data()
    report_shape(df)
    report_missingness(df)
    report_class_balance(df)
    report_new_machine_sparsity(df)
    report_maintenance_resets(df)


if __name__ == "__main__":
    main()
