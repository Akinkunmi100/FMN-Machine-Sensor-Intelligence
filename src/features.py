"""
Feature and label engineering for the 24h-horizon failure prediction task.

Every engineered feature is strictly causal: it only uses data at or before
the row's own timestamp t (trailing rolling windows, backward diffs). Nothing
here may look forward into (t, t+horizon] — that window is reserved for the
label. This matters specifically because Phase 0 found that 77.3% of
maintenance resets coincide with or immediately follow the failure event
they'd otherwise seem to "predict" — a forward-looking maintenance feature
would leak the label almost directly.
"""

import numpy as np
import pandas as pd

NEW_MACHINES = ["MCH-300", "MCH-301"]

SENSOR_COLUMNS = ["temperature_c", "vibration_mm_s"]

FEATURE_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "temp_roll_mean_6h",
    "temp_roll_std_6h",
    "temp_roll_mean_24h",
    "temp_roll_std_24h",
    "temp_diff_1h",
    "temp_diff_6h",
    "vib_roll_mean_6h",
    "vib_roll_std_6h",
    "vib_roll_mean_24h",
    "vib_roll_std_24h",
    "vib_diff_1h",
    "vib_diff_6h",
    "run_hours_since_maintenance",
    "recently_reset_24h",
    "line_Line A",
    "line_Line B",
    "line_Line C",
]


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    return df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)


def impute_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Per-machine forward-fill, matching Phase 0's finding that ~99% of
    gaps are isolated single-hour points. bfill only covers a leading NaN
    at the very start of a machine's series (ffill can't reach it)."""
    df = df.copy()
    for col in SENSOR_COLUMNS:
        df[col] = df.groupby("machine_id")[col].transform(lambda s: s.ffill().bfill())
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds causal rolling/diff features per machine, plus one-hot line
    columns. `df` must already have imputed sensor columns."""
    df = df.copy()
    machine_ids = df["machine_id"]
    grouped = df.groupby("machine_id", group_keys=False)

    def _per_machine(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        for col, prefix in [("temperature_c", "temp"), ("vibration_mm_s", "vib")]:
            g[f"{prefix}_roll_mean_6h"] = g[col].rolling(6, min_periods=1).mean()
            g[f"{prefix}_roll_std_6h"] = g[col].rolling(6, min_periods=1).std().fillna(0.0)
            g[f"{prefix}_roll_mean_24h"] = g[col].rolling(24, min_periods=1).mean()
            g[f"{prefix}_roll_std_24h"] = g[col].rolling(24, min_periods=1).std().fillna(0.0)
            g[f"{prefix}_diff_1h"] = g[col].diff(1).fillna(0.0)
            g[f"{prefix}_diff_6h"] = g[col].diff(6).fillna(0.0)
        g["recently_reset_24h"] = (
            g["run_hours_since_maintenance"].rolling(24, min_periods=1).min() < 24
        ).astype(int)
        return g

    # pandas >=2.2 drops the grouping column from each group passed to apply,
    # so machine_id has to be reattached by index afterward.
    df = grouped.apply(_per_machine)
    df["machine_id"] = machine_ids.reindex(df.index)
    line_dummies = pd.get_dummies(df["line"], prefix="line")
    for col in ["line_Line A", "line_Line B", "line_Line C"]:
        if col not in line_dummies.columns:
            line_dummies[col] = 0
    df = pd.concat([df.reset_index(drop=True), line_dummies.reset_index(drop=True)], axis=1)
    return df


def build_labels(df: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
    """label=1 if failure_event=1 occurs in (t, t+horizon_hours] for that
    machine. Rows in the trailing `horizon_hours` of each machine's series
    are dropped (censored — we don't know the true future outcome)."""
    df = df.copy()
    machine_ids = df["machine_id"]

    def _label_machine(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        # reverse the series, take a trailing rolling sum (= forward window
        # in original order) over the horizon, excluding the current row.
        fail = g["failure_event"].to_numpy()
        n = len(fail)
        label = np.zeros(n, dtype=int)
        for i in range(n):
            end = min(n, i + 1 + horizon_hours)
            label[i] = 1 if fail[i + 1:end].sum() > 0 else 0
        g["label"] = label
        g["censored"] = False
        tail = max(0, n - horizon_hours)
        g.iloc[tail:, g.columns.get_loc("censored")] = True
        return g

    df = df.groupby("machine_id", group_keys=False).apply(_label_machine)
    df["machine_id"] = machine_ids.reindex(df.index)
    return df[~df["censored"]].drop(columns=["censored"]).reset_index(drop=True)


def build_dataset(csv_path: str, horizon_hours: int = 24):
    """Full pipeline for established machines only. Returns (df, feature_cols)."""
    raw = load_data(csv_path)
    established = raw[~raw["machine_id"].isin(NEW_MACHINES)].reset_index(drop=True)
    imputed = impute_sensors(established)
    featured = engineer_features(imputed)
    labeled = build_labels(featured, horizon_hours=horizon_hours)
    return labeled, FEATURE_COLUMNS
