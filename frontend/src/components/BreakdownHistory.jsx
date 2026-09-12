// Shows every recorded breakdown in one table, filterable by peak risk band.
// Each row is clickable — jumps the dashboard to that exact failure moment.

import { useState, useMemo } from "react";
import { formatTimestamp } from "../format.js";

const FILTERS = [
  { value: null, label: "All" },
  { value: "HIGH", label: "High risk" },
  { value: "MEDIUM", label: "Watch" },
  { value: "LOW", label: "Missed" },
];

const pct = (v) => `${(v * 100).toFixed(v >= 0.1 ? 0 : 1)}%`;

export default function BreakdownHistory({ meta, onJumpTo }) {
  const events = meta?.activity?.events;
  if (!events || events.length === 0) return null;

  const [bandFilter, setBandFilter] = useState(null);

  const filtered = useMemo(
    () =>
      bandFilter ? events.filter((e) => e.peak_band === bandFilter) : events,
    [events, bandFilter]
  );

  const filterCounts = useMemo(() => {
    const c = { HIGH: 0, MEDIUM: 0, LOW: 0 };
    events.forEach((e) => c[e.peak_band]++);
    return c;
  }, [events]);

  return (
    <section className="panel breakdown-history">
      <div className="panel-head">
        <h2>Breakdown History</h2>
      </div>
      <p className="section-caption">
        Every recorded machine failure and the highest risk level the model
        assigned in the 24 hours before it happened. Select a row to
        time-travel the dashboard to that moment.
      </p>

      <div className="filterbar">
        <span className="label">Filter</span>
        <div className="filter-group">
          {FILTERS.map(({ value, label }) => {
            const count =
              value === null ? events.length : filterCounts[value] ?? 0;
            return (
              <button
                key={label}
                className={`filter-chip${bandFilter === value ? " active" : ""}`}
                aria-pressed={bandFilter === value}
                onClick={() => setBandFilter(bandFilter === value ? null : value)}
              >
                {label} ({count})
              </button>
            );
          })}
        </div>
        {bandFilter !== null && (
          <button
            className="filter-clear"
            onClick={() => setBandFilter(null)}
            type="button"
          >
            Clear
          </button>
        )}
      </div>

      <div className="table-scroll">
        <table className="breakdown-table">
          <caption className="sr-only">
            Recorded breakdowns with pre-failure risk levels. Select a row to
            view the fleet at that moment.
          </caption>
          <thead>
            <tr>
              <th>Machine</th>
              <th>Line</th>
              <th>Breakdown time</th>
              <th>Pre-failure state</th>
              <th className="num">Peak risk</th>
              <th className="num">Lead time</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((e) => {
              const band = e.peak_band.toLowerCase();
              return (
                <tr
                  key={`${e.machine_id}-${e.timestamp}`}
                  className={`row band-${band}`}
                  onClick={() => onJumpTo(e.timestamp)}
                  tabIndex={0}
                  role="button"
                  aria-label={`Jump to ${e.machine_id} breakdown at ${formatTimestamp(e.timestamp)}`}
                  onKeyDown={(ev) => {
                    if (ev.key === "Enter" || ev.key === " ") {
                      ev.preventDefault();
                      onJumpTo(e.timestamp);
                    }
                  }}
                >
                  <td>
                    <span className="machine-id">{e.machine_id}</span>
                  </td>
                  <td className="muted">{e.line.replace("Line ", "")}</td>
                  <td className="mono">{formatTimestamp(e.timestamp)}</td>
                  <td>
                    <span className={`pill band-${band}`}>
                      {e.peak_band === "MEDIUM" ? "WATCH" : e.peak_band}
                    </span>
                  </td>
                  <td className="num mono">{pct(e.peak_risk)}</td>
                  <td className="num mono">
                    {e.lead_hours != null ? `${e.lead_hours}h` : "—"}
                  </td>
                </tr>
              );
            })}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={6} className="no-results">
                  No breakdowns match this filter.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
