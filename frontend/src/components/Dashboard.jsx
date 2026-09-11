import { useMemo, useState } from "react";
import Sparkline from "./Sparkline.jsx";
import { formatThreshold, formatTimestamp } from "../format.js";

const pct = (v) => `${(v * 100).toFixed(v >= 0.1 ? 0 : 1)}%`;

const LINES = ["Line A", "Line B", "Line C"];
const BANDS = [
  { value: "HIGH", label: "High risk" },
  { value: "MEDIUM", label: "Watch" },
  { value: "LOW", label: "Low risk" },
];

export default function Dashboard({ fleet, loading, meta, selected, onSelect }) {
  // All hooks run on every render, before any early return — a hook called
  // only on some renders (e.g. after "if (!fleet) return null") changes the
  // hook count between renders and crashes React (error #310). guardedFilter
  // covers the case where `fleet` is not yet loaded.
  const [lineFilter, setLineFilter] = useState(null);
  const [bandFilter, setBandFilter] = useState(null);
  const [coldFilter, setColdFilter] = useState(null); // null | true | false

  const machines = fleet?.machines ?? [];
  const filtered = useMemo(() => {
    return machines.filter((m) => {
      if (lineFilter && m.line !== lineFilter) return false;
      if (bandFilter && m.risk_band !== bandFilter) return false;
      if (coldFilter === true && !m.cold_start) return false;
      if (coldFilter === false && m.cold_start) return false;
      return true;
    });
  }, [machines, lineFilter, bandFilter, coldFilter]);

  if (loading && !fleet) {
    return (
      <section className="panel fleet">
        <div className="panel-head">
          <h2>Fleet</h2>
        </div>
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="skeleton" style={{ width: `${92 - i * 4}%` }} />
        ))}
      </section>
    );
  }
  if (!fleet) return null;

  const { counts, as_of, not_yet_reporting = [] } = fleet;
  const calm = counts.HIGH === 0 && counts.MEDIUM === 0;
  const threshold = meta?.risk_bands?.high ?? 0.2;

  const anyFilterActive = lineFilter || bandFilter || coldFilter !== null;
  const clearFilters = () => {
    setLineFilter(null);
    setBandFilter(null);
    setColdFilter(null);
  };

  return (
    <section className="panel fleet">
      <div className="panel-head">
        <h2>Fleet</h2>
        <span className="mono muted tiny">{formatTimestamp(as_of)}</span>
      </div>
      <p className="section-caption fleet-caption">
        Every machine ranked by its estimated chance of failure in the next {meta?.horizon_hours ?? 24} hours.
        Select a row to see the details.
      </p>

      <div className="counts">
        {BANDS.map(({ value, label }) => (
          <div
            key={value}
            className={[
              "count",
              `band-${value.toLowerCase()}`,
              counts[value] > 0 ? "is-live" : "",
            ].join(" ")}
          >
            <span className="count-n">{counts[value]}</span>
            <span className="count-l">{label}</span>
          </div>
        ))}
      </div>

      {calm && (
        <p className="note">
          Every machine is in the low band at this moment — see “historical
          model performance” above for what the system has actually caught.
        </p>
      )}

      {not_yet_reporting.length > 0 && (
        <p className="note">
          Showing {machines.length} of {fleet.total_machines} machines.{" "}
          {not_yet_reporting.join(", ")} had not started reporting data yet
          as of this time — not an error, just earlier than their history
          begins.
        </p>
      )}

      <div className="filterbar">
        <span className="label">Show</span>
        <div className="filter-group">
          {LINES.map((l) => (
            <button
              key={l}
              className={`filter-chip${lineFilter === l ? " active" : ""}`}
              aria-pressed={lineFilter === l}
              onClick={() => setLineFilter(lineFilter === l ? null : l)}
            >
              {l}
            </button>
          ))}
        </div>
        <div className="filter-group">
          {BANDS.map(({ value, label }) => (
            <button
              key={value}
              className={`filter-chip${bandFilter === value ? " active" : ""}`}
              aria-pressed={bandFilter === value}
              onClick={() => setBandFilter(bandFilter === value ? null : value)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="filter-group">
          <button
            className={`filter-chip${coldFilter === true ? " active" : ""}`}
            aria-pressed={coldFilter === true}
            onClick={() => setColdFilter(coldFilter === true ? null : true)}
          >
            New machines
          </button>
        </div>
        {anyFilterActive && (
          <button className="filter-clear" onClick={clearFilters} type="button">
            Clear
          </button>
        )}
      </div>

      <div className="table-scroll">
      <table className="fleet-table">
        <caption className="sr-only">Machines ranked by estimated failure risk. Select a row to open its details.</caption>
        <thead>
          <tr>
            <th>Machine</th>
            <th>Line</th>
            <th className="num">Risk</th>
            <th>State</th>
            <th className="num">Vibration</th>
            <th className="num">Temperature</th>
            <th>Recent risk</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((m) => {
            const band = m.risk_band.toLowerCase();
            return (
              <tr
                key={m.machine_id}
                onClick={() => onSelect(m.machine_id)}
                tabIndex={0}
                role="button"
                aria-pressed={selected === m.machine_id}
                aria-label={`Open details for ${m.machine_id}, ${pct(m.risk)} risk, ${m.risk_band.toLowerCase()}`}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSelect(m.machine_id);
                  }
                }}
                className={[
                  "row",
                  `band-${band}`,
                  selected === m.machine_id ? "is-selected" : "",
                ].join(" ")}
              >
                <td>
                  <span className="machine-id">{m.machine_id}</span>
                  {m.cold_start && (
                    <span
                      className="chip"
                      title="Only ~3 days of history and no recorded failures — this score has never been validated for this machine"
                    >
                      new
                    </span>
                  )}
                </td>
                <td className="muted">{m.line.replace("Line ", "")}</td>
                <td className="num mono">{pct(m.risk)}</td>
                <td>
                  <span className={`pill band-${band}`}>{m.risk_band}</span>
                </td>
                <td className="num mono">{m.vibration_mm_s.toFixed(2)}</td>
                <td className="num mono">{m.temperature_c.toFixed(1)}</td>
                <td className="sparkline-cell">
                  <Sparkline points={m.sparkline} threshold={threshold} />
                </td>
              </tr>
            );
          })}
          {filtered.length === 0 && (
            <tr>
              <td colSpan={7} className="no-results">
                No machines match the current filters.
              </td>
            </tr>
          )}
        </tbody>
      </table>
      </div>

      {meta && (
        <p className="foot mono muted">
          High risk starts at {formatThreshold(meta.risk_bands.high)} · watch starts at {formatThreshold(meta.risk_bands.medium)} ·{" "}
          looks ahead {meta.horizon_hours} hours
        </p>
      )}
    </section>
  );
}
