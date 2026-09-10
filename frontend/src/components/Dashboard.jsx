import { useMemo, useState } from "react";
import Sparkline from "./Sparkline.jsx";

const pct = (v) => `${(v * 100).toFixed(v >= 0.1 ? 0 : 1)}%`;

const LINES = ["Line A", "Line B", "Line C"];
const BANDS = ["HIGH", "MEDIUM", "LOW"];

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
        <span className="mono muted tiny">{as_of}</span>
      </div>

      <div className="counts">
        {BANDS.map((band) => (
          <div
            key={band}
            className={[
              "count",
              `band-${band.toLowerCase()}`,
              counts[band] > 0 ? "is-live" : "",
            ].join(" ")}
          >
            <span className="count-n">{counts[band]}</span>
            <span className="count-l">{band}</span>
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
        <span className="label">Filter</span>
        <div className="filter-group">
          {LINES.map((l) => (
            <button
              key={l}
              className={`filter-chip${lineFilter === l ? " active" : ""}`}
              onClick={() => setLineFilter(lineFilter === l ? null : l)}
            >
              {l.replace("Line ", "L")}
            </button>
          ))}
        </div>
        <div className="filter-group">
          {BANDS.map((b) => (
            <button
              key={b}
              className={`filter-chip${bandFilter === b ? " active" : ""}`}
              onClick={() => setBandFilter(bandFilter === b ? null : b)}
            >
              {b}
            </button>
          ))}
        </div>
        <div className="filter-group">
          <button
            className={`filter-chip${coldFilter === true ? " active" : ""}`}
            onClick={() => setColdFilter(coldFilter === true ? null : true)}
          >
            New machines
          </button>
        </div>
        {anyFilterActive && (
          <button className="filter-clear" onClick={clearFilters}>
            Clear
          </button>
        )}
      </div>

      <table className="fleet-table">
        <thead>
          <tr>
            <th>Machine</th>
            <th>Line</th>
            <th className="num">Risk</th>
            <th>State</th>
            <th className="num">Vib</th>
            <th className="num">Temp</th>
            <th>30-day history</th>
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
                onKeyDown={(e) => e.key === "Enter" && onSelect(m.machine_id)}
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

      {meta && (
        <p className="foot mono muted">
          alert ≥ {meta.risk_bands.high} · watch ≥ {meta.risk_bands.medium} ·{" "}
          {meta.n_features} features · {meta.horizon_hours}h horizon
        </p>
      )}
    </section>
  );
}
