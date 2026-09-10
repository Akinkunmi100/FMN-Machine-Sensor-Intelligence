const pct = (v) => `${(v * 100).toFixed(v >= 0.1 ? 0 : 1)}%`;

export default function Dashboard({ fleet, loading, meta, selected, onSelect }) {
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

  const { counts, machines, as_of, not_yet_reporting = [] } = fleet;
  const calm = counts.HIGH === 0 && counts.MEDIUM === 0;

  return (
    <section className="panel fleet">
      <div className="panel-head">
        <h2>Fleet</h2>
        <span className="mono muted tiny">{as_of}</span>
      </div>

      <div className="counts">
        {["HIGH", "MEDIUM", "LOW"].map((band) => (
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
          Every machine is in the low band at this moment — the last recorded
          breakdown was 30 April 04:00, so nothing is inside a warning window.
          That is the real state of the data, not an empty screen. Use{" "}
          <strong>jump to a recorded breakdown</strong> above to watch the model
          during an actual event.
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

      <table className="fleet-table">
        <thead>
          <tr>
            <th>Machine</th>
            <th>Line</th>
            <th className="num">Risk</th>
            <th>State</th>
            <th className="num">Vib</th>
            <th className="num">Temp</th>
          </tr>
        </thead>
        <tbody>
          {machines.map((m) => {
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
              </tr>
            );
          })}
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
