const PCT = (v) => `${(v * 100).toFixed(v >= 0.1 ? 0 : 1)}%`;

export default function Dashboard({ fleet, loading, meta, selected, onSelect }) {
  if (loading && !fleet) {
    return <section className="panel"><p>Loading fleet…</p></section>;
  }
  if (!fleet) return null;

  const { counts, machines, as_of } = fleet;

  return (
    <section className="panel fleet">
      <div className="panel-head">
        <h2>Fleet</h2>
        <span className="mono muted">{as_of}</span>
      </div>

      <div className="counts">
        {["HIGH", "MEDIUM", "LOW"].map((band) => (
          <div key={band} className={`count band-${band.toLowerCase()}`}>
            <span className="count-n">{counts[band]}</span>
            <span className="count-l">{band}</span>
          </div>
        ))}
      </div>

      {counts.HIGH === 0 && counts.MEDIUM === 0 && (
        <p className="note">
          Every machine is in the low band at this moment. That is the real state
          of the data here, not an empty view — use “jump to a recorded
          breakdown” above to see the model during an actual event.
        </p>
      )}

      <table className="fleet-table">
        <thead>
          <tr>
            <th>Machine</th>
            <th>Line</th>
            <th className="num">Risk</th>
            <th>Band</th>
            <th className="num">Vibration</th>
            <th className="num">Temp</th>
          </tr>
        </thead>
        <tbody>
          {machines.map((m) => (
            <tr
              key={m.machine_id}
              onClick={() => onSelect(m.machine_id)}
              className={[
                "row",
                `band-${m.risk_band.toLowerCase()}`,
                selected === m.machine_id ? "is-selected" : "",
              ].join(" ")}
            >
              <td>
                {m.machine_id}
                {m.cold_start && (
                  <span
                    className="chip"
                    title="Only ~3 days of history — score is unvalidated"
                  >
                    new
                  </span>
                )}
              </td>
              <td className="muted">{m.line.replace("Line ", "")}</td>
              <td className="num mono">{PCT(m.risk)}</td>
              <td>
                <span className={`pill band-${m.risk_band.toLowerCase()}`}>
                  {m.risk_band}
                </span>
              </td>
              <td className="num mono">{m.vibration_mm_s.toFixed(2)}</td>
              <td className="num mono">{m.temperature_c.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {meta && (
        <p className="foot mono muted">
          Alert at {meta.risk_bands.high} · watch at {meta.risk_bands.medium} ·{" "}
          {meta.n_features} features
        </p>
      )}
    </section>
  );
}
