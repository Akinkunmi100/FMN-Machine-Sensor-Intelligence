import { useEffect, useState } from "react";
import { api } from "../api.js";
import TrendChart from "./TrendChart.jsx";
import SensorChart from "./SensorChart.jsx";

export default function MachineDetail({ machineId, asOf, meta }) {
  const [snap, setSnap] = useState(null);
  const [trend, setTrend] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [explaining, setExplaining] = useState(false);
  const [error, setError] = useState(null);
  const [showEvidence, setShowEvidence] = useState(false);

  useEffect(() => {
    setSnap(null);
    setExplanation(null);
    setError(null);
    setShowEvidence(false);
    api.machine(machineId, asOf).then(setSnap).catch((e) => setError(e.message));
    api.trend(machineId).then(setTrend).catch(() => setTrend(null));
  }, [machineId, asOf]);

  if (error) {
    return (
      <section className="panel">
        <p className="error">{error}</p>
      </section>
    );
  }
  if (!snap) {
    return <section className="panel"><p>Loading {machineId}…</p></section>;
  }

  const askForExplanation = () => {
    setExplaining(true);
    setError(null);
    api
      .explain(machineId, asOf)
      .then(setExplanation)
      .catch((e) => setError(e.message))
      .finally(() => setExplaining(false));
  };

  return (
    <section className="panel detail">
      <div className="panel-head">
        <h2>
          {snap.machine_id}
          <span className={`pill band-${snap.risk_band.toLowerCase()}`}>
            {snap.risk_band}
          </span>
        </h2>
        <span className="mono muted">{snap.as_of}</span>
      </div>

      {snap.cold_start && (
        <p className="note warn">
          Only {snap.hours_of_history} hours of history and no recorded failures.
          This score is a real prediction but it has never been validated for
          this machine — treat it with lower confidence.
        </p>
      )}

      <div className="stat-row">
        <Stat label="Risk (next 24h)" value={`${(snap.risk * 100).toFixed(1)}%`} />
        <Stat label="24h ago" value={fmtPct(snap.risk_24h_ago)} />
        <Stat label="7d ago" value={fmtPct(snap.risk_7d_ago)} />
        <Stat
          label="Since maintenance"
          value={`${snap.current_readings.run_hours_since_maintenance} h`}
        />
      </div>

      <h3>Why — from this machine’s own numbers</h3>
      <button
        className="primary"
        onClick={askForExplanation}
        disabled={explaining}
      >
        {explaining ? "Asking…" : explanation ? "Ask again" : "Explain this risk"}
      </button>

      {explanation && (
        <div className="explanation">
          <p>{explanation.explanation}</p>
          <p className="mono muted tiny">
            {explanation.model}
            {explanation.cached ? " · cached for this hour" : " · generated now"}
          </p>
          <button className="linkish" onClick={() => setShowEvidence((v) => !v)}>
            {showEvidence ? "Hide" : "Show"} the numbers it was given
          </button>
          {showEvidence && (
            <pre className="evidence">
              {JSON.stringify(explanation.evidence_sent, null, 2)}
            </pre>
          )}
        </div>
      )}

      <h3>Risk drivers</h3>
      <table className="drivers">
        <thead>
          <tr>
            <th>Reading</th>
            <th className="num">Now</th>
            <th className="num">Its normal</th>
            <th className="num">vs normal</th>
            <th className="num">Percentile</th>
          </tr>
        </thead>
        <tbody>
          {snap.drivers.map((d) => (
            <tr key={d.feature}>
              <td>{d.label}</td>
              <td className="num mono">
                {d.value} <span className="muted">{d.unit}</span>
              </td>
              <td className="num mono muted">{d.machine_median}</td>
              <td className="num mono">
                {d.times_own_normal != null ? (
                  <span className={d.times_own_normal >= 2 ? "hot" : ""}>
                    {d.times_own_normal}×
                  </span>
                ) : (
                  <span className="muted">—</span>
                )}
              </td>
              <td className="num mono">{d.percentile_vs_own_history}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Risk over time</h3>
      {trend ? <TrendChart trend={trend} meta={meta} /> : <p className="muted">No trend data.</p>}

      <h3>Sensor history</h3>
      <SensorChart history={snap.sensor_history} />
    </section>
  );
}

function Stat({ label, value }) {
  return (
    <div className="stat">
      <span className="stat-v mono">{value}</span>
      <span className="stat-l">{label}</span>
    </div>
  );
}

const fmtPct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
