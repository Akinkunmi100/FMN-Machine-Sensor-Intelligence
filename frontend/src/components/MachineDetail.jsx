import { useEffect, useState } from "react";
import { api } from "../api.js";
import TrendChart from "./TrendChart.jsx";
import SensorChart from "./SensorChart.jsx";
import { formatTimestamp } from "../format.js";

export default function MachineDetail({ machineId, asOf, meta }) {
  const [snap, setSnap] = useState(null);
  const [trend, setTrend] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [explaining, setExplaining] = useState(false);
  const [error, setError] = useState(null);
  const [showEvidence, setShowEvidence] = useState(false);
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    setSnap(null);
    setExplanation(null);
    setError(null);
    setShowEvidence(false);
    api.machine(machineId, asOf).then(setSnap).catch((e) => setError(e.message));
    api.trend(machineId).then(setTrend).catch(() => setTrend(null));
  }, [machineId, asOf, retry]);

  if (error) {
    return (
      <section className="panel" role="alert">
        <p className="error">We couldn’t load this machine’s details.</p>
        <button className="primary" type="button" onClick={() => setRetry((n) => n + 1)}>
          Try again
        </button>
      </section>
    );
  }
  if (!snap) {
    return (
      <section className="panel">
        <div className="panel-head"><h2>{machineId}</h2></div>
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="skeleton" style={{ width: `${95 - i * 6}%` }} />
        ))}
      </section>
    );
  }

  const band = snap.risk_band.toLowerCase();

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
          <span className="mono">{snap.machine_id}</span>
          <span className={`pill band-${band}`}>{snap.risk_band}</span>
        </h2>
        <span className="mono muted tiny">{formatTimestamp(snap.as_of)}</span>
      </div>

      <div className={`risk-hero band-${band}`}>
        <span className="risk-hero-n">{(snap.risk * 100).toFixed(1)}%</span>
        <span className="risk-hero-l">
          chance of failure in the next {meta ? meta.horizon_hours : 24} hours
          <br />
          <span className="muted tiny">
            {snap.line} · {snap.hours_of_history.toLocaleString()} hours of recorded history
          </span>
        </span>
      </div>

      {snap.cold_start && (
        <p className="note warn">
          This machine has only {snap.hours_of_history} hours of recorded history
          and no recorded failures. Its score is a real estimate, but it has not
          yet been validated on this machine, so treat it with extra caution.
        </p>
      )}

      <div className="stat-row">
        <Stat label="Risk 24 hours ago" value={fmtPct(snap.risk_24h_ago)} />
        <Stat label="Risk 7 days ago" value={fmtPct(snap.risk_7d_ago)} />
        <Stat
          label="Hours since maintenance"
          value={`${snap.current_readings.run_hours_since_maintenance} h`}
        />
        <Stat label="Recorded failures" value={snap.prior_failures_on_record} />
      </div>

      <h3>Why this risk?</h3>
      <p className="section-caption">
        The explanation compares this machine’s readings with its own usual behaviour.
      </p>
      <button className="primary" onClick={askForExplanation} disabled={explaining}>
        {explaining ? "Preparing explanation…" : explanation ? "Explain again" : "Explain in plain language"}
      </button>

      {explanation && (
        <div className="explanation">
          <p>{explanation.explanation}</p>
          <p className="mono muted tiny">
            Based on this machine’s readings
            {explanation.cached ? " · reused for this same hour" : " · prepared just now"}
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

      <h3>What is driving the risk?</h3>
      <p className="section-caption">
        “Compared with usual” shows how far each reading is from this machine’s normal level.
      </p>
      <div className="table-scroll">
      <table className="drivers">
        <caption className="sr-only">Current readings compared with this machine’s usual levels.</caption>
        <thead>
          <tr>
            <th>Measure</th>
            <th className="num">Current</th>
            <th className="num">Usual level</th>
            <th className="num">Compared with usual</th>
            <th className="num">How unusual</th>
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
                  <span className={ratioClass(d.times_own_normal)}>
                    {d.times_own_normal}×
                  </span>
                ) : (
                  <span className="muted">—</span>
                )}
              </td>
              <td className="num mono muted">{d.percentile_vs_own_history}</td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>
      <p className="tiny muted">
        “How unusual” is the current reading’s position in this machine’s own history.
        A value of 1.0× compared with usual means business as usual.
      </p>

      <h3>Risk over time</h3>
      <p className="section-caption">
        How the model’s confidence in a breakdown moved week by week —
        scored honestly, using only what was knowable at each point in time,
        not with the benefit of hindsight.
      </p>
      {trend ? (
        <TrendChart trend={trend} meta={meta} />
      ) : (
        <p className="muted tiny">No trend data for this machine.</p>
      )}

      <h3>Sensor history</h3>
      <p className="section-caption">
        Raw temperature and vibration readings for this machine, with
        maintenance visits (dashed) and recorded breakdowns (solid red)
        marked so you can see how sensor behaviour changed around each one.
      </p>
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

// Thresholds match the severity language used everywhere else: 2x its own
// normal reads as a problem, 1.5x as worth watching.
const ratioClass = (r) => (r >= 2 ? "ratio-hot" : r >= 1.5 ? "ratio-warm" : "");
