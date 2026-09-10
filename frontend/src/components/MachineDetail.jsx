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
        <span className="mono muted tiny">{snap.as_of}</span>
      </div>

      <div className={`risk-hero band-${band}`}>
        <span className="risk-hero-n">{(snap.risk * 100).toFixed(1)}%</span>
        <span className="risk-hero-l">
          chance of failure in the next {meta ? meta.horizon_hours : 24} hours
          <br />
          <span className="muted tiny">
            {snap.line} · {snap.hours_of_history.toLocaleString()} hours of history
          </span>
        </span>
      </div>

      {snap.cold_start && (
        <p className="note warn">
          Only {snap.hours_of_history} hours of history and no recorded failures.
          This is a real prediction, but it has never been validated for this
          machine — treat it with lower confidence than the rest of the fleet.
        </p>
      )}

      <div className="stat-row">
        <Stat label="24h ago" value={fmtPct(snap.risk_24h_ago)} />
        <Stat label="7d ago" value={fmtPct(snap.risk_7d_ago)} />
        <Stat
          label="Since maintenance"
          value={`${snap.current_readings.run_hours_since_maintenance} h`}
        />
        <Stat label="Failures on record" value={snap.prior_failures_on_record} />
      </div>

      <h3>Why — read from this machine’s own numbers</h3>
      <button className="primary" onClick={askForExplanation} disabled={explaining}>
        {explaining ? "Asking…" : explanation ? "Ask again" : "Explain this risk"}
      </button>

      {explanation && (
        <div className="explanation">
          <p>{explanation.explanation}</p>
          <p className="mono muted tiny">
            {explanation.model}
            {explanation.cached ? " · cached for this hour" : " · generated just now"}
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
            <th className="num">Pctile</th>
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
      <p className="tiny muted">
        “vs normal” is the reading divided by this machine’s own running median —
        the same ratio the risk model keys on. 1.0× is business as usual.
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
