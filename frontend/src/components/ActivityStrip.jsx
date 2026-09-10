// Always visible on load — the whole reason this exists is that the current
// snapshot can be entirely calm (every machine LOW right now) while the
// system has genuinely caught real breakdowns in its history. Without this,
// a first-time visitor sees an all-green table and has no way to tell the
// tool does anything at all.
//
// Every number here comes from backend/main.py's /api/meta, which computes
// it live at startup from the committed walk-forward artifact (see
// risk_context.fleet_activity_summary) — nothing in this component is a
// static or invented figure.

export default function ActivityStrip({ meta, onJumpTo }) {
  const a = meta.activity;
  if (!a) return null;

  const recent = a.most_recent_failure;
  const uncovered = a.total_recorded_failures - a.evaluable_in_walk_forward;

  return (
    <section className="activity" aria-label="Historical model performance">
      <div className="activity-cell">
        <span className="activity-v">{a.total_recorded_failures}</span>
        <span className="activity-l">RECORDED BREAKDOWNS ON FILE</span>
        {recent && (
          <span className="activity-sub">
            Most recent: {recent.machine_id} · {recent.timestamp.slice(0, 16)}
          </span>
        )}
      </div>

      <div className="activity-cell accent">
        <span className="activity-v">
          {a.caught_in_walk_forward}/{a.evaluable_in_walk_forward}
        </span>
        <span className="activity-l">CAUGHT, EVALUATED HONESTLY</span>
        <span className="activity-sub">
          Scored only by what the model could have known at the time
          {uncovered > 0 &&
            ` — ${uncovered} earlier breakdown${uncovered === 1 ? "" : "s"} predate enough history to score`}
        </span>
      </div>

      <div className="activity-cell">
        <span className="activity-v">
          {a.median_lead_hours != null ? `${a.median_lead_hours}h` : "—"}
        </span>
        <span className="activity-l">MEDIAN WARNING TIME</span>
        <span className="activity-sub">
          {a.min_lead_hours != null
            ? `As little as ${a.min_lead_hours}h in the tightest case`
            : "Before the breakdown occurred"}
        </span>
      </div>

      {recent && (
        <div className="activity-cell">
          <button
            className="linkish"
            style={{ fontSize: "0.82rem" }}
            onClick={() => onJumpTo(recent.timestamp)}
          >
            View the fleet at the last breakdown →
          </button>
          <span className="activity-sub">
            {recent.machine_id} on {recent.line}, {recent.timestamp.slice(0, 16)}
          </span>
        </div>
      )}
    </section>
  );
}
