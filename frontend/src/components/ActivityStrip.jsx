// Always visible on load — the whole reason this exists is that the current
// snapshot can be entirely calm (every machine LOW right now) while the
// system has genuinely caught real breakdowns in its history. Without this,
// a first-time visitor sees an all-green table and has no way to tell the
// tool does anything at all.
//
import { formatTimestamp } from "../format.js";

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
    <section className="activity" aria-labelledby="activity-heading">
      <h2 id="activity-heading" className="sr-only">What the model has demonstrated</h2>
      <div className="activity-cell">
        <span className="activity-v">{a.total_recorded_failures}</span>
        <span className="activity-l">Recorded breakdowns</span>
        {recent && (
          <span className="activity-sub">
            {recent.machine_id} was most recently recorded on {formatTimestamp(recent.timestamp)}
          </span>
        )}
      </div>

      <div className="activity-cell accent">
        <span className="activity-v">
          {a.caught_in_walk_forward}/{a.evaluable_in_walk_forward}
        </span>
        <span className="activity-l">Caught in honest testing</span>
        <span className="activity-sub">
          Breakdown windows where the model gave an alert using only earlier data
          {uncovered > 0 &&
            ` — ${uncovered} earlier breakdown${uncovered === 1 ? "" : "s"} did not have enough earlier history to evaluate`}
        </span>
      </div>

      <div className="activity-cell">
        <span className="activity-v">
          {a.median_lead_hours != null ? `${a.median_lead_hours}h` : "—"}
        </span>
        <span className="activity-l">Typical warning time</span>
        <span className="activity-sub">
          {a.min_lead_hours != null
            ? `The shortest warning was ${a.min_lead_hours}h`
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
            Review the last breakdown →
          </button>
          <span className="activity-sub">
            View the fleet as it looked at {formatTimestamp(recent.timestamp)}
          </span>
        </div>
      )}
    </section>
  );
}
