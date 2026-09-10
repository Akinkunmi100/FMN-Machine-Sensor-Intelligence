// A compact, real history strip for one machine's fleet-table row. The point
// is specifically to show risk that happened even when the CURRENT value is
// 0 — a machine can look perfectly calm right now and still have had a real
// spike three weeks ago that a single latest-value number would hide.

const W = 84;
const H = 26;
const THRESHOLD = 0.2; // kept in sync with the backend's HIGH band by the caller

export default function Sparkline({ points, threshold = THRESHOLD }) {
  if (!points || points.length < 2) {
    return <span className="tiny muted">—</span>;
  }

  const max = Math.max(1, ...points.map((p) => p.risk));
  const n = points.length;
  const x = (i) => (i / (n - 1)) * (W - 2) + 1;
  const y = (v) => H - 3 - (v / max) * (H - 6);

  const path = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(p.risk).toFixed(1)}`)
    .join(" ");

  const peak = points.reduce((best, p, i) => (p.risk > points[best].risk ? i : best), 0);
  const hadAlert = points.some((p) => p.risk >= threshold);

  return (
    <svg
      className="sparkline"
      width={W}
      height={H}
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={
        hadAlert
          ? `Risk history, peaked at ${(points[peak].risk * 100).toFixed(0)}% on ${points[peak].day}`
          : "Risk history, stayed low throughout"
      }
    >
      <path
        d={path}
        fill="none"
        stroke={hadAlert ? "var(--sev-high)" : "var(--ink-muted)"}
        strokeWidth="1.4"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      {hadAlert && (
        <circle cx={x(peak)} cy={y(points[peak].risk)} r="1.8" fill="var(--sev-high)" />
      )}
    </svg>
  );
}
