import {
  Area,
  CartesianGrid,
  ComposedChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatThreshold } from "../format.js";

// WALK-FORWARD OUT-OF-SAMPLE risk: each week scored by a model trained only on
// earlier data. Deliberately not the shipped model, which was refit on all
// history and would grade its own homework here.
//
// Rows before the first trainable window carry risk=null and connectNulls is
// off, so the gap renders as a gap. A flat line there would imply the model
// was quiet when in fact it did not yet exist.

export default function TrendChart({ trend, meta }) {
  const points = trend.points.map((p) => ({
    t: p.timestamp,
    label: p.timestamp.slice(5, 16),
    risk: p.risk,
  }));
  const gapCount = points.filter((p) => p.risk == null).length;
  const threshold = trend.alert_threshold ?? meta?.risk_bands?.high ?? 0.2;

  const axis = { fontSize: 10.5, fill: "var(--ink-muted)", fontFamily: "var(--mono)" };

  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height={225}>
        <ComposedChart data={points} margin={{ top: 8, right: 14, bottom: 2, left: 0 }}>
          {/* Everything above the alert line is the zone that means "act".
              On a categorical x-axis Recharts drops a ReferenceArea that has
              no x bounds, so the span is given explicitly. */}
          {points.length > 0 && (
            <ReferenceArea
              x1={points[0].label}
              x2={points[points.length - 1].label}
              y1={threshold}
              y2={1}
              fill="var(--sev-high)"
              fillOpacity={0.07}
              stroke="none"
              ifOverflow="extendDomain"
            />
          )}
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis dataKey="label" tick={axis} minTickGap={52} tickLine={false}
                 stroke="var(--rule-strong)" />
          <YAxis
            domain={[0, 1]}
            tick={axis}
            width={40}
            tickLine={false}
            stroke="var(--rule-strong)"
            tickFormatter={(v) => `${Math.round(v * 100)}%`}
          />
          <Tooltip
            formatter={(v) => [`${(v * 100).toFixed(1)}%`, "risk"]}
            labelFormatter={(l) => l}
            contentStyle={{
              fontSize: 12, fontFamily: "var(--mono)",
              background: "var(--surface)", border: "1px solid var(--rule-strong)",
              borderRadius: 4, color: "var(--ink)",
            }}
          />
          <ReferenceLine
            y={threshold}
            stroke="var(--sev-high)"
            strokeWidth={1.5}
            strokeDasharray="4 3"
          />
          {trend.failures.map((f) => (
            <ReferenceLine
              key={f}
              x={f.slice(5, 16)}
              stroke="var(--sev-high)"
              strokeWidth={2}
            />
          ))}
          <Area
            type="monotone"
            dataKey="risk"
            stroke="var(--series-1)"
            fill="var(--series-1)"
            fillOpacity={0.16}
            strokeWidth={1.8}
            connectNulls={false}
            dot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="chart-legend">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: "var(--series-1)" }} />
          Historical risk estimate
        </span>
        <span className="legend-item">
          <span className="legend-swatch dashed" style={{ borderTopColor: "var(--sev-high)" }} />
          High-risk threshold ({formatThreshold(threshold)})
        </span>
        {trend.failures.length > 0 && (
          <span className="legend-item">
            <span className="legend-swatch" style={{ background: "var(--sev-high)", height: 11, width: 2 }} />
            Recorded breakdown
          </span>
        )}
      </div>

      <p className="tiny muted" style={{ marginTop: 8 }}>
        Each week is scored by a model trained only on earlier data, so this is
        what the system would genuinely have known at the time.
        {gapCount > 0 && (
          <>
            {" "}
            The first {gapCount.toLocaleString()} hours are blank because their
            training window held fewer than two recorded failures — there was no
            model yet.
          </>
        )}{" "}
        Not comparable with the headline score above, which uses the model refit
        on all history.
      </p>
    </div>
  );
}
