import {
  Area,
  CartesianGrid,
  ComposedChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

// This chart shows WALK-FORWARD OUT-OF-SAMPLE risk: each week scored by a
// model trained only on earlier data. It deliberately does not use the shipped
// model, which was refit on all history and would grade its own homework here.
// Rows before the first trainable window carry risk=null so the gap renders as
// a gap — a flat line there would imply the model was quiet when in fact it
// did not exist yet.

export default function TrendChart({ trend, meta }) {
  const points = trend.points.map((p) => ({
    t: p.timestamp.slice(5, 16),
    risk: p.risk,
    status: p.status,
  }));
  const scored = points.filter((p) => p.risk != null);
  const gapCount = points.length - scored.length;
  const threshold = trend.alert_threshold ?? meta?.risk_bands?.high ?? 0.2;

  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={points} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis
            dataKey="t"
            tick={{ fontSize: 11, fill: "var(--ink-muted)" }}
            minTickGap={48}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 11, fill: "var(--ink-muted)" }}
            width={38}
            tickFormatter={(v) => `${Math.round(v * 100)}%`}
          />
          <Tooltip
            formatter={(v) => [`${(v * 100).toFixed(1)}%`, "risk"]}
            contentStyle={{ fontSize: 12 }}
          />
          <ReferenceLine
            y={threshold}
            stroke="var(--status-good)"
            strokeWidth={2}
            label={{ value: "alert", fontSize: 11, position: "right" }}
          />
          <Area
            type="monotone"
            dataKey="risk"
            stroke="var(--series-1)"
            fill="var(--series-1)"
            fillOpacity={0.15}
            strokeWidth={2}
            connectNulls={false}
            dot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>

      <p className="tiny muted">
        Out-of-sample: each week scored by a model trained only on earlier data.
        {gapCount > 0 && (
          <>
            {" "}
            {gapCount.toLocaleString()} earlier hours are blank because their
            training window held fewer than two recorded failures — the model
            could not exist yet.
          </>
        )}{" "}
        Not comparable to the current-state score above, which uses the model
        refit on all history.
      </p>

      {trend.failures.length > 0 && (
        <p className="tiny muted">
          Recorded breakdowns: {trend.failures.map((f) => f.slice(0, 16)).join(" · ")}
        </p>
      )}
    </div>
  );
}
