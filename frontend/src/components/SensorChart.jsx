import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

// Temperature and vibration are different units on different scales, so they
// get their own charts rather than a dual y-axis — a dual axis lets you imply
// any correlation you like by sliding one scale against the other.
//
// Maintenance resets are marked because Phase 0 found 77% of them follow the
// breakdown they appear to precede; seeing the reset land on top of the
// failure is the clearest way to convey that.

const SERIES = [
  { key: "vibration_mm_s", label: "Vibration", unit: "mm/s", color: "var(--series-1)" },
  { key: "temperature_c", label: "Temperature", unit: "°C", color: "var(--series-2)" },
];

export default function SensorChart({ history }) {
  if (!history || history.length === 0) {
    return <p className="muted tiny">No sensor history in range.</p>;
  }

  const data = history.map((h) => ({
    label: h.timestamp.slice(5, 16),
    vibration_mm_s: h.vibration_mm_s,
    temperature_c: h.temperature_c,
  }));
  const resets = history.filter((h) => h.maintenance_reset);
  const failures = history.filter((h) => h.failure_event);

  const axis = { fontSize: 10, fill: "var(--ink-muted)", fontFamily: "var(--mono)" };

  return (
    <div className="chart">
      {SERIES.map((s) => (
        <div key={s.key}>
          <p className="chart-label">
            {s.label} <span className="muted">({s.unit})</span>
          </p>
          <ResponsiveContainer width="100%" height={132}>
            <LineChart data={data} margin={{ top: 4, right: 14, bottom: 2, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="label" tick={axis} minTickGap={58} tickLine={false}
                     stroke="var(--rule-strong)" />
              <YAxis tick={axis} width={40} tickLine={false}
                     stroke="var(--rule-strong)" domain={["auto", "auto"]} />
              <Tooltip
                contentStyle={{
                  fontSize: 12, fontFamily: "var(--mono)",
                  background: "var(--surface)", border: "1px solid var(--rule-strong)",
                  borderRadius: 4, color: "var(--ink)",
                }}
              />
              {resets.map((r) => (
                <ReferenceLine
                  key={`reset-${r.timestamp}`}
                  x={r.timestamp.slice(5, 16)}
                  stroke="var(--ink-muted)"
                  strokeDasharray="4 3"
                />
              ))}
              {failures.map((f) => (
                <ReferenceLine
                  key={`fail-${f.timestamp}`}
                  x={f.timestamp.slice(5, 16)}
                  stroke="var(--sev-high)"
                  strokeWidth={2}
                />
              ))}
              <Line
                type="monotone"
                dataKey={s.key}
                stroke={s.color}
                strokeWidth={1.6}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ))}

      <div className="chart-legend">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: "var(--series-1)" }} />
          Vibration
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: "var(--series-2)" }} />
          Temperature
        </span>
        <span className="legend-item">
          <span className="legend-swatch dashed" />
          Maintenance reset
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: "var(--sev-high)", height: 11, width: 2 }} />
          Breakdown
        </span>
      </div>
    </div>
  );
}
