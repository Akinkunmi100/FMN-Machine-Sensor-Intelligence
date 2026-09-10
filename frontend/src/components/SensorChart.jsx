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

// Temperature and vibration are on different scales and different units, so
// they get their own charts rather than a dual y-axis — a dual axis lets you
// imply any correlation you like by sliding one scale against the other.
// Maintenance resets are marked, since Phase 0 found 77% of them follow the
// failure they appear to precede.

export default function SensorChart({ history }) {
  if (!history || history.length === 0) {
    return <p className="muted">No sensor history in range.</p>;
  }

  const data = history.map((h) => ({
    t: h.timestamp.slice(5, 16),
    temperature_c: h.temperature_c,
    vibration_mm_s: h.vibration_mm_s,
  }));
  const resets = history.filter((h) => h.maintenance_reset);
  const failures = history.filter((h) => h.failure_event);

  const series = [
    { key: "vibration_mm_s", label: "Vibration (mm/s)", color: "var(--series-1)" },
    { key: "temperature_c", label: "Temperature (°C)", color: "var(--series-2)" },
  ];

  return (
    <div className="chart">
      {series.map((s) => (
        <div key={s.key}>
          <p className="tiny muted">{s.label}</p>
          <ResponsiveContainer width="100%" height={130}>
            <LineChart data={data} margin={{ top: 4, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis
                dataKey="t"
                tick={{ fontSize: 10, fill: "var(--ink-muted)" }}
                minTickGap={56}
              />
              <YAxis
                tick={{ fontSize: 10, fill: "var(--ink-muted)" }}
                width={38}
                domain={["auto", "auto"]}
              />
              <Tooltip contentStyle={{ fontSize: 12 }} />
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
                  stroke="var(--status-critical)"
                  strokeWidth={2}
                />
              ))}
              <Line
                type="monotone"
                dataKey={s.key}
                stroke={s.color}
                strokeWidth={1.8}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ))}
      <p className="tiny muted">
        Dashed line = maintenance reset · solid red = recorded breakdown
      </p>
    </div>
  );
}
