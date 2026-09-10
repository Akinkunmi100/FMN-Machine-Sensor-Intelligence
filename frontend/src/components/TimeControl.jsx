// At the dataset's final hour every machine reads LOW — the last failure was
// 30 Apr 04:00, so nothing is in a pre-failure window. Without a way to move
// the clock, the landing view is an all-green fleet with nothing to inspect.
// The quick-picks jump to real recorded breakdowns.

export default function TimeControl({ meta, asOf, onChange }) {
  const failures = meta.recorded_failures || [];

  return (
    <div className="timecontrol">
      <label>
        <span className="label">Viewing as of</span>
        <input
          type="datetime-local"
          value={(asOf || meta.time_range.end).replace(" ", "T").slice(0, 16)}
          min={meta.time_range.start.replace(" ", "T").slice(0, 16)}
          max={meta.time_range.end.replace(" ", "T").slice(0, 16)}
          onChange={(e) => onChange(e.target.value.replace("T", " ") + ":00")}
        />
      </label>

      <label>
        <span className="label">Jump to a recorded breakdown</span>
        <select
          value=""
          onChange={(e) => e.target.value && onChange(e.target.value)}
        >
          <option value="">
            {failures.length} on record — pick one
          </option>
          {failures.map((f) => (
            <option key={`${f.machine_id}-${f.timestamp}`} value={f.timestamp}>
              {f.machine_id} · {f.timestamp.slice(0, 16)}
            </option>
          ))}
        </select>
      </label>

      {asOf && (
        <button className="linkish" onClick={() => onChange(null)}>
          Back to latest
        </button>
      )}
    </div>
  );
}
