// Dataset timestamps are recorded without a timezone. Keep them in that
// recorded plant time when presenting them; letting the browser shift them to
// the visitor's local timezone makes historical events look different for
// different stakeholders.
export function formatTimestamp(value) {
  if (!value) return "—";
  const [date, time = ""] = String(value).split(" ");
  const [year, month, day] = date.split("-");
  const [hour = "00", minute = "00"] = time.split(":");
  const monthName = new Date(Date.UTC(Number(year), Number(month) - 1, 1))
    .toLocaleString("en", { month: "short", timeZone: "UTC" });
  return `${day} ${monthName} ${year}, ${hour}:${minute}`;
}

export function formatThreshold(value) {
  return `${(Number(value) * 100).toFixed(0)}%`;
}
