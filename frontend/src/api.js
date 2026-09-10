// Thin fetch layer. Every call hits the FastAPI backend; nothing is computed
// or mocked here, so a blank panel always means a real backend problem rather
// than a frontend fallback quietly hiding one.

async function get(path) {
  const res = await fetch(path);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

const q = (asOf) => (asOf ? `?as_of=${encodeURIComponent(asOf)}` : "");

export const api = {
  meta: () => get("/api/meta"),
  fleet: (asOf) => get(`/api/fleet${q(asOf)}`),
  machine: (id, asOf) => get(`/api/machines/${id}${q(asOf)}`),
  trend: (id) => get(`/api/machines/${id}/trend`),
  explain: (id, asOf) => post(`/api/machines/${id}/explain${q(asOf)}`),
  ask: (question) => post("/api/qa", { question }),
};
