import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import Dashboard from "./components/Dashboard.jsx";
import MachineDetail from "./components/MachineDetail.jsx";
import AskBox from "./components/AskBox.jsx";
import TimeControl from "./components/TimeControl.jsx";

export default function App() {
  const [meta, setMeta] = useState(null);
  const [fleet, setFleet] = useState(null);
  const [asOf, setAsOf] = useState(null);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.meta().then(setMeta).catch((e) => setError(e.message));
  }, []);

  const loadFleet = useCallback((at) => {
    setLoading(true);
    api
      .fleet(at)
      .then((d) => {
        setFleet(d);
        setError(null);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    loadFleet(asOf);
  }, [asOf, loadFleet]);

  if (error && !fleet) {
    return (
      <div className="app">
        <div className="error-page">
          <h1>Can’t reach the risk service</h1>
          <p className="mono">{error}</p>
          <p>
            Start the backend with{" "}
            <code>uvicorn backend.main:app --port 8000</code>, then reload.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>Plant Failure Risk</h1>
          <p className="sub">
            Chance each machine fails in the next{" "}
            {meta ? meta.horizon_hours : 24} hours
          </p>
        </div>
        {meta && (
          <TimeControl
            meta={meta}
            asOf={asOf}
            onChange={(t) => {
              setAsOf(t);
              setSelected(null);
            }}
          />
        )}
      </header>

      {/* A stale as_of or a transient request failure must stay visible even
          though the last-good fleet is still on screen — otherwise the view
          silently freezes with no indication anything went wrong. */}
      {error && fleet && (
        <div className="note warn" role="alert" style={{ marginTop: 14 }}>
          Couldn’t update the view: {error}. Showing the last data that
          loaded successfully ({fleet.as_of}).
        </div>
      )}

      {meta && <AskBox />}

      <main className="layout">
        <Dashboard
          fleet={fleet}
          loading={loading}
          meta={meta}
          selected={selected}
          onSelect={setSelected}
        />
        {selected ? (
          <MachineDetail machineId={selected} asOf={asOf} meta={meta} />
        ) : (
          <section className="panel placeholder">
            <p>Select a machine to see its sensor history, risk drivers and trend.</p>
          </section>
        )}
      </main>
    </div>
  );
}
