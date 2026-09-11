import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import Dashboard from "./components/Dashboard.jsx";
import MachineDetail from "./components/MachineDetail.jsx";
import AskBox from "./components/AskBox.jsx";
import TimeControl from "./components/TimeControl.jsx";
import ActivityStrip from "./components/ActivityStrip.jsx";
import ThemeToggle from "./components/ThemeToggle.jsx";
import { formatTimestamp } from "./format.js";

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
          <h1>Plant data is unavailable</h1>
          <p>We couldn’t connect to the risk service, so the dashboard cannot load yet.</p>
          <p>
            Try refreshing the page. If the problem continues, ask the technical
            team to check the risk service.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none">
              <path
                d="M12 2L3 7v6c0 5 4 8.5 9 9 5-.5 9-4 9-9V7l-9-5z"
                stroke="white" strokeWidth="1.6" strokeLinejoin="round"
              />
              <path
                d="M8.5 12.5l2 2 5-5"
                stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"
              />
            </svg>
          </span>
          <div>
            <h1>Plant Failure Risk</h1>
            <p className="sub">
              Chance each machine fails in the next{" "}
              {meta ? meta.horizon_hours : 24} hours
            </p>
          </div>
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
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
          <ThemeToggle />
        </div>
      </header>

      {/* A stale as_of or a transient request failure must stay visible even
          though the last-good fleet is still on screen — otherwise the view
          silently freezes with no indication anything went wrong. */}
      {error && fleet && (
        <div className="note warn" role="alert" style={{ marginTop: 14 }}>
          We couldn’t update the view. Showing the last data that loaded
          successfully ({formatTimestamp(fleet.as_of)}). Try again shortly.
        </div>
      )}

      {meta && (
        <ActivityStrip
          meta={meta}
          onJumpTo={(timestamp) => {
            setAsOf(timestamp);
            setSelected(null);
          }}
        />
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
