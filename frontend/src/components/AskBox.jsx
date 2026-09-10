import { useState } from "react";
import { api } from "../api.js";

const EXAMPLES = [
  "Which machines need attention right now?",
  "Has MCH-207 ever failed and when was it serviced?",
  "What has been happening on Line C?",
];

export default function AskBox() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [showEvidence, setShowEvidence] = useState(false);

  const ask = (q) => {
    const text = (q ?? question).trim();
    if (!text) return;
    setQuestion(text);
    setBusy(true);
    setError(null);
    setShowEvidence(false);
    api
      .ask(text)
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setBusy(false));
  };

  return (
    <section className="panel ask">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask();
        }}
      >
        <input
          type="text"
          value={question}
          placeholder="Ask about any machine, line, failure or maintenance record…"
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button className="primary" type="submit" disabled={busy}>
          {busy ? "Looking…" : "Ask"}
        </button>
      </form>

      {!result && !busy && (
        <div className="examples">
          {EXAMPLES.map((e) => (
            <button key={e} className="chip-btn" onClick={() => ask(e)}>
              {e}
            </button>
          ))}
        </div>
      )}

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="answer">
          <p className="answer-text">{result.answer}</p>
          <button className="linkish" onClick={() => setShowEvidence((v) => !v)}>
            {showEvidence ? "Hide" : "Show"} the records this used
          </button>
          {showEvidence && (
            <pre className="evidence">
              {JSON.stringify(result.evidence, null, 2)}
            </pre>
          )}
        </div>
      )}
    </section>
  );
}
