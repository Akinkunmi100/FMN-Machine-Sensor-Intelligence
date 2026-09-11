import { useState } from "react";
import { api } from "../api.js";

const EXAMPLES = [
  "Which machines are high risk right now?",
  "Has MCH-207 ever failed and when was it serviced?",
  "What happened on April 8th?",
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
    <section className="panel ask" aria-labelledby="ask-heading">
      <h2 id="ask-heading">Ask about the plant</h2>
      <p className="section-caption">
        Ask in plain language about machines, lines, breakdowns, or maintenance.
        Answers are based on the records in this dashboard.
      </p>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask();
        }}
      >
        <input
          id="plant-question"
          aria-label="Ask a question about the plant"
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
            <button key={e} className="chip-btn" type="button" onClick={() => ask(e)}>
              {e}
            </button>
          ))}
        </div>
      )}

      {error && <p className="error" role="alert">We couldn’t answer that right now. Please try again.</p>}

      {result && (
        <div className="answer" aria-live="polite">
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
