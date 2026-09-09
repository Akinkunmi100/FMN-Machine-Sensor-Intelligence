"""
Grounded free-text Q&A over the sensor dataset (Groq).

Two-stage by design:
  1. RETRIEVE — parse the question deterministically (regex, no LLM) into
     filters, then pull real rows via risk_context.query(). This step cannot
     hallucinate: it either finds rows or it doesn't.
  2. ANSWER — pass ONLY those retrieved facts to the model, with instructions
     to refuse anything the evidence doesn't cover.

The retrieved evidence is returned alongside the answer so the UI can show
provenance, and so a wrong answer can be traced to retrieval vs generation.
"""

import json
import os
import re

import pandas as pd
from dotenv import load_dotenv

from llm_explain import MAX_TOKENS, REASONING_EFFORT, _clean, get_client
from risk_context import (
    RISK_BANDS,
    fleet_snapshot,
    machine_snapshot,
    query,
    score_frame,
)

load_dotenv()

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """\
You answer questions for a plant maintenance team about their machines.

You are given an EVIDENCE block containing real retrieved records from the \
plant's sensor database. Rules:
- Answer ONLY from the EVIDENCE. It is the entire world of facts available.
- If the EVIDENCE does not contain what was asked, say plainly what is missing \
and what you would need. Never fill a gap with general knowledge about \
machinery, and never guess a number.
- Cite the specific machines, readings and timestamps you used.
- Risk is the modelled chance of a failure in the next 24 hours. Express it as \
a percentage or a plain word, never as a bare decimal.
- Machines flagged cold_start have only ~3 days of history; say so when you \
report their scores.
- Be direct and brief — 1-4 sentences unless a list is genuinely clearer. No \
preamble, no restating the question.
"""

MACHINE_RE = re.compile(r"\bMCH[-\s]?(\d{3})\b", re.I)
LINE_RE = re.compile(r"\bline\s+([abc])\b", re.I)
LAST_N_RE = re.compile(r"\blast\s+(\d+)\s*(hour|hours|day|days|week|weeks)\b", re.I)
TOP_N_RE = re.compile(r"\btop\s+(\d+)\b", re.I)

RISK_WORDS = ("risk", "risky", "riskiest", "danger", "fail", "failing",
              "worst", "attention", "worry", "concern", "alert")
FAILURE_WORDS = ("failure", "failed", "breakdown", "broke", "outage")
MAINT_WORDS = ("maintenance", "serviced", "service", "repair", "overhaul")


def parse_question(question: str) -> dict:
    """Deterministic intent extraction — no LLM, so it cannot invent filters."""
    q = question.lower()
    machines = ["MCH-" + m for m in MACHINE_RE.findall(question)]

    line = None
    line_match = LINE_RE.search(question)
    if line_match:
        line = "Line " + line_match.group(1).upper()

    last_hours = None
    span = LAST_N_RE.search(question)
    if span:
        n, unit = int(span.group(1)), span.group(2).lower()
        last_hours = n * {"hour": 1, "hours": 1, "day": 24, "days": 24,
                          "week": 168, "weeks": 168}[unit]

    top_match = TOP_N_RE.search(question)
    top_n = int(top_match.group(1)) if top_match else None

    return {
        "machines": machines,
        "line": line,
        "last_hours": last_hours,
        "top_n": top_n,
        "wants_risk": any(w in q for w in RISK_WORDS),
        "wants_failures": any(w in q for w in FAILURE_WORDS),
        "wants_maintenance": any(w in q for w in MAINT_WORDS),
    }


def retrieve(scored: pd.DataFrame, question: str) -> dict:
    """Pull the real records that could answer this question."""
    intent = parse_question(question)
    evidence = {"question_understood_as": intent}

    fleet = fleet_snapshot(scored)
    if intent["line"]:
        fleet = [m for m in fleet if m["line"] == intent["line"]]
    evidence["fleet_latest_ranked_by_risk"] = fleet[: (intent["top_n"] or len(fleet))]
    evidence["as_of"] = str(scored["timestamp"].max())
    evidence["risk_bands"] = RISK_BANDS

    # Per-machine detail when specific machines are named.
    if intent["machines"]:
        details = []
        for mid in intent["machines"]:
            try:
                details.append(machine_snapshot(scored, mid))
            except ValueError:
                details.append({"machine_id": mid, "error": "no such machine in dataset"})
        evidence["machine_detail"] = details

    # Recent readings when a time span was named.
    if intent["last_hours"]:
        rows = query(scored, machines=intent["machines"] or None,
                     line=intent["line"], last_hours=intent["last_hours"], limit=200)
        evidence["recent_readings_sample"] = [
            {
                "machine_id": r["machine_id"], "timestamp": str(r["timestamp"]),
                "temperature_c": round(float(r["temperature_c"]), 2),
                "vibration_mm_s": round(float(r["vibration_mm_s"]), 3),
                "risk": round(float(r["risk"]), 4),
            }
            for _, r in rows.tail(40).iterrows()
        ]

    # Real failure history when asked about failures.
    if intent["wants_failures"]:
        fails = scored[scored["failure_event"] == 1]
        if intent["machines"]:
            fails = fails[fails["machine_id"].isin(intent["machines"])]
        if intent["line"]:
            fails = fails[fails["line"] == intent["line"]]
        evidence["recorded_failure_events"] = [
            {"machine_id": r["machine_id"], "line": r["line"],
             "timestamp": str(r["timestamp"])}
            for _, r in fails.sort_values("timestamp").iterrows()
        ]

    # Real maintenance resets when asked about maintenance.
    if intent["wants_maintenance"]:
        resets = scored[scored["run_hours_since_maintenance"] == 0]
        if intent["machines"]:
            resets = resets[resets["machine_id"].isin(intent["machines"])]
        if intent["line"]:
            resets = resets[resets["line"] == intent["line"]]
        evidence["maintenance_resets"] = [
            {"machine_id": r["machine_id"], "timestamp": str(r["timestamp"])}
            for _, r in resets.sort_values("timestamp").iterrows()
        ][-40:]

    return evidence


def answer_question(question: str, scored: pd.DataFrame = None, client=None,
                    model: str = None, temperature: float = 0.2) -> dict:
    """Retrieve real records, then answer strictly from them."""
    scored = score_frame() if scored is None else scored
    client = client or get_client()
    model = model or DEFAULT_MODEL

    evidence = retrieve(scored, question)
    payload = json.dumps(evidence, indent=2, default=str)

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"EVIDENCE:\n{payload}\n\nQUESTION: {question}"},
        ],
    )
    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        raise RuntimeError(
            f"Groq returned empty content (finish_reason={choice.finish_reason})."
        )
    return {
        "question": question,
        "answer": _clean(text),
        "evidence": evidence,
        "model": model,
    }


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    scored = score_frame()
    questions = [
        "Which machines should I look at first?",
        "How has MCH-213 been behaving, and has it ever failed?",
        "Has anything on Line C failed, and when was it last serviced?",
        "What is the vibration on MCH-301 and can I trust its score?",
        "Which machine has the highest oil pressure?",
    ]
    for q in questions:
        result = answer_question(q, scored=scored)
        keys = list(result["evidence"].keys())
        print("=" * 78)
        print("Q:", q)
        print("-" * 78)
        print(result["answer"])
        print(f"\n   [retrieved: {', '.join(k for k in keys if k != 'question_understood_as')}]")
        print()
