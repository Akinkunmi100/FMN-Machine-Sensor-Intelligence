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
You answer questions for a plant maintenance team — often non-technical \
stakeholders deciding what to act on today. Write for that reader.

You are given an EVIDENCE block containing real retrieved records from the \
plant's sensor database. Rules:
- Answer ONLY from the EVIDENCE. It is the entire world of facts available. \
If a "note" or "note_on_date" field is present, it usually means the filtered \
search came back empty — read it and say so plainly, don't ignore it and \
answer from the unfiltered fleet list instead.
- If the EVIDENCE does not contain what was asked (a sensor type that doesn't \
exist, a machine not in the fleet, a date outside the dataset), say plainly \
what is missing and what you would need. Never fill a gap with general \
knowledge about machinery, and never guess a number.
- Cite the specific machines, readings and timestamps you used — this is what \
lets someone verify the answer, so be concrete rather than general.
- When you state how many machines match something, use a count field the \
evidence gives you directly (like `fleet_count`) rather than counting list \
items yourself — miscounting a list is a real mistake language models make, \
so never do it when the exact number is already provided.
- Risk is the modelled chance of a failure in the next 24 hours. Express it as \
a percentage or a plain word (low / elevated / high), never as a bare decimal \
and never as "percentile" — that word should not appear in your answer.
- Machines flagged cold_start have only ~3 days of history; ALWAYS name them \
individually and say so in plain terms ("too new to fully trust this score \
yet") even when their risk band matches everyone else's — cold-start status \
is a reliability exception, separate from whatever the risk band says, and \
summarizing the group must never absorb them into "all machines are fine."
- Be direct and brief by default — 1-4 sentences — but when the evidence \
supports more (several relevant machines, a real pattern across readings), \
give the fuller answer rather than truncating useful detail. No preamble, no \
restating the question, no hedging filler ("it appears that", "it's worth \
noting").
- When most or all machines share the same answer (e.g. "is everything \
safe?" and they're all in the same risk band), state the total count once \
("all 17 machines are low-risk") and name ONLY the exceptions individually \
by machine ID — cold-start status, a different band, or anything else that \
sets a machine apart. Do NOT enumerate every machine ID one by one when they \
all say the same thing — that is the opposite of direct and no stakeholder \
wants to read a 17-item list to learn "everything is fine."
- Plain prose only. Never use markdown: no **bold**, no bullet points with \
* or -, no headers. This text is displayed as-is with no formatting applied, \
so markdown syntax would show up as literal asterisks and look broken. If a \
list is genuinely the clearest way to answer, write it as a short run of \
sentences instead.
"""

MACHINE_RE = re.compile(r"\bMCH[-\s]?(\d{3})\b", re.I)
LINE_RE = re.compile(r"\bline\s+([abc])\b", re.I)
LAST_N_RE = re.compile(r"\blast\s+(\d+)\s*(hour|hours|day|days|week|weeks)\b", re.I)
TOP_N_RE = re.compile(r"\btop\s+(\d+)\b", re.I)

# CLAUDE.md requires Q&A retrieval to filter by "machine, date range, or risk
# level." Machine and relative date range were covered; band and absolute
# date were not — parse_question() previously computed `wants_risk` but never
# turned it into an actual band filter, so "show me the high-risk machines"
# was answered by handing the LLM the entire 17-machine fleet and trusting it
# to filter correctly itself. That is now a real, retrieval-stage filter.
BAND_PATTERNS = [
    ("HIGH", re.compile(r"\b(high[- ]?risk|critical|in trouble|danger(ous)?)\b", re.I)),
    ("MEDIUM", re.compile(r"\b(medium[- ]?risk|watch(list)?|elevated)\b", re.I)),
    ("LOW", re.compile(r"\b(low[- ]?risk|safe|normal|healthy|fine)\b", re.I)),
]

# Absolute dates: ISO (2026-04-08), "April 8", "8 April", with or without a
# year. Ambiguous or unparseable matches are dropped rather than guessed —
# a wrong silent date filter is worse than no date filter, since the LLM is
# told the evidence is the entire world of facts and would trust it blindly.
DATE_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?(?:,?\s+\d{4})?)\b",
    re.I,
)

FAILURE_WORDS = ("failure", "failed", "breakdown", "broke", "outage")
MAINT_WORDS = ("maintenance", "serviced", "service", "repair", "overhaul")


def _parse_date(question: str, default_year: int):
    """Best-effort single-day extraction. Returns None rather than a guess
    when the match doesn't parse — silence beats a wrong filter here."""
    m = DATE_RE.search(question)
    if not m:
        return None
    text = m.group(0)
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        candidate = text
    elif re.search(r"\d{4}", text):
        candidate = text
    else:
        candidate = f"{text}, {default_year}"
    try:
        ts = pd.Timestamp(candidate)
    except (ValueError, TypeError):
        return None
    return ts.normalize()


def parse_question(question: str, default_year: int = None) -> dict:
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

    band = None
    for name, pattern in BAND_PATTERNS:
        if pattern.search(question):
            band = name
            break

    date = _parse_date(question, default_year) if default_year else None

    return {
        "machines": machines,
        "line": line,
        "last_hours": last_hours,
        "top_n": top_n,
        "band": band,
        "date": str(date.date()) if date is not None else None,
        "wants_failures": any(w in q for w in FAILURE_WORDS),
        "wants_maintenance": any(w in q for w in MAINT_WORDS),
    }


def retrieve(scored: pd.DataFrame, question: str) -> dict:
    """Pull the real records that could answer this question."""
    # Use the dataset's own year as the default for a bare "April 8" — this
    # is historical plant data, not a live feed, so "today" would be wrong.
    default_year = int(scored["timestamp"].max().year)
    intent = parse_question(question, default_year=default_year)
    evidence = {"question_understood_as": intent}

    fleet = fleet_snapshot(scored)
    if intent["line"]:
        fleet = [m for m in fleet if m["line"] == intent["line"]]
    if intent["band"]:
        # Retrieval-stage filter by risk level, not left for the LLM to sift
        # out of the full fleet itself.
        fleet = [m for m in fleet if m["risk_band"] == intent["band"]]
    evidence["fleet_latest_ranked_by_risk"] = fleet[: (intent["top_n"] or len(fleet))]
    # Precomputed, not left for the model to count off the list itself — an
    # LLM miscounting a 17-item array is a real, observed failure mode.
    evidence["fleet_count"] = len(evidence["fleet_latest_ranked_by_risk"])
    if intent["band"] and not fleet:
        evidence["note"] = f"No machines are currently in the {intent['band']} band."
    evidence["as_of"] = str(scored["timestamp"].max())
    evidence["risk_bands"] = RISK_BANDS

    # A specific calendar day was named — pull that day's readings directly,
    # rather than relying on the relative "last N hours" path.
    if intent["date"]:
        day = pd.Timestamp(intent["date"])
        rows = query(scored, machines=intent["machines"] or None,
                     line=intent["line"], start=day, end=day + pd.Timedelta(days=1),
                     limit=200)
        if rows.empty:
            evidence["readings_on_date"] = []
            evidence["note_on_date"] = (
                f"No sensor data exists for {intent['date']} "
                f"under these filters — it may be outside the dataset's range."
            )
        else:
            evidence["readings_on_date"] = [
                {
                    "machine_id": r["machine_id"], "timestamp": str(r["timestamp"]),
                    "temperature_c": round(float(r["temperature_c"]), 2),
                    "vibration_mm_s": round(float(r["vibration_mm_s"]), 3),
                    "risk": round(float(r["risk"]), 4),
                    "failure_event": bool(r["failure_event"] == 1),
                }
                for _, r in rows.tail(48).iterrows()
            ]

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
