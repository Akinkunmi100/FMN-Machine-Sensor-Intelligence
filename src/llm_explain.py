"""
Runtime LLM explanations for a machine's failure risk (Groq).

Grounding contract: the model is given ONLY the retrieved snapshot from
risk_context.machine_snapshot() and is instructed to reason from those numbers.
The snapshot carries each driver's percentile against that machine's OWN
history, so the reasoning — not just the digits — has to change when the
sensor picture changes. src/test_llm_grounded.py enforces that.

Requires GROQ_API_KEY in the environment (loaded from .env, which is gitignored).
"""

import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# gpt-oss models on Groq spend completion tokens on a `reasoning` field before
# emitting `content`. At low effort with a generous cap this stays cheap; with
# a small max_tokens the reasoning consumes the whole budget and content comes
# back EMPTY with finish_reason="length".
REASONING_EFFORT = "low"
MAX_TOKENS = 1000

SYSTEM_PROMPT = """\
You are a condition-monitoring advisor writing for a plant maintenance team. \
A plant manager reads your output in five seconds, so be brief and concrete.

Rules:
- Use ONLY the numbers in the JSON provided. Never invent a reading, a trend, \
or a machine you were not given.
- Lead with what the sensor evidence actually shows, then what to do about it.
- Name the specific driver that stands out and say why it stands out, using its \
percentile against that machine's own history. A reading that is high in \
absolute terms but normal for that machine is NOT a concern; say so.
- If nothing is anomalous, say the machine looks normal and recommend no action. \
Do not manufacture concern.
- If cold_start is true, add one short clause noting the machine has very little \
history so the score is less reliable.
- 2-3 sentences. No bullet points, no headings, no preamble. Plain language a \
technician would use — no ML jargon.
- Express risk as a percentage chance of failure in the next 24 hours, or as a \
plain word (low / elevated / high). NEVER print a bare decimal like "0.94" or \
"risk of 0" — "0" reads as impossible rather than unlikely.
"""


def _clean(text: str) -> str:
    """Normalise the exotic spaces and hyphens models emit (U+202F, U+00A0,
    U+2011); they break cp1252 consoles and render oddly in the UI."""
    text = text.replace(" ", " ").replace(" ", " ").replace("‑", "-")
    return re.sub(r"(\d) (?=(th|st|nd|rd)\b)", r"\1", text)


def get_client(api_key: str = None) -> Groq:
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Put it in .env as GROQ_API_KEY=<key> "
            "(the file is gitignored) or export it in your shell."
        )
    return Groq(api_key=key)


def explain_machine_risk(snapshot: dict, client: Groq = None,
                         model: str = None, temperature: float = 0.2) -> dict:
    """Live Groq call. Returns the explanation plus the exact evidence sent,
    so callers (and the UI) can show what the text was grounded in."""
    client = client or get_client()
    model = model or DEFAULT_MODEL

    evidence = json.dumps(snapshot, indent=2)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Machine risk snapshot:\n\n{evidence}\n\n"
                "Write the explanation."},
        ],
    )
    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        raise RuntimeError(
            f"Groq returned empty content (finish_reason={choice.finish_reason}). "
            "If finish_reason is 'length', raise MAX_TOKENS — reasoning tokens "
            "are consuming the budget before any content is emitted."
        )
    text = _clean(text)
    return {
        "machine_id": snapshot.get("machine_id"),
        "as_of": snapshot.get("as_of"),
        "risk": snapshot.get("risk"),
        "risk_band": snapshot.get("risk_band"),
        "explanation": text,
        "model": model,
        "evidence_sent": snapshot,
    }


if __name__ == "__main__":
    import sys

    import pandas as pd

    sys.stdout.reconfigure(encoding="utf-8")

    from risk_context import machine_snapshot, score_frame

    scored = score_frame()

    # A genuinely high-risk historical moment (MCH-213 ran into a breakdown at
    # 18:00 on 8 April) alongside a quiet one, to show the text really moves.
    cases = [
        ("MCH-213", pd.Timestamp("2026-04-08 04:00:00")),
        ("MCH-202", pd.Timestamp("2026-04-20 12:00:00")),
    ]
    for machine_id, at in cases:
        snap = machine_snapshot(scored, machine_id, at=at)
        result = explain_machine_risk(snap)
        print("=" * 78)
        print(f"{machine_id} as of {snap['as_of']}  "
              f"risk={snap['risk']:.4f} ({snap['risk_band']})")
        print("=" * 78)
        print(result["explanation"])
        print()
