"""
FastAPI backend for the predictive maintenance dashboard.

Holds no modelling logic of its own. Everything it serves comes from the src/
modules built in Phases 0-3 — risk_context for retrieval and scoring,
llm_explain and llm_qa for the runtime Groq calls. The API layer's only jobs
are caching the scored frame at startup (so nothing retrains per request) and
shaping JSON.

Two distinct score sources, deliberately kept apart:
  - the CURRENT state (dashboard, drill-down, explanations) uses the shipped
    model refit on all history, which is correct for scoring the present;
  - the TREND chart uses models/historical_scores_oos.parquet, walk-forward
    out-of-sample scores, because re-scoring the past with a model that trained
    on it is how you get a chart that flatters itself.

Run locally:  uvicorn backend.main:app --reload --port 8000
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import llm_explain  # noqa: E402
import llm_qa  # noqa: E402
from features import NEW_MACHINES  # noqa: E402
from risk_context import (  # noqa: E402
    RISK_BANDS,
    band_report,
    fleet_snapshot,
    load_model,
    machine_snapshot,
    score_frame,
)

OOS_PATH = ROOT / "models" / "historical_scores_oos.parquet"
FRONTEND_DIST = ROOT / "frontend" / "dist"

STATE: dict = {}
# Explanations are cached per (machine, timestamp) purely so re-renders and tab
# switches don't re-bill Groq. A miss always makes a live call, and a different
# snapshot is a different key — caching is not precomputation.
EXPLANATION_CACHE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    scored = score_frame()
    _, meta = load_model()
    STATE["scored"] = scored
    STATE["meta"] = meta
    STATE["oos"] = pd.read_parquet(OOS_PATH) if OOS_PATH.exists() else None
    failures = (
        scored[scored["failure_event"] == 1][["machine_id", "line", "timestamp"]]
        .sort_values("timestamp")
    )
    STATE["failures"] = failures
    print(f"[startup] scored {len(scored):,} machine-hours, "
          f"{scored['machine_id'].nunique()} machines, "
          f"{len(failures)} recorded failures, "
          f"OOS scores {'loaded' if STATE['oos'] is not None else 'MISSING'}")
    yield
    STATE.clear()


app = FastAPI(title="Predictive Maintenance API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Vite dev server. In production the SPA is served from this same origin,
    # so no cross-origin request happens at all.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def scored_frame() -> pd.DataFrame:
    if "scored" not in STATE:
        raise HTTPException(503, "model not loaded yet")
    return STATE["scored"]


# --------------------------------------------------------------------- meta
@app.get("/api/meta")
def get_meta():
    scored = scored_frame()
    return {
        "risk_bands": RISK_BANDS,
        "horizon_hours": STATE["meta"]["prediction_horizon_hours"],
        "target": STATE["meta"]["target"],
        "model": STATE["meta"]["model_type"],
        "model_params": STATE["meta"]["model_params"],
        "n_features": len(STATE["meta"]["feature_columns"]),
        "trained_on": STATE["meta"]["trained_on"],
        "cold_start_machines": NEW_MACHINES,
        "time_range": {
            "start": str(scored["timestamp"].min()),
            "end": str(scored["timestamp"].max()),
        },
        "band_coverage": band_report(scored),
        # Quick-picks: at the dataset's final hour every machine reads LOW,
        # because the last failure was 30 Apr 04:00. Without these the landing
        # view is an all-green fleet with nothing to look at.
        "recorded_failures": [
            {"machine_id": r["machine_id"], "line": r["line"],
             "timestamp": str(r["timestamp"])}
            for _, r in STATE["failures"].iterrows()
        ],
    }


# -------------------------------------------------------------------- fleet
@app.get("/api/fleet")
def get_fleet(as_of: str | None = Query(None, description="ISO timestamp")):
    scored = scored_frame()
    if as_of:
        try:
            cutoff = pd.Timestamp(as_of)
        except ValueError:
            raise HTTPException(400, f"could not parse as_of: {as_of!r}")
        scored = scored[scored["timestamp"] <= cutoff]
        if scored.empty:
            raise HTTPException(404, f"no data at or before {as_of}")
    fleet = fleet_snapshot(scored)
    return {
        "as_of": max(m["as_of"] for m in fleet),
        "counts": {
            band: sum(1 for m in fleet if m["risk_band"] == band)
            for band in ("HIGH", "MEDIUM", "LOW")
        },
        "machines": fleet,
    }


# ------------------------------------------------------------------ machine
@app.get("/api/machines/{machine_id}")
def get_machine(machine_id: str, as_of: str | None = Query(None),
                history_hours: int = Query(168, ge=24, le=2880)):
    scored = scored_frame()
    try:
        snap = machine_snapshot(
            scored, machine_id, at=pd.Timestamp(as_of) if as_of else None
        )
    except (ValueError, IndexError):
        raise HTTPException(404, f"no data for machine {machine_id}")

    end = pd.Timestamp(snap["as_of"])
    hist = scored[(scored["machine_id"] == machine_id)
                  & (scored["timestamp"] <= end)
                  & (scored["timestamp"] > end - pd.Timedelta(hours=history_hours))]
    snap["sensor_history"] = [
        {
            "timestamp": str(r["timestamp"]),
            "temperature_c": round(float(r["temperature_c"]), 2),
            "vibration_mm_s": round(float(r["vibration_mm_s"]), 3),
            "run_hours_since_maintenance": int(r["run_hours_since_maintenance"]),
            "maintenance_reset": bool(r["run_hours_since_maintenance"] == 0),
            "failure_event": bool(r["failure_event"] == 1),
        }
        for _, r in hist.iterrows()
    ]
    return snap


@app.get("/api/machines/{machine_id}/trend")
def get_trend(machine_id: str):
    """Walk-forward out-of-sample risk. Rows before the first trainable window
    are returned with risk=null and a status, so the UI renders an honest gap
    rather than implying the model was quiet there."""
    oos = STATE.get("oos")
    if oos is None:
        raise HTTPException(
            503, "historical scores missing — run python src/build_historical_scores.py"
        )
    g = oos[oos["machine_id"] == machine_id].sort_values("timestamp")
    if g.empty:
        raise HTTPException(404, f"no trend data for machine {machine_id}")

    fails = STATE["failures"]
    return {
        "machine_id": machine_id,
        "alert_threshold": RISK_BANDS["high"],
        "note": ("Out-of-sample: each week scored by a model trained only on "
                 "earlier data. Not comparable to the current-state score, "
                 "which uses the model refit on all history."),
        "points": [
            {"timestamp": str(r["timestamp"]),
             "risk": None if pd.isna(r["risk"]) else round(float(r["risk"]), 4),
             "status": r["status"]}
            for _, r in g.iterrows()
        ],
        "failures": [str(t) for t in
                     fails.loc[fails["machine_id"] == machine_id, "timestamp"]],
    }


# ------------------------------------------------------------------ the LLM
@app.post("/api/machines/{machine_id}/explain")
def explain(machine_id: str, as_of: str | None = Query(None)):
    scored = scored_frame()
    try:
        snap = machine_snapshot(
            scored, machine_id, at=pd.Timestamp(as_of) if as_of else None
        )
    except (ValueError, IndexError):
        raise HTTPException(404, f"no data for machine {machine_id}")

    key = (machine_id, snap["as_of"])
    if key in EXPLANATION_CACHE:
        cached = dict(EXPLANATION_CACHE[key])
        cached["cached"] = True
        return cached
    try:
        result = llm_explain.explain_machine_risk(snap)
    except Exception as exc:
        raise HTTPException(502, f"explanation service unavailable: {exc}")
    result["cached"] = False
    EXPLANATION_CACHE[key] = result
    return result


class Question(BaseModel):
    question: str


@app.post("/api/qa")
def qa(payload: Question):
    q = payload.question.strip()
    if not q:
        raise HTTPException(400, "question is empty")
    try:
        return llm_qa.answer_question(q, scored=scored_frame())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"question service unavailable: {exc}")


@app.get("/api/health")
def health():
    return {"ok": "scored" in STATE, "oos_loaded": STATE.get("oos") is not None}


# ----------------------------------------------------------- static frontend
# In production the built SPA ships from this same origin, so there is one
# Render service and no CORS. Mounted last so /api/* always wins.
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"),
              name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404, "unknown API route")
        return FileResponse(FRONTEND_DIST / "index.html")
