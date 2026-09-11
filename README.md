# Predictive Maintenance: Industrial Sensor Intelligence Platform

A condition monitoring and failure risk forecasting platform for industrial manufacturing equipment. The system ingests hourly multi-sensor telemetry, predicts equipment breakdown risk within a 24-hour forward window, generates diagnostic summaries via a live LLM layer, tracks walk-forward out-of-sample risk trends, and provides natural-language querying over fleet operational history.

**Live Application:** [https://predictive-maintenance-ypxa.onrender.com/](https://predictive-maintenance-ypxa.onrender.com/)

---

## 1. System Architecture

```
                                  +------------------------------+
                                  |     Browser (React SPA)      |
                                  +--------------+---------------+
                                                 | HTTP / JSON
                                                 v
+------------------------------------------------------------------------------------------------+
| Backend Service (FastAPI: backend/main.py)                                                     |
|                                                                                                |
|   /api/meta              /api/fleet                 /api/machines/{id}     /api/qa             |
|   /api/health            /api/machines/{id}/trend   /api/machines/{id}/explain                 |
|                                                                                                |
|   +--------------------+     +---------------------------+     +---------------------------+   |
|   |  Runtime & Caching |     |   Feature & Scoring Core  |     |   LLM Integration (Groq)  |   |
|   |  - Startup caching |     |   - src/features.py       |     |   - src/llm_explain.py    |   |
|   |  - In-memory cache |     |   - src/risk_context.py   |     |   - src/llm_qa.py         |   |
|   |  - Rate limiting   |     |   - models/*.parquet      |     |                           |   |
|   +--------------------+     +---------------------------+     +---------------------------+   |
+------------------------------------------------------------------------------------------------+
```

---

## 2. Problem Formulation & Dataset Characteristics

### Dataset Profile (`project2_manufacturing_sensors.csv`)
* **Fleet Structure:**
  * **15 established machines:** 120 days of continuous hourly telemetry (January to April 2026).
  * **2 cold-start machines (`MCH-300`, `MCH-301`):** 72 hours (~3 days) of operating history.
* **Severe Class Imbalance:**
  * 17 total failure events across 43,354 machine-operating hours (0.039% raw event rate).
  * A naive classifier predicting zero failures achieves 99.96% accuracy while failing to detect any breakdown.
* **Reactive Maintenance Dynamic:**
  * Analysis of `run_hours_since_maintenance` reveals that 77.3% of service resets coincide with or immediately follow a breakdown event.
  * Maintenance is predominantly reactive. Any forward-looking derivation from maintenance resets introduces direct label leakage; all features are restricted to causal history.

### Data Hygiene & Edge Cases
* **Duplicate Rows:** The raw dataset contains 10 duplicate rows on machine `MCH-200` (identical timestamps and sensor measurements, none during breakdown events). These rows are retained in the pipeline to preserve exact benchmark parity with initial exploratory data analyses rather than silently altering row counts.
* **Telemetry Missingness:** Sensor dropouts in the dataset occur predominantly as single, isolated missing hours. These are linearly interpolated along the causal time axis without lookahead.

---

## 3. Machine Learning Methodology

### Risk Definition
Failure risk is formulated as:
$$\text{Risk} = P(\text{Failure Event occurs in } (t, t + 24\text{h}] \mid \text{Telemetry up to } t)$$

Each machine-hour is assigned a binary label of `1` if a breakdown occurs within the subsequent 24 hours, and `0` otherwise. The final 24 hours of each machine's recording are censored (dropped from training) because their future state cannot be observed.

### Evaluation Metrics Under Imbalance
* **Primary Ranking Metric:** **AUC-PR** (Area Under the Precision-Recall Curve). AUC-PR evaluates positive-class discrimination without distortion from the large volume of true negative hours.
* **Operational Optimization Metric:** **Event-Level Capture vs. False Alarm Episodes**. Rather than evaluating isolated hours, consecutive alert hours on a machine are grouped into distinct alert episodes. Models are evaluated by the proportion of real breakdowns detected prior to failure against the total number of false alarm episodes generated.

### Causal Feature Engineering (26 Features)
1. **Raw Aggregates (19 features):** Rolling means (6h, 24h), rolling standard deviations (6h, 24h), backward differences (1h, 6h), operating hours since maintenance, a 24h reset indicator, and one-hot production line encodings.
2. **Machine-Relative Normalization (7 features):** Each rolling aggregate divided by that machine's expanding historical median:
   $$\text{Baseline} = \text{shift}(1).\text{expanding}(\text{min\_periods}=24).\text{median}()$$
   $$\text{Relative Feature} = \frac{\text{Current Aggregate}}{\text{Baseline}}$$
   This captures whether sensor volatility or temperature is elevated relative to that specific asset's baseline. Machine-relative features account for 70.7% of total model importance and reduce false alarms by approximately 30% at equal event capture.

### Validation Scheme
* **Strict Time-Ordered Splits:** Evaluated across a primary chronological 80/20 split and two expanding time-series cross-validation folds.
* **24-Hour Embargo Buffer:** Training rows within 24 hours of split boundaries are purged to prevent forward-looking label contamination across splits.

### Model Selection Results

| Model Candidate | Fold 1 (Mar) AUC-PR | Fold 2 (Apr) AUC-PR | Primary Split AUC-PR | Mean AUC-PR | Real Breakdowns Caught | False Alarm Episodes |
|---|---|---|---|---|---|---|
| Naive Baseline (Never Alert) | 0.0086 | 0.0180 | 0.0198 | 0.0155 | 0 / 19 | 0 |
| Logistic Regression | 0.5711 | 0.6575 | 0.6705 | 0.6330 | 18 / 19 | 39 |
| Decision Tree (depth=4) | 0.1265 | 0.4042 | 0.3688 | 0.2998 | 11 / 19 | 62 |
| **Random Forest (`leaf=20`) [Shipped]** | **0.5561** | **0.7052** | **0.7241** | **0.6618** | **19 / 19** | **38** |
| HistGradientBoosting | 0.1878 | 0.6215 | 0.4908 | 0.4334 | 14 / 19 | 51 |
| XGBoost | 0.3202 | 0.5995 | 0.4559 | 0.4585 | 13 / 19 | 48 |
| Ensemble (Averaged Probability) | 0.5030 | 0.6423 | 0.5884 | 0.5779 | 17 / 19 | 42 |

**Selected Model Configuration:** Random Forest Classifier (300 estimators, `max_depth=8`, `min_samples_leaf=20`, balanced class weights). Operating threshold set at **0.20**, providing 100% breakdown detection across validation folds (19 of 19 events) with a median lead time of 24 hours (minimum 16 hours).

### Operational Risk Thresholds & Trade-offs
* **HIGH ($\ge 0.20$):** Actionable breakdown alert requiring immediate inspection. Accounts for ~2.8% of fleet hours.
* **WATCH ($0.02 - 0.20$):** Elevated risk advisory. Accounts for ~1.5% of fleet hours.
* **LOW ($< 0.02$):** Normal operating condition. Accounts for ~95.7% of fleet hours.

**The Asymmetric Cost Trade-off:** At the 0.20 alert line, approximately 3 out of every 5 alert episodes turn out to be false alarms (precision ~38%). In manufacturing operations, this trade-off is intentional: a false alarm triggers a 15-minute diagnostic inspection, whereas a missed failure results in uncoordinated downtime, product spoilage, and secondary mechanical damage. Maximizing event recall while maintaining a manageable alert volume is the operational objective.

### Warning Lead-Time Interpretation
* **Cross-Validation Minimum Lead Time (16 hours):** Evaluated strictly on held-out test splits against unseen machine breakdowns.
* **Walk-Forward Trend Minimum Lead Time (23 hours):** Evaluated on the continuous out-of-sample weekly historical scoring timeline displayed in the live application's activity strip. Both metrics reflect empirical lead warning across different evaluation spans.

### Dual-Scoring Strategy
* **Present State Scoring:** Uses the production model refit on all established historical data (`models/failure_risk_rf.joblib`).
* **Historical Trend Scoring:** Employs rolling out-of-sample walk-forward scoring (`models/historical_scores_oos.parquet`), ensuring past risk trends are evaluated only using data available prior to each scored week.

### Cold-Start Management (`MCH-300`, `MCH-301`)
* Assets with under 24 hours of baseline default relative ratios to 1.0 (neutral), relying on raw telemetry.
* Machine identifiers (`machine_id`) are excluded from training features, allowing the model to generalize to unseen assets.
* Cold-start assets are flagged with explicit confidence indicators in the user interface.

---

## 4. Condition Monitoring & LLM Reasoning Layer

The application integrates Groq runtime inference (`openai/gpt-oss-120b`) for explainability and fleet Q&A:

### 1. Grounded Machine Explanations (`src/llm_explain.py`)
* The backend builds a structured numeric snapshot per machine: current sensor readings, relative ratios against baseline, and driver percentiles.
* The LLM converts these ratios into plain-language summaries for maintenance personnel instead of templated strings.
* Tested across different sensor conditions to confirm it produces varied, condition-specific output rather than generic boilerplate.

### 2. Two-Stage Natural Language Q&A (`src/llm_qa.py`)
* **Stage 1 (Deterministic Extraction):** User queries are parsed via regex into entity filters (machine IDs, production lines, risk bands, dates, and failure history) to retrieve exact rows from the parquet dataset.
* **Stage 2 (Bounded Synthesis):** The retrieved rows are passed to the LLM with strict instructions to answer only from the provided records. The user interface exposes a provenance drawer displaying the exact retrieved rows behind every response.

### 3. Resilience, Caching & Rate-Limiting
* **Decoupled Architecture:** Telemetry ingestion, risk scoring, fleet tables, and sensor trend charts function independently of the LLM. If the Groq API key is missing or invalid, the core application continues to operate without interruption; only LLM-dependent endpoints return informative notices.
* **In-Memory Caching:** Diagnostic explanations are cached in memory per machine-hour, preventing redundant external API calls during routine dashboard navigation.
* **Provider Rate-Limit Handling:** Groq free-tier limits (8,000 tokens/minute) are managed gracefully. The backend catches upstream rate limits, logs the root cause server-side, and returns user-friendly HTTP 429 status codes with retry advisories rather than exposing raw provider errors.

---

## 5. Web Application Architecture

* **Backend:** FastAPI service with startup caching, in-memory rate limiting, and RESTful endpoints:
  * `GET /api/meta`: Fleet metadata, threshold definitions, and operational activity summaries.
  * `GET /api/fleet`: Fleet snapshot with risk rankings, line filters, and 30-day sparklines.
  * `GET /api/machines/{id}`: Machine sensor history, risk drivers, and baseline comparisons.
  * `GET /api/machines/{id}/trend`: Out-of-sample historical walk-forward risk series.
  * `POST /api/machines/{id}/explain`: Runtime condition monitoring summary.
  * `POST /api/qa`: Natural-language fleet search with retrieved evidence.
  * `GET /api/health`: Service health verification.
* **Frontend:** React application built with Vite, Recharts, and vanilla CSS:
  * **Fleet Overview Table:** Real-time risk sorting, production line filtering, and per-machine risk sparklines.
  * **Operational Activity Strip:** Historical breakdown capture rate and lead-time metrics with direct navigation to recorded failure events.
  * **Time-Travel Control:** Historical selector enabling inspection of fleet telemetry at any historical timestamp.
  * **Machine Drill-Down:** Individual sensor charts (temperature and vibration rendered independently to prevent dual-axis distortion), risk driver tables, and walk-forward trend lines.
  * **Theme Support:** Native dark and light modes with system auto-detection.

---

## 6. Project Directory Layout

```
├── backend/
│   └── main.py                     FastAPI application, routes, caching, rate limits
├── docs/
│   └── model_selection.md          Comprehensive model comparison and validation report
├── frontend/
│   ├── src/
│   │   ├── components/             React UI components (Dashboard, Charts, ActivityStrip, Q&A)
│   │   ├── App.jsx                 Root application layout
│   │   ├── api.js                  API client layer
│   │   └── styles.css              Industrial design system and theme tokens
│   ├── package.json                Frontend dependencies (React, Vite, Recharts)
│   └── vite.config.js              Vite build and bundle optimization configuration
├── models/
│   ├── failure_risk_rf.joblib      Fitted Random Forest classifier
│   ├── failure_risk_rf.meta.json   Model hyperparameters and training metadata
│   ├── current_scores.parquet      Pre-scored fleet dataset for sub-millisecond API response
│   └── historical_scores_oos.parquet Out-of-sample walk-forward risk scores
├── src/
│   ├── config.py                   Shared paths, risk thresholds, and model hyperparameters
│   ├── data_exploration.py         Dataset profiling, missingness, and imbalance analysis
│   ├── features.py                 Causal feature engineering, rolling stats, and relative ratios
│   ├── risk_context.py             Telemetry snapshots, baseline calculations, and retrieval core
│   ├── train_baseline.py           Time-series cross-validation and candidate model comparisons
│   ├── threshold_sweep.py          Empirical threshold derivation across validation folds
│   ├── event_level_analysis.py     Event-level detection, false alarm episode calculations
│   ├── train_final_model.py        Final model fitting and artifact serialization
│   ├── build_current_scores.py     Batch scoring pipeline for latest telemetry
│   ├── build_historical_scores.py  Rolling weekly walk-forward out-of-sample scoring pipeline
│   ├── llm_explain.py              Groq condition monitoring explanation service
│   └── llm_qa.py                   Two-stage deterministic retrieval and natural-language Q&A
├── Dockerfile                      Multi-stage build (Node 24 build stage -> Python 3.13 runtime)
├── render.yaml                     Render cloud deployment blueprint
├── requirements.txt                Production Python dependencies
└── project2_manufacturing_sensors.csv Telemetry dataset
```

---

## 7. Installation & Local Setup

### Prerequisites
* Python 3.11+
* Node.js 18+

### Setup Instructions

1. **Clone the repository and install backend dependencies:**
   ```bash
   git clone https://github.com/Akinkunmi100/FMN-Machine-Sensor-Intelligence.git
   cd FMN-Machine-Sensor-Intelligence
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Add your GROQ_API_KEY to .env
   ```

3. **Install frontend dependencies and build assets:**
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

4. **Start the application server:**
   ```bash
   uvicorn backend.main:app --port 8000
   ```
   The application will be accessible at `http://localhost:8000`.

### Development Mode (Separate API and Vite Dev Server)
```bash
# Terminal 1: Backend API
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Frontend Vite Server
cd frontend
npm run dev
# Accessible at http://localhost:5173 (proxies API requests to :8000)
```

---

## 8. Docker Deployment

Build and run the production container locally:

```bash
docker build -t predictive-maintenance .
docker run -p 8000:8000 -e GROQ_API_KEY="your_groq_api_key_here" predictive-maintenance
```

---

## 9. Pipeline Reproduction Reference

To reproduce the analysis and model artifacts from the raw dataset:

```bash
python -m src.data_exploration        # Run exploratory sensor data analysis
python -m src.train_baseline          # Evaluate candidate models across time-series splits
python -m src.threshold_sweep         # Run empirical threshold sweep
python -m src.event_level_analysis    # Evaluate event-level breakdown capture vs false alarms
python -m src.train_final_model       # Fit and export production Random Forest model
python -m src.build_current_scores    # Pre-compute current fleet risk scores
python -m src.build_historical_scores # Generate walk-forward out-of-sample historical scores
```

---

## 10. Operational Considerations & Engineering Roadmap

* **Sample Size & Generalization:** 17 failure events provide a focused basis for statistical learning. Detecting 100% of tested breakdowns (19 of 19 validation events across folds) demonstrates strong empirical discrimination on historical data, though continuous operational monitoring is required as fleet telemetry accumulates.
* **Cold-Start Monitoring:** With 72 operating hours and zero historical failure events, `MCH-300` and `MCH-301` are unvalidated against true breakdown patterns. The platform surfaces explicit cold-start flags until assets accumulate sufficient operating history to establish individualized baselines.
* **Walk-Forward Trend Horizon:** Historical walk-forward scoring begins mid-timeline because early weeks must be reserved as initial training history before the rolling retrain loop begins.
* **Future Roadmap:**
  * **Distributed Caching:** Introduce Redis to support distributed multi-instance rate limiting and shared explanation caching across clustered API workers.
  * **Extended Imputation:** Implement model-based imputation for telemetry outages exceeding 6 continuous hours.
  * **Dedicated Cold-Start Evaluation:** Formulate a separate benchmark evaluation for new equipment once operational runtime yields candidate failure incidents.
