# Predictive Maintenance: Industrial Sensor Intelligence Platform

A condition monitoring and failure risk forecasting platform for industrial manufacturing equipment. The system ingests hourly multi-sensor telemetry, predicts equipment breakdown risk within a 24-hour forward window, generates diagnostic summaries via a live LLM layer, tracks walk-forward out-of-sample risk trends, and provides natural-language querying over fleet operational history.

**Live Application:** [https://predictive-maintenance-ypxa.onrender.com/](https://predictive-maintenance-ypxa.onrender.com/)

---

## 1. Problem Understanding

### Operational Context & Sponsor Mandate
Continuous industrial manufacturing operations depend heavily on electro-mechanical equipment running uninterrupted across multi-stage production lines. When a critical machine suffers an unexpected breakdown, the impact is rarely isolated: downstream production halts, raw product batches spoil, and mechanical components sustain secondary damage. 

The sponsor's operational mandate is to transition plant operations from reactive fire-fighting to proactive condition monitoring by delivering:
1. An actionable **24-hour forward warning window** that provides maintenance crews sufficient lead time to inspect equipment, coordinate planned changeovers, and stage replacement parts before catastrophic breakdown occurs.
2. **Interpretable diagnostics** grounded in machine-specific sensor telemetry, moving beyond opaque risk scores so maintenance technicians understand exactly which physical indicators are driving an alert.
3. An operational software platform providing both real-time fleet surveillance and verifiable historical risk auditing.

### Operational Tension: Recall vs. Alert Fatigue
The central engineering challenge in industrial monitoring is balancing failure detection against operational credibility. A system tuned to alert only on extreme anomalies will miss subtle pre-failure degradation, leading to costly uncoordinated downtime. Conversely, a system that generates frequent false alarms quickly induces alert fatigue, causing plant personnel to ignore notifications entirely. Resolving this tension requires aligning evaluation metrics directly with event-level maintenance interventions rather than naive row-level statistics.

### Dataset Characteristics & Severe Class Imbalance
The operational dataset (`project2_manufacturing_sensors.csv`) exhibits several distinct structural properties:
* **Fleet Profile:** 15 established machines tracked across 120 days of continuous hourly telemetry (January to April 2026), alongside 2 newly commissioned cold-start machines (`MCH-300`, `MCH-301`) with 72 hours (approximately 3 days) of operating history.
* **Extreme Event Sparsity:** 17 total failure events occur across 43,354 recorded machine-hours, representing a baseline failure rate of 0.039%.
* **The Accuracy Paradox:** A naive classification baseline that predicts zero failures achieves 99.96% raw accuracy while detecting zero breakdowns. Accuracy is therefore fundamentally disqualified as an evaluation metric.

### The Reactive Maintenance Trap (Label Leakage)
During exploratory analysis, the `run_hours_since_maintenance` field appeared to offer a direct proxy for machine wear. However, detailed investigation revealed that 77.3% of all maintenance resets coincide with or immediately follow a recorded breakdown event. 

In this plant, maintenance operations are overwhelmingly *reactive*: machines run until failure occurs, after which repair crews service the asset and reset the counter. Deriving forward-looking features from maintenance resets (such as estimating time until next service) introduces direct label leakage, artificially inflating offline performance while collapsing in production. To preserve causal integrity, all feature pipelines are strictly restricted to backward-looking history up to timestamp $t$.

### Data Hygiene & Edge Cases
* **Duplicate Telemetry Rows:** Machine `MCH-200` contains 10 duplicate rows with identical timestamps and sensor readings (none occurring during breakdown events). These rows are retained in the pipeline to maintain exact row-count parity with initial exploratory baselines rather than silently modifying source data.
* **Sensor Missingness:** Telemetry dropouts in the dataset occur predominantly as isolated, single-hour missing records. These are imputed causally via linear interpolation along the time index without lookahead leakage.

---

## 2. Approach

### System Architecture Overview
The platform couples an offline causal machine learning pipeline with a decoupled, high-performance web service and an in-memory LLM diagnostic layer:

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

### Risk Formulation & Evaluation Metrics
Failure risk is framed as a 24-hour forward probability:

$$\text{Risk} = P(\text{Failure Event occurs in } (t, t + 24\text{h}] \mid \text{Telemetry up to } t)$$

Each machine-hour is labeled `1` if a failure occurs within the subsequent 24 hours, and `0` otherwise. The final 24 hours of each machine recording are censored (excluded from training) because their subsequent operational state cannot be verified.

#### Evaluation Metric Hierarchy:
1. **Ranking Metric (AUC-PR):** Area Under the Precision-Recall Curve serves as the primary discriminator. Unlike ROC-AUC, AUC-PR is not flattered by the overwhelming volume of true negative operating hours.
2. **Decision Metric (Event Capture vs. False Alarm Episodes):** Rather than tallying individual hourly predictions, consecutive alert hours on an asset are consolidated into single operational alert episodes. Models are selected based on the total count of real breakdown events detected prior to failure versus the total volume of false alarm episodes generated.

### Causal Feature Engineering (26 Features)
The feature engineering pipeline transforms raw sensor streams into 26 causal dimensions:
1. **Raw Aggregates (19 features):** Rolling sensor averages (6h, 24h), rolling standard deviations (6h, 24h), backward differences (1h, 6h), run hours since maintenance, a 24h reset flag, and one-hot line identifiers.
2. **Machine-Relative Normalization (7 features):** Rolling aggregates normalized by that specific machine's expanding historical median:
   $$\text{Baseline} = \text{shift}(1).\text{expanding}(\text{min\_periods}=24).\text{median}()$$
   $$\text{Relative Feature} = \frac{\text{Current Aggregate}}{\text{Baseline}}$$

Relative features account for 70.7% of total model importance and reduce false alarms by approximately 30% at identical breakdown capture rates.

#### Feature Importance Distribution:

| Feature Dimension | Relative Contribution | Operational Significance |
|---|---|---|
| 24h average temperature (relative to machine baseline) | Highest (Rank 1) | Detects persistent thermal buildup specific to that unit |
| 24h vibration volatility (relative to machine baseline) | 2nd (Rank 2) | Detects abnormal mechanical oscillation shifts |
| 6h average temperature (relative to machine baseline) | 3rd (Rank 3) | Captures rapid short-term thermal acceleration |
| 24h vibration volatility (raw absolute value) | 4th (Rank 4) | Enforces absolute fleet-wide mechanical ceilings |
| 24h temperature volatility (raw absolute value) | 5th (Rank 5) | Enforces absolute fleet-wide thermal ceilings |
| Hours since last maintenance | Negligible (<0.1%) | Confirms leakage prevention; reactive resets carry minimal causal signal |

### Model Selection & Validation Scheme
Evaluation is conducted across a strict 80/20 chronological split and two expanding time-series cross-validation folds. A 24-hour embargo buffer purges training observations adjacent to split boundaries, preventing label contamination across temporal folds.

#### Candidate Benchmark Comparison:

| Model Candidate | Fold 1 (Mar) AUC-PR | Fold 2 (Apr) AUC-PR | Primary Split AUC-PR | Mean AUC-PR | Real Breakdowns Caught | False Alarm Episodes |
|---|---|---|---|---|---|---|
| Naive Baseline (Never Alert) | 0.0086 | 0.0180 | 0.0198 | 0.0155 | 0 / 19 | 0 |
| Logistic Regression | 0.5711 | 0.6575 | 0.6705 | 0.6330 | 18 / 19 | 39 |
| Decision Tree (depth=4) | 0.1265 | 0.4042 | 0.3688 | 0.2998 | 11 / 19 | 62 |
| **Random Forest (`leaf=20`) [Shipped]** | **0.5561** | **0.7052** | **0.7241** | **0.6618** | **19 / 19** | **38** |
| HistGradientBoosting | 0.1878 | 0.6215 | 0.4908 | 0.4334 | 14 / 19 | 51 |
| XGBoost | 0.3202 | 0.5995 | 0.4559 | 0.4585 | 13 / 19 | 48 |
| Ensemble (Averaged Probability) | 0.5030 | 0.6423 | 0.5884 | 0.5779 | 17 / 19 | 42 |

**Selection Rationale:** The Random Forest classifier (300 estimators, `max_depth=8`, `min_samples_leaf=20`, balanced class weights) won on both primary objectives simultaneously: it detected 100% of tested failure events (19 of 19 across validation splits) while generating the lowest false alarm count (38 episodes). Row-level metrics can be deceptively optimistic (for example, Logistic Regression scored competitive hourly metrics while missing an entire breakdown event). Event-level validation was the decisive selection criterion.

### Operational Thresholds & Alert Economics
Predictions are categorized into three operational tiers:

| Alert Band | Probability Cutoff | Fleet Hours Share | Action Protocol |
|---|---|---|---|
| **HIGH** | $\ge 0.20$ | ~2.8% | Immediate on-site physical inspection and parts preparation |
| **WATCH** | $0.02 - 0.20$ | ~1.5% | Elevated telemetry monitoring across subsequent shift |
| **LOW** | $< 0.02$ | ~95.7% | Standard operational baseline |

**The Asymmetric Cost Rationale:** At the 0.20 operating threshold, precision is approximately 38% (roughly 3 out of 5 alert episodes are false alarms). In manufacturing plant operations, this asymmetry is intentional. A false alarm incurs a 15-minute diagnostic inspection by a shift technician. A missed breakdown results in hours of unplanned downtime, scrap material, and extensive mechanical damage. Prioritizing breakdown recall while containing alerts to under 3% of fleet runtime represents the optimal operational trade-off.

**Lead-Time Metrics:**
* **Cross-Validation Test Splits:** 16-hour minimum warning lead time prior to failure.
* **Continuous Walk-Forward Historical Timeline:** 23-hour minimum warning lead time prior to failure.

### Dual-Scoring Strategy & Cold-Start Handling
* **Dual Scoring:** Real-time monitoring uses the final model trained on all historical records (`models/current_scores.parquet`), while the UI activity timeline uses rolling out-of-sample walk-forward scoring (`models/historical_scores_oos.parquet`) to reflect strictly causal historical performance.
* **Cold-Start Assets (`MCH-300`, `MCH-301`):** Assets with less than 24 hours of operating history default relative ratios to 1.0 (neutral), relying on raw telemetry boundaries. `machine_id` is excluded from model features to ensure generalizability, and cold-start assets are flagged with explicit UI confidence indicators.

### Condition Monitoring & LLM Reasoning Layer
The platform integrates Groq runtime inference (`openai/gpt-oss-120b`) for explainability and fleet auditing:
* **Grounded Machine Explanations (`src/llm_explain.py`):** The backend builds a structured numeric telemetry snapshot per machine (current readings, baseline ratios, driver percentiles). The LLM translates these ratios into concise plain-language maintenance summaries. Verification scripts confirm that output wording dynamically varies with sensor severity rather than returning canned templates.
* **Two-Stage Deterministic Q&A (`src/llm_qa.py`):** Stage 1 uses deterministic regex parsing to extract entities (machine IDs, production lines, risk tiers, dates, failure history) and query the parquet store directly. Stage 2 passes only the retrieved rows to the LLM with strict instructions to answer exclusively from evidence. The UI provides a provenance drawer displaying the exact records backing each response.
* **Decoupled Resilience & Caching:** Dashboards, charts, and risk rankings function entirely independently of the LLM. Missing or invalid API keys do not impair core platform operations. Explanations are cached in memory per machine-hour, and upstream Groq rate limits (8,000 tokens/minute) are intercepted to return clean HTTP 429 advisories rather than raw provider errors.

---

## 3. How to Run

### Live Deployment
The production application is deployed on Render via a multi-stage Docker container:
* **Live Application URL:** [https://predictive-maintenance-ypxa.onrender.com/](https://predictive-maintenance-ypxa.onrender.com/)

### Prerequisites
* Python 3.11+
* Node.js 18+
* Git

### Local Installation & Setup

1. **Clone the repository and install backend dependencies:**
   ```bash
   git clone https://github.com/Akinkunmi100/FMN-Machine-Sensor-Intelligence.git
   cd FMN-Machine-Sensor-Intelligence
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Add your GROQ_API_KEY to .env (core telemetry functions without an API key)
   ```

3. **Install frontend dependencies and compile static assets:**
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

4. **Launch the production service:**
   ```bash
   uvicorn backend.main:app --port 8000
   ```
   Access the dashboard at `http://localhost:8000`.

### Development Mode (Hot-Reloading)
To run frontend and backend processes independently during development:

```bash
# Terminal 1: Backend API service
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Frontend Vite development server
cd frontend
npm run dev
# Accessible at http://localhost:5173 (proxies API requests to :8000)
```

### Docker Deployment
To build and execute the container locally:

```bash
docker build -t predictive-maintenance .
docker run -p 8000:8000 -e GROQ_API_KEY="your_groq_api_key_here" predictive-maintenance
```

### Pipeline Reproduction Reference
To re-run exploratory data analysis, feature generation, model training, and score compilation from source data:

```bash
python -m src.data_exploration        # Dataset profiling and class imbalance audit
python -m src.train_baseline          # Candidate model evaluation across time splits
python -m src.threshold_sweep         # Operational threshold derivation
python -m src.event_level_analysis    # Breakdown event capture and false alarm evaluation
python -m src.train_final_model       # Fit and serialize production Random Forest model
python -m src.build_current_scores    # Pre-compute latest fleet risk scores
python -m src.build_historical_scores # Compile rolling out-of-sample historical scores
```

---

## 4. Limitations & Next Steps

### Current Operational Constraints & Trade-offs
* **Sample Size Boundaries:** The historical dataset contains 17 failure events across 43,354 hours. While detecting 100% of tested breakdowns (19 of 19 validation events across folds) demonstrates strong empirical separation, this is a small-sample finding rather than a permanent guarantee. Ongoing model governance is required as operating runtime expands.
* **False Alarm Overhead:** Operating at the 0.20 decision boundary produces an approximate 38% precision rate (~3 out of 5 alert episodes are false alarms). While operationally justified by the asymmetric cost of missed failures, this false positive rate requires transparent communication to prevent maintenance friction.
* **Unvalidated Cold-Start Equipment:** With only 72 hours of operating telemetry and zero historical breakdown events, machines `MCH-300` and `MCH-301` are unvalidated against real failure signatures. Predictions on these units are marked as preliminary until assets establish sufficient individual baseline history.
* **Walk-Forward Trend Horizon:** Out-of-sample historical trend scoring begins mid-timeline because early weeks must be reserved as initial training history before rolling retraining iterations can execute.
* **Telemetry Gap Imputation Limits:** The current causal imputation scheme handles single isolated missing hours via linear interpolation. Sustained telemetry outages exceeding 6 continuous hours are not modeled and would require sensor-fault fallbacks.
* **Single-Process In-Memory State:** Rate limiting and explanation caching are maintained in process memory. Multi-instance horizontal scaling would require centralized caching infrastructure.

### Engineering Next Steps & Improvements
With additional development time, the following enhancements would be prioritized:
1. **Distributed Caching Infrastructure:** Integrate a Redis instance to manage shared rate limiting, cross-process explanation caching, and task queuing across horizontal API replicas.
2. **Extended Outage Imputation:** Implement model-based spatial imputation (leveraging cross-sensor correlations across identical machine types) to handle multi-hour telemetry dropouts without data corruption.
3. **Dedicated Cold-Start Evaluation:** Formalize a specialized evaluation track for newly commissioned assets once operational history generates their first failure incidents.
4. **Automated Plant Dispatch:** Implement webhook, SMS, and email alert routing to notify shift supervisors immediately when an asset crosses into `HIGH` risk status.
5. **Direct Telemetry Ingestion:** Transition from batch parquet scoring to continuous streaming ingestion by connecting FastAPI endpoints directly to industrial MQTT brokers or Apache Kafka topics.

---

## Appendix: Project Directory Layout

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
