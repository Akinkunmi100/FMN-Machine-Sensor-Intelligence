# Predictive Maintenance: Industrial Sensor Intelligence Platform

A condition monitoring and failure risk forecasting platform for industrial manufacturing equipment. The system ingests hourly multi-sensor telemetry, predicts equipment breakdown risk within a 24-hour forward window, generates diagnostic summaries via a live LLM layer, tracks walk-forward out-of-sample risk trends, and provides natural-language querying over fleet operational history.

**Live Application:** [https://predictive-maintenance-ypxa.onrender.com/](https://predictive-maintenance-ypxa.onrender.com/)

---

## 1. Problem Understanding

### Operational Context & Objectives
Continuous manufacturing operations rely on mechanical equipment running across production lines. Unplanned breakdowns interrupt production schedules, generate scrap material, and risk secondary mechanical damage.

This project delivers an industrial predictive maintenance system designed around three core operational requirements:
1. **24-Hour Warning Window:** Predict breakdown risk within a 24-hour forward window, giving maintenance teams adequate lead time to inspect equipment, coordinate scheduled stops, and prepare replacement parts.
2. **Interpretable Diagnostics:** Provide sensor-level explanations for elevated risk scores so maintenance technicians understand which telemetry streams drove an alert.
3. **Operational Platform:** Deliver an interactive web dashboard for real-time fleet surveillance alongside historical out-of-sample trend analysis.

### Operational Trade-off: Detection vs. False Alarms
Industrial condition monitoring requires balancing early detection against operational costs. Setting alert thresholds too conservatively risks missing developing equipment degradation. Conversely, excessive false alarms waste technician hours and erode confidence in automated alerts. The evaluation framework is therefore built around event-level detection rates and false alarm episode counts rather than standard row-level metrics.

### Dataset Characteristics & Class Imbalance
The operational dataset (`project2_manufacturing_sensors.csv`) covers two groups of machines:
* **Fleet Profile:** 15 established machines tracked across 120 days of continuous hourly telemetry (January to April 2026), and 2 newly commissioned cold-start machines (`MCH-300`, `MCH-301`) with 72 hours (approximately 3 days) of operating history.
* **Event Sparsity:** 17 total failure events occur across 43,354 recorded machine-hours, representing a baseline failure rate of 0.039%.
* **Limitations of Raw Accuracy:** A baseline model predicting zero failures achieves 99.96% accuracy while identifying zero breakdowns. Evaluation must therefore rely on metrics tailored to severe class imbalance.

### Reactive Maintenance Dynamics & Label Leakage Prevention
During initial data analysis, the `run_hours_since_maintenance` counter appeared to offer an indicator of equipment usage. However, 77.3% of all service resets coincide with or immediately follow a recorded breakdown event.

Plant maintenance in this facility is predominantly reactive: machines operate until a failure occurs, after which repair crews service the unit and reset the counter. Deriving forward-looking features from maintenance resets (such as estimating time until next service) introduces direct label leakage, artificially inflating offline evaluation metrics while failing in live deployment. All engineered features are strictly restricted to historical telemetry up to timestamp $t$.

### Data Preprocessing & Edge Cases
* **Duplicate Telemetry Rows:** Machine `MCH-200` contains 10 duplicate rows with identical timestamps and sensor readings (none occurring during breakdown events). These rows are retained in the pipeline to maintain exact row-count parity with exploratory baselines rather than altering raw data.
* **Sensor Missingness:** Telemetry dropouts in the dataset occur predominantly as isolated single-hour gaps. These are interpolated along the time index using causal linear interpolation without lookahead.

---

## 2. Approach

### System Architecture Overview
The platform consists of an offline machine learning pipeline, a FastAPI web application, and a localized LLM diagnostic interface:

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

Each machine-hour is labeled `1` if a failure occurs within the subsequent 24 hours, and `0` otherwise. The final 24 hours of each machine recording are censored (excluded from training) because their subsequent operational state is unknown.

#### Evaluation Metric Hierarchy:
1. **Ranking Metric (AUC-PR):** Area Under the Precision-Recall Curve serves as the primary discriminator. Unlike ROC-AUC, AUC-PR evaluates positive-class ranking performance without distortion from the large volume of non-failure operating hours.
2. **Decision Metric (Event Capture vs. False Alarm Episodes):** Consecutive alert hours on an asset are grouped into single operational alert episodes. Models are selected based on the total number of real breakdowns detected prior to failure versus the total number of false alarm episodes produced.

### Causal Feature Engineering (26 Features)
The feature engineering pipeline transforms raw sensor telemetry into 26 causal features:
1. **Raw Aggregates (19 features):** Rolling sensor averages (6h, 24h), rolling standard deviations (6h, 24h), backward differences (1h, 6h), run hours since maintenance, a 24h reset flag, and one-hot line identifiers.
2. **Machine-Relative Normalization (7 features):** Rolling aggregates normalized by each machine's historical median:
   $$\text{Baseline} = \text{shift}(1).\text{expanding}(\text{min\_periods}=24).\text{median}()$$
   $$\text{Relative Feature} = \frac{\text{Current Aggregate}}{\text{Baseline}}$$

Machine-relative features account for 70.7% of total model importance and reduce false alarms by approximately 30% at equivalent breakdown detection rates.

#### Feature Importance Distribution:

| Feature Dimension | Relative Contribution | Operational Significance |
|---|---|---|
| 24h average temperature (relative to machine baseline) | Highest (Rank 1) | Identifies persistent thermal buildup specific to that unit |
| 24h vibration volatility (relative to machine baseline) | 2nd (Rank 2) | Identifies abnormal mechanical oscillation shifts |
| 6h average temperature (relative to machine baseline) | 3rd (Rank 3) | Captures short-term thermal acceleration |
| 24h vibration volatility (raw absolute value) | 4th (Rank 4) | Enforces absolute fleet-wide vibration limits |
| 24h temperature volatility (raw absolute value) | 5th (Rank 5) | Enforces absolute fleet-wide temperature limits |
| Hours since last maintenance | Minimal (<0.1%) | Confirms leakage prevention; reactive resets provide little causal predictive signal |

### Model Selection & Validation Scheme
Evaluation is conducted using an 80/20 chronological split and two expanding time-series cross-validation folds. A 24-hour buffer window is applied between training and test sets to prevent label contamination across split boundaries.

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

**Selection Rationale:** The Random Forest classifier (300 estimators, `max_depth=8`, `min_samples_leaf=20`, balanced class weights) delivered the strongest operational performance: it detected 100% of tested failure events (19 of 19 across validation splits) while generating the lowest false alarm count (38 episodes). Row-level metrics can be misleading; for example, Logistic Regression showed competitive hourly metrics but failed to detect one full breakdown event. Event-level evaluation was the decisive factor.

### Operational Thresholds & Trade-offs
Model predictions are mapped to three operational action levels:

| Alert Band | Probability Cutoff | Fleet Hours Share | Action Protocol |
|---|---|---|---|
| **HIGH** | $\ge 0.20$ | ~2.8% | Immediate physical inspection and parts staging |
| **WATCH** | $0.02 - 0.20$ | ~1.5% | Heightened monitoring across upcoming shifts |
| **LOW** | $< 0.02$ | ~95.7% | Standard operational baseline |

**Operational Cost Consideration:** At the 0.20 operating threshold, precision is approximately 38% (roughly 3 out of 5 alert episodes do not result in a breakdown). In industrial plant environments, this trade-off is intentional. A false alert results in a 15-minute diagnostic inspection. A missed breakdown results in hours of unplanned downtime, scrap material, and equipment damage. Prioritizing breakdown recall while limiting alerts to under 3% of fleet operating hours provides a practical balance.

**Lead-Time Metrics:**
* **Cross-Validation Test Splits:** 16-hour minimum warning lead time prior to failure.
* **Continuous Walk-Forward Historical Timeline:** 23-hour minimum warning lead time prior to failure.

### Dual-Scoring Strategy & Cold-Start Handling
* **Dual Scoring:** Real-time fleet monitoring uses the model trained on all historical observations (`models/current_scores.parquet`), while the UI activity timeline uses rolling out-of-sample walk-forward scoring (`models/historical_scores_oos.parquet`) to reflect strictly causal historical outputs.
* **Cold-Start Assets (`MCH-300`, `MCH-301`):** Assets with less than 24 hours of operating history default relative ratios to 1.0 (neutral), relying on raw telemetry boundaries. `machine_id` is excluded from model training features to allow generalization to unseen equipment, and cold-start assets are flagged with explicit confidence indicators in the user interface.

### Condition Monitoring & LLM Reasoning Layer
The platform integrates Groq runtime inference (`openai/gpt-oss-120b`) for explainability and fleet queries:
* **Grounded Machine Explanations (`src/llm_explain.py`):** The backend builds a structured numeric telemetry snapshot per machine (current readings, baseline ratios, driver percentiles). The LLM translates these ratios into concise plain-language maintenance summaries rather than static response templates. Verification tests confirm that output text reflects underlying sensor severity.
* **Two-Stage Deterministic Q&A (`src/llm_qa.py`):** Stage 1 uses deterministic regex parsing to extract entities (machine IDs, production lines, risk tiers, dates, failure history) and queries the parquet store directly. Stage 2 passes only the retrieved rows to the LLM with instructions to answer strictly from the provided records. The UI includes a provenance drawer displaying the exact records behind each response.
* **Fault Tolerance & Caching:** Dashboards, sensor charts, and fleet rankings operate independently of the LLM service. Missing or invalid API keys do not interrupt core platform functionality. Explanations are cached in memory per machine-hour, and upstream Groq rate limits (8,000 tokens/minute) are caught to return clear HTTP 429 notices rather than exposing raw server errors.

---

## 3. How to Run

### Live Deployment
The production application is deployed on Render using a multi-stage Docker container:
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
* **False Alarm Overhead:** Operating at the 0.20 decision boundary produces an approximate 38% precision rate (~3 out of 5 alert episodes are false alarms). While operationally justified by the asymmetric cost of missed failures, this false positive rate requires clear communication with maintenance teams to ensure operational trust.
* **Unvalidated Cold-Start Equipment:** With only 72 hours of operating telemetry and zero historical breakdown events, machines `MCH-300` and `MCH-301` are unvalidated against real failure signatures. Predictions on these units are marked as preliminary until assets establish sufficient individual baseline history.
* **Walk-Forward Trend Horizon:** Out-of-sample historical trend scoring begins mid-timeline because early weeks must be reserved as initial training history before rolling retraining iterations can execute.
* **Missing Data Handling:** The current causal imputation scheme handles single isolated missing hours via linear interpolation. Sustained telemetry outages exceeding 6 continuous hours are not modeled and would require dedicated sensor-fault fallbacks.
* **Process-Level Caching:** Rate limiting and explanation caching are maintained in process memory. Multi-instance horizontal scaling would require centralized caching infrastructure.

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
