# Multi-stage build: the frontend needs Node, the app needs Python, and
# pinning both in one Dockerfile removes any ambiguity about what Render's
# native build image happens to include. Versions match local dev
# (Python 3.13, Node 24) so "works on my machine" stays true in production.

# ---- stage 1: build the React SPA ----
FROM node:24-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- stage 2: the actual service ----
FROM python:3.13-slim AS runtime
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the artifacts committed to the repo (the trained
# model and the walk-forward scores) — nothing is retrained at boot.
COPY backend/ ./backend/
COPY src/ ./src/
COPY models/ ./models/
COPY project2_manufacturing_sensors.csv ./

# Built SPA from stage 1, served by FastAPI from the same origin as /api/*.
COPY --from=frontend /app/frontend/dist ./frontend/dist

EXPOSE 8000

# GROQ_API_KEY is supplied at runtime via Render's environment variables —
# never baked into the image. python-dotenv's load_dotenv() is a no-op when
# no .env file exists, which is the case here by design.
#
# Shell form (not exec form) so $PORT expands — Render assigns this at
# runtime and the container must bind to it, not to a hardcoded port.
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
