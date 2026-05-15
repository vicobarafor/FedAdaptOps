# FedAdaptOps API Reference

Phase 6 introduces a FastAPI service for run inspection and selector recommendations.

## Launch

```bash
python scripts/serve_api.py
```

or:

```bash
uvicorn fedadaptops.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

```text
http://localhost:8000/docs
```

## Endpoints

```text
GET  /health
GET  /runs
GET  /runs/{run_id}
GET  /runs/{run_id}/metrics
GET  /runs/{run_id}/metrics?filename=<metric_file.csv>
GET  /clients/{client_id}
POST /selector/recommend
```

## Deployment

Local Docker deployment:

```bash
docker compose up --build
```

Services:

```text
API:        http://localhost:8000
Dashboard: http://localhost:8501
```

## Positioning

The API is intentionally lightweight. It is not claiming big-tech production scale, but it demonstrates production-style interfaces around reproducible ML experiment artifacts.
