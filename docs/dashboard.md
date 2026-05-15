# Streamlit Dashboard

Phase 5 introduces a local observability dashboard for FedAdaptOps run artifacts.

## Launch

```bash
python scripts/launch_dashboard.py
```

or:

```bash
streamlit run src/fedadaptops/dashboard/app.py
```

## Views

```text
Overview
Federated Training
Personalization
Adaptive Routing
```

## Data source

The dashboard reads local run artifacts under:

```text
runs/<run_id>/
```

It does not require a database. This keeps Phase 5 lightweight while preserving a clean path toward a future FastAPI run registry.

## Current visualizations

- run inventory
- FedAvg round metrics
- client-level FL metrics
- personalization policy comparisons
- accuracy versus personalization cost
- client resource profiles
- selector recommendations
- selector summary
- oracle headroom
