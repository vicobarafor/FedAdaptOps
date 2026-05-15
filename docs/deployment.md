# Deployment

FedAdaptOps supports local deployment through Docker Compose.

## Build and run

```bash
docker compose up --build
```

## Services

```text
FastAPI:    http://localhost:8000
API docs:   http://localhost:8000/docs
Dashboard:  http://localhost:8501
```

## Volumes

The compose setup mounts:

```text
./runs    -> /app/runs
./configs -> /app/configs
```

This allows the API and dashboard to inspect locally generated experiment artifacts.

## Scope

This is a production-style local deployment prototype, not a claim of cloud-scale production readiness.
