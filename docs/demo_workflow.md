# Demo Workflow

This workflow demonstrates the full FedAdaptOps pipeline.

## 1. Install

```bash
pip install -e ".[dev]"
```

## 2. Run tests

```bash
pytest
```

## 3. Train FedAvg

```bash
python scripts/train_fedavg.py --config configs/cifar10_fedavg.yaml
```

## 4. Personalize clients

Edit `configs/cifar10_personalization.yaml` and optionally set:

```yaml
personalization:
  checkpoint_path: runs/<fedavg_run_id>/checkpoints/global_round_best.pt
```

Run:

```bash
python scripts/personalize.py --config configs/cifar10_personalization.yaml
```

## 5. Recommend policies

Edit `configs/cifar10_routing.yaml`:

```yaml
routing:
  personalization_results_path: runs/<personalization_run_id>/personalization_results.csv
```

Run:

```bash
python scripts/recommend_policies.py --config configs/cifar10_routing.yaml
```

## 6. Launch dashboard

```bash
python scripts/launch_dashboard.py
```

Open:

```text
http://localhost:8501
```

## 7. Launch API

```bash
python scripts/serve_api.py
```

Open:

```text
http://localhost:8000/docs
```

## 8. Docker deployment

```bash
docker compose up --build
```

## What to show in a demo

1. `pytest` passing
2. one FedAvg run directory
3. one personalization run directory
4. one routing run directory
5. dashboard Overview page
6. dashboard Adaptive Routing page
7. FastAPI `/docs`
8. `GET /runs`
9. `POST /selector/recommend`
