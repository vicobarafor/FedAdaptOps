# Federated Training Engine

Phase 2 introduces the FedAvg training engine.

## Components

```text
clients/FederatedClient
training/ClientSampler
training/sample_weighted_average
training/FederatedTrainer
scripts/train_fedavg.py
```

## Artifact contract

A FedAvg run writes:

```text
runs/<run_id>/
  config.yaml
  environment.json
  run_metadata.json
  client_partitions.json
  partition_summary.csv
  partition_summary_records.json
  federated_round_metrics.csv
  client_round_metrics.csv
  selected_clients.json
  summary.json
  checkpoints/global_round_best.pt
```

## Design intent

This is structured as platform infrastructure: clients are explicit trainable units, sampling/dropout is isolated from training logic, aggregation is separately tested, metrics are dashboard/API-ready, and global checkpoints use stable naming.
```
