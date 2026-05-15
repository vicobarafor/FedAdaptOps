# Metrics Contract

FedAdaptOps uses explicit metric contracts so dashboards, APIs, reports, and experiment comparison tools can rely on stable columns.

## Federated round metrics

```text
round_id
selected_clients
active_clients
dropped_clients
total_active_samples
mean_client_train_loss
mean_client_train_accuracy
eval_loss
eval_accuracy
round_seconds
```

## Client round metrics

```text
round_id
client_id
num_samples
train_loss
train_accuracy
local_epochs
status
```

## Design rule

Future phases may add columns, but should not remove or rename these base columns without a migration note.
