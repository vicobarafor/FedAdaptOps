# Personalization Engine

Phase 3 introduces client-level personalization policies.

## Supported policies

```text
head_only
partial_finetune
full_finetune
```

## Policy semantics

- `head_only`: freezes the feature extractor and adapts the classifier head.
- `partial_finetune`: freezes early layers and adapts the final feature block plus classifier.
- `full_finetune`: adapts the entire model.

## Runnable command

```bash
python scripts/personalize.py --config configs/cifar10_personalization.yaml
```

## Optional checkpoint use

To personalize from a trained FedAvg checkpoint, set:

```yaml
personalization:
  checkpoint_path: runs/<fedavg_run_id>/checkpoints/global_round_best.pt
```

If no checkpoint is provided, the script uses the initialized model. This is useful for fast smoke tests, but serious experiments should use a trained global checkpoint.

## Artifact contract

```text
runs/<run_id>/
  personalization_results.csv
  client_policy_metrics.csv
  personalization_summary.json
  summary.json
```

`client_policy_metrics.csv` is intentionally selector-ready for Phase 4 adaptive routing.
