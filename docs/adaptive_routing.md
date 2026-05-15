# Adaptive Routing and Policy Selection

Phase 4 introduces selector infrastructure for choosing personalization policies under heterogeneous client constraints.

## Selectors

```text
metadata
resource_aware
oracle
```

## Selector semantics

- `metadata`: chooses a policy from resource tier.
- `resource_aware`: optimizes observed personalization accuracy against compute, latency, memory, and bandwidth costs.
- `oracle`: chooses the best observed policy per client and acts as an upper-bound comparator.

## Runnable command

First run Phase 3 personalization and locate:

```text
runs/<personalization_run_id>/personalization_results.csv
```

Then edit:

```yaml
routing:
  personalization_results_path: runs/<personalization_run_id>/personalization_results.csv
```

Run:

```bash
python scripts/recommend_policies.py --config configs/cifar10_routing.yaml
```

## Artifact contract

```text
runs/<run_id>/
  client_resource_profiles.csv
  selector_recommendations.csv
  selector_summary.csv
  oracle_headroom.csv
  routing_summary.json
  summary.json
```

This is the bridge toward the Streamlit dashboard and FastAPI recommendation endpoint.
