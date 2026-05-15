# FedAdaptOps Sample Run

This directory is reserved for curated example reports that can be committed without large generated artifacts.

Real experiment reports are generated under:

```text
runs/<run_id>/reports/sample_run.md
```

Generate one with:

```bash
python scripts/generate_report.py --run-dir runs/<run_id>
```

Do not commit full `runs/` directories. Keep only small curated examples in this folder.
