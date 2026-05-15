# Contributing

FedAdaptOps is currently a portfolio/research engineering project, but contributions should follow production-style hygiene.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Quality checks

```bash
pytest
ruff check .
black --check .
```

## Formatting

```bash
black .
ruff check . --fix
```

## Artifact policy

Do not commit:

```text
data/
runs/
.venv/
*.pt
*.pth
```
