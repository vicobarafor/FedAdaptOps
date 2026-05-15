.PHONY: train test lint format

train:
	python scripts/train.py --config configs/cifar10_dirichlet.yaml

test:
	pytest

lint:
	ruff check .

format:
	black .
	ruff check . --fix
