# Reproducibility

FedAdaptOps treats reproducibility as infrastructure, not an afterthought.

Current guarantees:

- explicit seed control for Python, NumPy, and PyTorch
- persisted YAML config for every run
- run-specific artifact directory
- environment metadata capture
- deterministic client partitioning
- checkpoint payload includes epoch and metrics

Known limitations:

- exact GPU determinism can still vary by hardware, driver, and PyTorch/CUDA versions
- CIFAR-10 download source availability depends on torchvision mirrors
