from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

from fedadaptops.config.validation import validate_config
from fedadaptops.data.datasets import get_cifar10, make_loader
from fedadaptops.data.partitioning import dirichlet_partition
from fedadaptops.data.reports import save_partition_reports
from fedadaptops.models.registry import build_model
from fedadaptops.tracking.artifacts import ArtifactManager
from fedadaptops.tracking.checkpointing import CheckpointManager
from fedadaptops.tracking.logger import MetricsLogger
from fedadaptops.tracking.metadata import build_run_metadata
from fedadaptops.tracking.run_store import create_run_dir, write_summary
from fedadaptops.training.trainer import evaluate, resolve_device, train_one_epoch
from fedadaptops.utils.config import load_config, save_config
from fedadaptops.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedAdaptOps Phase 1 baseline trainer.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_cfg = load_config(args.config)
    cfg = validate_config(raw_cfg)
    seed_everything(cfg.seed)

    run_id, run_dir = create_run_dir(cfg.project.output_dir, cfg.project.run_name)
    artifacts = ArtifactManager(run_dir)

    save_config(OmegaConf.create(cfg), run_dir / "config.yaml")

    device = resolve_device(cfg.training.device)
    artifacts.write_json(
        "run_metadata.json",
        build_run_metadata(run_id=run_id, config=cfg, device=str(device)),
    )

    if cfg.artifacts.save_environment:
        artifacts.capture_environment()

    train_ds, test_ds = get_cifar10(cfg.data.data_dir)
    labels = [int(y) for _, y in train_ds]

    partitions = dirichlet_partition(
        labels=labels,
        num_clients=cfg.data.num_clients,
        alpha=cfg.data.dirichlet_alpha,
        seed=cfg.seed,
        min_samples_per_client=cfg.data.min_samples_per_client,
    )

    artifacts.write_json("client_partitions.json", {str(k): v for k, v in partitions.items()})
    save_partition_reports(labels, partitions, run_dir)

    all_train_indices = [idx for indices in partitions.values() for idx in indices]
    train_loader = make_loader(
        train_ds,
        all_train_indices,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    test_loader = make_loader(
        test_ds,
        indices=None,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    model = build_model(cfg.model.name, cfg.model.num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    logger = MetricsLogger(run_dir)
    checkpoint_manager = CheckpointManager(
        run_dir / "checkpoints", monitor="eval_accuracy", mode="max"
    )

    for epoch in range(1, cfg.training.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, cfg.training.max_train_batches
        )
        eval_metrics = evaluate(model, test_loader, device, cfg.training.max_eval_batches)

        row = {
            "run_id": run_id,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "eval_loss": eval_metrics["loss"],
            "eval_accuracy": eval_metrics["accuracy"],
            "device": str(device),
        }
        logger.log(row)
        print(row)

        if cfg.artifacts.save_checkpoint:
            checkpoint_manager.maybe_save_best(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=row,
                extra={"run_id": run_id},
            )

    best = checkpoint_manager.best_record
    summary = {
        "run_id": run_id,
        "best_eval_accuracy": best.metric_value if best else None,
        "best_checkpoint": best.path if best else None,
        "num_clients": cfg.data.num_clients,
        "dirichlet_alpha": cfg.data.dirichlet_alpha,
        "device": str(device),
        "status": "completed",
    }
    write_summary(run_dir, summary)
    artifacts.write_json(
        "run_metadata.json",
        build_run_metadata(
            run_id=run_id,
            config=cfg,
            device=str(device),
            status="completed",
        ),
    )
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
