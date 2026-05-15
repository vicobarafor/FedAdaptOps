from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from fedadaptops.clients.client import FederatedClient
from fedadaptops.config.federated_validation import validate_federated_config
from fedadaptops.data.datasets import get_cifar10, make_loader
from fedadaptops.data.partitioning import dirichlet_partition
from fedadaptops.data.reports import save_partition_reports
from fedadaptops.models.registry import build_model
from fedadaptops.tracking.artifacts import ArtifactManager
from fedadaptops.tracking.metadata import build_run_metadata
from fedadaptops.tracking.run_store import create_run_dir, write_summary
from fedadaptops.training.federated_trainer import FederatedTrainer
from fedadaptops.training.trainer import resolve_device
from fedadaptops.utils.config import load_config, save_config
from fedadaptops.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedAdaptOps Phase 2 FedAvg trainer.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_cfg = load_config(args.config)
    cfg = validate_federated_config(raw_cfg)
    seed_everything(cfg.seed)
    run_id, run_dir = create_run_dir(cfg.project.output_dir, cfg.project.run_name)
    artifacts = ArtifactManager(run_dir)
    save_config(OmegaConf.create(cfg), run_dir / "config.yaml")
    device = resolve_device(cfg.training.device)
    artifacts.write_json(
        "run_metadata.json", build_run_metadata(run_id=run_id, config=cfg, device=str(device))
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
    clients = []
    for client_id, indices in partitions.items():
        loader = make_loader(train_ds, indices, cfg.data.batch_size, True, cfg.data.num_workers)
        clients.append(FederatedClient(client_id=client_id, train_loader=loader))
    test_loader = make_loader(test_ds, None, cfg.data.batch_size, False, cfg.data.num_workers)

    def model_factory():
        return build_model(cfg.model.name, cfg.model.num_classes)

    trainer = FederatedTrainer(
        global_model=model_factory(),
        client_model_factory=model_factory,
        clients=clients,
        test_loader=test_loader,
        device=device,
        num_rounds=cfg.federation.num_rounds,
        clients_per_round=cfg.federation.clients_per_round,
        dropout_probability=cfg.federation.dropout_probability,
        local_epochs=cfg.client_training.local_epochs,
        lr=cfg.client_training.lr,
        weight_decay=cfg.client_training.weight_decay,
        max_batches_per_client=cfg.client_training.max_batches_per_client,
        eval_every_round=cfg.evaluation.eval_every_round,
        max_eval_batches=cfg.evaluation.max_eval_batches,
        seed=cfg.seed,
        run_dir=run_dir,
        checkpoint_enabled=cfg.artifacts.save_checkpoint,
    )
    summary = trainer.fit()
    summary.update(
        {
            "run_id": run_id,
            "num_clients": cfg.data.num_clients,
            "clients_per_round": cfg.federation.clients_per_round,
            "dirichlet_alpha": cfg.data.dirichlet_alpha,
            "device": str(device),
            "status": "completed",
        }
    )
    artifacts.write_json("selected_clients.json", trainer.selections)
    write_summary(run_dir, summary)
    artifacts.write_json(
        "run_metadata.json",
        build_run_metadata(run_id=run_id, config=cfg, device=str(device), status="completed"),
    )
    print(f"FedAvg run complete: {run_dir}")


if __name__ == "__main__":
    main()
