from __future__ import annotations

import argparse
import random

import torch
from omegaconf import OmegaConf

from fedadaptops.config.personalization_validation import validate_personalization_config
from fedadaptops.data.datasets import get_cifar10, make_loader
from fedadaptops.data.partitioning import dirichlet_partition
from fedadaptops.data.reports import save_partition_reports
from fedadaptops.models.registry import build_model
from fedadaptops.personalization.evaluator import PersonalizationEvaluator
from fedadaptops.tracking.artifacts import ArtifactManager
from fedadaptops.tracking.metadata import build_run_metadata
from fedadaptops.tracking.run_store import create_run_dir, write_summary
from fedadaptops.training.trainer import resolve_device
from fedadaptops.utils.config import load_config, save_config
from fedadaptops.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedAdaptOps Phase 3 personalization evaluator.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def split_indices(
    indices: list[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * validation_fraction))
    eval_indices = shuffled[:eval_size]
    train_indices = shuffled[eval_size:]
    if not train_indices:
        train_indices = eval_indices
    return train_indices, eval_indices


def main() -> None:
    args = parse_args()
    raw_cfg = load_config(args.config)
    cfg = validate_personalization_config(raw_cfg)
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
    train_ds, _ = get_cifar10(cfg.data.data_dir)
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
    selected_client_ids = sorted(partitions.keys())
    if cfg.personalization.clients_limit is not None:
        selected_client_ids = selected_client_ids[: cfg.personalization.clients_limit]
    client_loaders = {}
    for client_id in selected_client_ids:
        train_indices, eval_indices = split_indices(
            partitions[client_id],
            validation_fraction=cfg.data.validation_fraction,
            seed=cfg.seed + client_id,
        )
        train_loader = make_loader(
            train_ds,
            indices=train_indices,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
        )
        eval_loader = make_loader(
            train_ds,
            indices=eval_indices,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
        )
        client_loaders[client_id] = (train_loader, eval_loader)

    def model_factory():
        return build_model(cfg.model.name, cfg.model.num_classes)

    base_model = model_factory()
    if cfg.personalization.checkpoint_path:
        checkpoint = torch.load(cfg.personalization.checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        base_model.load_state_dict(state_dict)
    initial_state = {
        key: value.detach().cpu().clone() for key, value in base_model.state_dict().items()
    }
    evaluator = PersonalizationEvaluator(
        model_factory=model_factory,
        initial_state_dict=initial_state,
        client_loaders=client_loaders,
        policies=cfg.personalization.policies,
        device=device,
        local_epochs=cfg.personalization.local_epochs,
        lr=cfg.personalization.lr,
        weight_decay=cfg.personalization.weight_decay,
        max_train_batches=cfg.personalization.max_train_batches_per_client,
        max_eval_batches=cfg.personalization.max_eval_batches_per_client,
        run_dir=run_dir,
    )
    summary = evaluator.run()
    summary.update(
        {
            "run_id": run_id,
            "num_total_clients": cfg.data.num_clients,
            "num_evaluated_clients": len(selected_client_ids),
            "policies": cfg.personalization.policies,
            "checkpoint_path": cfg.personalization.checkpoint_path,
            "device": str(device),
            "status": "completed",
        }
    )
    artifacts.write_json("personalization_summary.json", summary)
    write_summary(run_dir, summary)
    artifacts.write_json(
        "run_metadata.json",
        build_run_metadata(run_id=run_id, config=cfg, device=str(device), status="completed"),
    )
    print(f"Personalization run complete: {run_dir}")


if __name__ == "__main__":
    main()
