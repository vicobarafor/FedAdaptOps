from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from fedadaptops.config.routing_validation import validate_routing_config
from fedadaptops.selectors.recommendation_engine import PolicyRecommendationEngine
from fedadaptops.tracking.artifacts import ArtifactManager
from fedadaptops.tracking.metadata import build_run_metadata
from fedadaptops.tracking.run_store import create_run_dir, write_summary
from fedadaptops.utils.config import load_config, save_config
from fedadaptops.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedAdaptOps Phase 4 adaptive policy recommendation."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_cfg = load_config(args.config)
    cfg = validate_routing_config(raw_cfg)
    seed_everything(cfg.seed)

    run_id, run_dir = create_run_dir(cfg.project.output_dir, cfg.project.run_name)
    artifacts = ArtifactManager(run_dir)

    save_config(OmegaConf.create(cfg), run_dir / "config.yaml")
    artifacts.write_json(
        "run_metadata.json",
        build_run_metadata(run_id=run_id, config=cfg, device="routing-only"),
    )
    if cfg.artifacts.save_environment:
        artifacts.capture_environment()

    engine = PolicyRecommendationEngine(cfg=cfg, run_dir=run_dir)
    summary = engine.run()
    summary.update(
        {
            "run_id": run_id,
            "personalization_results_path": cfg.routing.personalization_results_path,
            "status": "completed",
        }
    )

    artifacts.write_json("routing_summary.json", summary)
    write_summary(run_dir, summary)
    artifacts.write_json(
        "run_metadata.json",
        build_run_metadata(run_id=run_id, config=cfg, device="routing-only", status="completed"),
    )

    print(f"Adaptive routing run complete: {run_dir}")


if __name__ == "__main__":
    main()
