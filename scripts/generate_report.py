from __future__ import annotations

import argparse

from fedadaptops.evaluation.reporting import generate_federated_run_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a markdown report for a FedAdaptOps run."
    )
    parser.add_argument("--run-dir", required=True, help="Path to a run directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = generate_federated_run_report(args.run_dir)
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
