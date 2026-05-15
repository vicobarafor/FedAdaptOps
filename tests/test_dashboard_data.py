from pathlib import Path

from fedadaptops.dashboard.data import detect_run_type, list_run_dirs


def test_detect_run_type_for_routing_run(tmp_path: Path):
    run_dir = tmp_path / "routing_run"
    run_dir.mkdir()
    (run_dir / "selector_recommendations.csv").write_text("client_id,selector\n", encoding="utf-8")

    assert detect_run_type(run_dir) == "adaptive_routing"


def test_detect_run_type_for_personalization_run(tmp_path: Path):
    run_dir = tmp_path / "personalization_run"
    run_dir.mkdir()
    (run_dir / "personalization_results.csv").write_text("client_id,policy\n", encoding="utf-8")

    assert detect_run_type(run_dir) == "personalization"


def test_detect_run_type_for_fedavg_run(tmp_path: Path):
    run_dir = tmp_path / "fedavg_run"
    run_dir.mkdir()
    (run_dir / "federated_round_metrics.csv").write_text(
        "round_id,eval_accuracy\n", encoding="utf-8"
    )

    assert detect_run_type(run_dir) == "fedavg"


def test_list_run_dirs_sorts_descending(tmp_path: Path):
    (tmp_path / "b_run").mkdir()
    (tmp_path / "a_run").mkdir()

    runs = list_run_dirs(tmp_path)

    assert [path.name for path in runs] == ["b_run", "a_run"]
