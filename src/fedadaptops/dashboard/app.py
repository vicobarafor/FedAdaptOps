from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from fedadaptops.dashboard.data import (
    detect_run_type,
    latest_run_by_type,
    list_run_dirs,
    read_csv_if_exists,
    read_json_if_exists,
)

st.set_page_config(
    page_title="FedAdaptOps Dashboard",
    page_icon="🧭",
    layout="wide",
)


def _fmt_path(path: Path | None) -> str:
    return "None" if path is None else str(path)


def run_selector(label: str, run_type: str) -> Path | None:
    run_dirs = [path for path in list_run_dirs("runs") if detect_run_type(path) == run_type]
    if not run_dirs:
        st.sidebar.warning(f"No {run_type} runs found.")
        return None

    options = {path.name: path for path in run_dirs}
    default = next(iter(options))
    selected = st.sidebar.selectbox(
        label, list(options.keys()), index=list(options.keys()).index(default)
    )
    return options[selected]


def metric_card(label: str, value, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def show_overview() -> None:
    st.title("FedAdaptOps")
    st.caption("Production-style adaptive federated personalization observability dashboard")

    run_dirs = list_run_dirs("runs")
    if not run_dirs:
        st.info(
            "No runs found yet. Run training, personalization, or routing scripts to populate `runs/`."
        )
        return

    counts = {}
    for run_dir in run_dirs:
        run_type = detect_run_type(run_dir)
        counts[run_type] = counts.get(run_type, 0) + 1

    cols = st.columns(5)
    cols[0].metric("Total runs", len(run_dirs))
    cols[1].metric("FedAvg runs", counts.get("fedavg", 0))
    cols[2].metric("Personalization runs", counts.get("personalization", 0))
    cols[3].metric("Routing runs", counts.get("adaptive_routing", 0))
    cols[4].metric("Baseline runs", counts.get("centralized_baseline", 0))

    st.subheader("Recent runs")
    rows = [
        {"run_id": path.name, "type": detect_run_type(path), "path": str(path)} for path in run_dirs
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def show_fedavg(run_dir: Path | None) -> None:
    st.header("Federated Training")

    if run_dir is None:
        st.info("No FedAvg run selected.")
        return

    st.caption(f"Run: `{run_dir}`")
    round_df = read_csv_if_exists(run_dir, "federated_round_metrics.csv")
    client_df = read_csv_if_exists(run_dir, "client_round_metrics.csv")
    summary = read_json_if_exists(run_dir, "summary.json")

    if summary:
        cols = st.columns(4)
        cols[0].metric("Rounds", summary.get("num_rounds"))
        cols[1].metric("Best eval accuracy", summary.get("best_eval_accuracy"))
        cols[2].metric("Clients per round", summary.get("clients_per_round"))
        cols[3].metric("Dirichlet alpha", summary.get("dirichlet_alpha"))

    if round_df is not None:
        st.subheader("Round metrics")
        st.line_chart(round_df.set_index("round_id")[["eval_accuracy"]], use_container_width=True)
        st.dataframe(round_df, use_container_width=True)

    if client_df is not None:
        st.subheader("Client round metrics")
        status_counts = client_df["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        st.dataframe(status_counts, use_container_width=True)
        st.dataframe(client_df, use_container_width=True)


def show_personalization(run_dir: Path | None) -> None:
    st.header("Personalization")

    if run_dir is None:
        st.info("No personalization run selected.")
        return

    st.caption(f"Run: `{run_dir}`")
    df = read_csv_if_exists(run_dir, "personalization_results.csv")
    summary = read_json_if_exists(run_dir, "personalization_summary.json")

    if df is None:
        st.warning("Missing personalization_results.csv")
        return

    if summary:
        cols = st.columns(4)
        cols[0].metric(
            "Evaluated clients", summary.get("num_evaluated_clients", summary.get("num_clients"))
        )
        cols[1].metric(
            "Policies",
            (
                len(summary.get("policies", []))
                if "policies" in summary
                else summary.get("num_policies")
            ),
        )
        cols[2].metric("Best mean policy", summary.get("best_mean_policy"))
        cols[3].metric("Best mean post accuracy", summary.get("best_mean_post_accuracy"))

    st.subheader("Policy comparison")
    by_policy = (
        df.groupby("policy")
        .agg(
            mean_pre_accuracy=("pre_accuracy", "mean"),
            mean_post_accuracy=("post_accuracy", "mean"),
            mean_accuracy_delta=("accuracy_delta", "mean"),
            mean_relative_compute_cost=("relative_compute_cost", "mean"),
            mean_trainable_parameter_fraction=("trainable_parameter_fraction", "mean"),
        )
        .reset_index()
    )
    st.dataframe(by_policy, use_container_width=True)

    chart_df = by_policy.set_index("policy")[["mean_post_accuracy", "mean_accuracy_delta"]]
    st.bar_chart(chart_df, use_container_width=True)

    st.subheader("Accuracy vs cost")
    st.scatter_chart(
        df,
        x="relative_compute_cost",
        y="post_accuracy",
        color="policy",
        size="trainable_parameter_fraction",
        use_container_width=True,
    )

    st.subheader("Per-client policy metrics")
    selected_clients = st.multiselect(
        "Filter clients",
        sorted(df["client_id"].unique().tolist()),
        default=sorted(df["client_id"].unique().tolist())[:5],
    )
    filtered = df[df["client_id"].isin(selected_clients)] if selected_clients else df
    st.dataframe(filtered, use_container_width=True)


def show_routing(run_dir: Path | None) -> None:
    st.header("Adaptive Routing")

    if run_dir is None:
        st.info("No adaptive routing run selected.")
        return

    st.caption(f"Run: `{run_dir}`")

    recs = read_csv_if_exists(run_dir, "selector_recommendations.csv")
    selector_summary = read_csv_if_exists(run_dir, "selector_summary.csv")
    resources = read_csv_if_exists(run_dir, "client_resource_profiles.csv")
    headroom = read_csv_if_exists(run_dir, "oracle_headroom.csv")
    summary = read_json_if_exists(run_dir, "routing_summary.json")

    if summary:
        cols = st.columns(3)
        cols[0].metric("Clients", summary.get("num_clients"))
        cols[1].metric("Selectors", len(summary.get("selectors", [])))
        cols[2].metric("Source results", Path(summary.get("personalization_results_path", "")).name)

    if selector_summary is not None:
        st.subheader("Selector summary")
        st.dataframe(selector_summary, use_container_width=True)
        st.bar_chart(
            selector_summary.set_index("selector")[
                ["mean_expected_accuracy", "mean_feasibility_score"]
            ],
            use_container_width=True,
        )

    if headroom is not None:
        st.subheader("Oracle headroom")
        st.dataframe(headroom, use_container_width=True)
        st.bar_chart(
            headroom.set_index("selector")[["mean_oracle_headroom"]], use_container_width=True
        )

    if recs is not None:
        st.subheader("Recommendations")
        st.dataframe(recs, use_container_width=True)

        st.subheader("Expected accuracy vs expected cost")
        st.scatter_chart(
            recs,
            x="expected_cost",
            y="expected_accuracy",
            color="selector",
            size="feasibility_score",
            use_container_width=True,
        )

    if resources is not None:
        st.subheader("Client resource profiles")
        st.dataframe(resources, use_container_width=True)
        st.bar_chart(resources["resource_tier"].value_counts(), use_container_width=True)


def main() -> None:
    st.sidebar.title("FedAdaptOps")
    page = st.sidebar.radio(
        "View",
        [
            "Overview",
            "Federated Training",
            "Personalization",
            "Adaptive Routing",
        ],
    )

    fedavg_run = latest_run_by_type("fedavg")
    personalization_run = latest_run_by_type("personalization")
    routing_run = latest_run_by_type("adaptive_routing")

    st.sidebar.divider()
    st.sidebar.caption("Latest detected runs")
    st.sidebar.write(f"FedAvg: `{_fmt_path(fedavg_run)}`")
    st.sidebar.write(f"Personalization: `{_fmt_path(personalization_run)}`")
    st.sidebar.write(f"Routing: `{_fmt_path(routing_run)}`")

    st.sidebar.divider()

    if page == "Overview":
        show_overview()
    elif page == "Federated Training":
        selected = run_selector("FedAvg run", "fedavg")
        show_fedavg(selected)
    elif page == "Personalization":
        selected = run_selector("Personalization run", "personalization")
        show_personalization(selected)
    elif page == "Adaptive Routing":
        selected = run_selector("Routing run", "adaptive_routing")
        show_routing(selected)


if __name__ == "__main__":
    main()
