from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "experiments" / "runs" / "week06"
IMG_ROOT = ROOT / "docs" / "reports" / "images" / "week06"

BACKEND_LABELS = {
    "vllm_server": "vLLM",
    "sglang_server": "SGLang",
}
WORKLOAD_LABELS = {
    "single_tool": "Single Tool",
    "multi_tool_serial": "Multi-Tool Serial",
    "long_observation": "Long Observation",
    "plain_chat": "Plain Chat",
}
COLORS = {
    "vLLM": "#1f77b4",
    "SGLang": "#d62728",
    "Single Tool": "#2ca02c",
    "Multi-Tool Serial": "#ff7f0e",
    "Long Observation": "#9467bd",
}


def main() -> None:
    IMG_ROOT.mkdir(parents=True, exist_ok=True)

    plain_df, plain_summary_df = _load_run_group("plain_baseline")
    agent_df, agent_summary_df = _load_run_group("agent_baseline")

    _plot_plain_backend_overview(plain_summary_df)
    _plot_plain_request_scaling(plain_df)
    _plot_agent_workload_structure(agent_df)
    _plot_agent_latency_ttft(agent_df)
    _plot_agent_backend_overview(agent_summary_df)

    print(f"Wrote Week 6 figures to {IMG_ROOT}")


def _load_run_group(group_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    request_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for backend_dir in sorted((RUN_ROOT / group_name).iterdir()):
        if not backend_dir.is_dir():
            continue
        payload = json.loads((backend_dir / "results.json").read_text(encoding="utf-8"))
        backend_name = BACKEND_LABELS.get(backend_dir.name, backend_dir.name)
        summary = payload["summary_metrics"]
        summary_rows.append(
            {
                "group": group_name,
                "backend": backend_name,
                "request_count": summary["request_count"],
                "mean_latency_ms": summary["mean_latency_ms"],
                "mean_ttft_ms": summary["mean_ttft_ms"],
                "mean_tpot_ms": summary["mean_tpot_ms"],
                "tokens_per_second": summary["tokens_per_second"],
                "requests_per_second": summary["requests_per_second"],
                "total_output_tokens": summary["total_output_tokens"],
            }
        )
        for req in payload["requests"]:
            metadata = req["metadata"]
            request_rows.append(
                {
                    "group": group_name,
                    "backend": backend_name,
                    "request_id": req["request_id"],
                    "trace_id": metadata.get("trace_id"),
                    "workload_type": metadata.get("workload_type"),
                    "workload_label": WORKLOAD_LABELS.get(
                        metadata.get("workload_type", ""),
                        metadata.get("workload_type", ""),
                    ),
                    "turn": metadata.get("turn", 0),
                    "input_tokens": req["input_tokens"],
                    "output_tokens": req["output_tokens"],
                    "ttft_ms": req["ttft_ms"],
                    "total_latency_ms": req["total_latency_ms"],
                    "decode_latency_ms": req["decode_latency_ms"],
                    "tpot_ms": req["tpot_ms"],
                    "expected_output_tokens": metadata.get("expected_output_tokens"),
                    "prefix_overlap_ratio": metadata.get("prefix_overlap_ratio", 0.0),
                    "finish_reason": metadata.get("finish_reason"),
                }
            )

    request_df = pd.DataFrame(request_rows)
    summary_df = pd.DataFrame(summary_rows)
    return request_df, summary_df


def _plot_plain_backend_overview(summary_df: pd.DataFrame) -> None:
    metrics = [
        ("mean_latency_ms", "Mean Latency (ms)"),
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms/token)"),
        ("tokens_per_second", "Tokens/s"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (column, title) in zip(axes, metrics):
        values = summary_df[column].tolist()
        labels = summary_df["backend"].tolist()
        colors = [COLORS[label] for label in labels]
        ax.bar(labels, values, color=colors, width=0.55)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Week 6 Plain Baseline: Backend Overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(IMG_ROOT / "week06_plain_backend_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_plain_request_scaling(request_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    plain_df = request_df.sort_values(["backend", "input_tokens"])

    for backend_name, group in plain_df.groupby("backend"):
        color = COLORS[backend_name]
        axes[0].plot(
            group["input_tokens"],
            group["total_latency_ms"],
            marker="o",
            linewidth=2,
            color=color,
            label=backend_name,
        )
        axes[1].plot(
            group["input_tokens"],
            group["ttft_ms"],
            marker="o",
            linewidth=2,
            color=color,
            label=backend_name,
        )

    axes[0].set_title("Plain Chat: Input Tokens vs Total Latency")
    axes[0].set_xlabel("Input Tokens")
    axes[0].set_ylabel("Total Latency (ms)")
    axes[0].grid(linestyle="--", alpha=0.3)

    axes[1].set_title("Plain Chat: Input Tokens vs TTFT")
    axes[1].set_xlabel("Input Tokens")
    axes[1].set_ylabel("TTFT (ms)")
    axes[1].grid(linestyle="--", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(IMG_ROOT / "week06_plain_request_scaling.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_agent_workload_structure(request_df: pd.DataFrame) -> None:
    structure_df = (
        request_df.groupby(["workload_label", "turn"], as_index=False)[
            ["input_tokens", "prefix_overlap_ratio"]
        ]
        .mean()
        .sort_values(["workload_label", "turn"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for workload_label, group in structure_df.groupby("workload_label"):
        color = COLORS[workload_label]
        axes[0].plot(
            group["turn"],
            group["input_tokens"],
            marker="o",
            linewidth=2,
            color=color,
            label=workload_label,
        )
        axes[1].plot(
            group["turn"],
            group["prefix_overlap_ratio"],
            marker="o",
            linewidth=2,
            color=color,
            label=workload_label,
        )

    axes[0].set_title("Agent Workloads: Input Tokens by Turn")
    axes[0].set_xlabel("Turn")
    axes[0].set_ylabel("Input Tokens")
    axes[0].grid(linestyle="--", alpha=0.3)

    axes[1].set_title("Agent Workloads: Prefix Overlap Ratio by Turn")
    axes[1].set_xlabel("Turn")
    axes[1].set_ylabel("Prefix Overlap Ratio")
    axes[1].grid(linestyle="--", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(IMG_ROOT / "week06_agent_workload_structure.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_agent_latency_ttft(request_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    backend_order = ["vLLM", "SGLang"]
    workload_order = ["Single Tool", "Multi-Tool Serial", "Long Observation"]

    for col_idx, backend_name in enumerate(backend_order):
        backend_df = request_df[request_df["backend"] == backend_name]
        for workload_label in workload_order:
            group = backend_df[backend_df["workload_label"] == workload_label].sort_values("turn")
            color = COLORS[workload_label]
            axes[0, col_idx].plot(
                group["turn"],
                group["total_latency_ms"],
                marker="o",
                linewidth=2,
                color=color,
                label=workload_label,
            )
            axes[1, col_idx].plot(
                group["turn"],
                group["ttft_ms"],
                marker="o",
                linewidth=2,
                color=color,
                label=workload_label,
            )

        axes[0, col_idx].set_title(f"{backend_name}: Total Latency by Turn")
        axes[0, col_idx].set_ylabel("Total Latency (ms)")
        axes[0, col_idx].grid(linestyle="--", alpha=0.3)

        axes[1, col_idx].set_title(f"{backend_name}: TTFT by Turn")
        axes[1, col_idx].set_xlabel("Turn")
        axes[1, col_idx].set_ylabel("TTFT (ms)")
        axes[1, col_idx].grid(linestyle="--", alpha=0.3)

    axes[0, 1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(IMG_ROOT / "week06_agent_latency_ttft.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_agent_backend_overview(summary_df: pd.DataFrame) -> None:
    metrics = [
        ("mean_latency_ms", "Mean Latency (ms)"),
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms/token)"),
        ("tokens_per_second", "Tokens/s"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (column, title) in zip(axes, metrics):
        values = summary_df[column].tolist()
        labels = summary_df["backend"].tolist()
        colors = [COLORS[label] for label in labels]
        ax.bar(labels, values, color=colors, width=0.55)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Week 6 Agent Baseline: Backend Overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(IMG_ROOT / "week06_agent_backend_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
