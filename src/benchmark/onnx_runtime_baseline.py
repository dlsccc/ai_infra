#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import onnxruntime as ort
except Exception as e:  # pragma: no cover
    raise RuntimeError("onnxruntime is required. Please install it first.") from e


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping.")
    return cfg


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def build_session_options(cfg: dict[str, Any]) -> ort.SessionOptions:
    so = ort.SessionOptions()

    intra_threads = int(cfg.get("intra_op_num_threads", 0))
    inter_threads = int(cfg.get("inter_op_num_threads", 0))
    if intra_threads > 0:
        so.intra_op_num_threads = intra_threads
    if inter_threads > 0:
        so.inter_op_num_threads = inter_threads

    opt_level_name = str(cfg.get("graph_optimization_level", "ORT_ENABLE_ALL")).upper()
    level_map = {
        "ORT_DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "ORT_ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "ORT_ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "ORT_ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    if opt_level_name not in level_map:
        raise ValueError(f"Unsupported graph_optimization_level: {opt_level_name}")

    so.graph_optimization_level = level_map[opt_level_name]
    return so


def run_one_iter_resnet(
    sess: ort.InferenceSession,
    batch_size: int,
    c: int,
    h: int,
    w: int,
) -> dict[str, float]:
    input_name = sess.get_inputs()[0].name

    t0 = time.perf_counter()
    x = np.random.randn(batch_size, c, h, w).astype(np.float32)
    t1 = time.perf_counter()

    outputs = sess.run(None, {input_name: x})
    t2 = time.perf_counter()

    _ = np.argmax(outputs[0], axis=1)
    t3 = time.perf_counter()

    return {
        "preprocess_ms": (t1 - t0) * 1000.0,
        "forward_ms": (t2 - t1) * 1000.0,
        "postprocess_ms": (t3 - t2) * 1000.0,
        "total_ms": (t3 - t0) * 1000.0,
    }


def run_one_iter_bert(
    sess: ort.InferenceSession,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> dict[str, float]:
    input_names = [x.name for x in sess.get_inputs()]

    t0 = time.perf_counter()
    input_ids = np.random.randint(0, max(vocab_size, 1000), size=(batch_size, seq_len), dtype=np.int64)
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
    token_type_ids = np.zeros((batch_size, seq_len), dtype=np.int64)

    feeds: dict[str, np.ndarray] = {}
    if "input_ids" in input_names:
        feeds["input_ids"] = input_ids
    if "attention_mask" in input_names:
        feeds["attention_mask"] = attention_mask
    if "token_type_ids" in input_names:
        feeds["token_type_ids"] = token_type_ids
    t1 = time.perf_counter()

    outputs = sess.run(None, feeds)
    t2 = time.perf_counter()

    last_hidden_state = outputs[0]
    cls_repr = last_hidden_state[:, 0, :]
    _ = np.mean(cls_repr, axis=1)
    t3 = time.perf_counter()

    return {
        "preprocess_ms": (t1 - t0) * 1000.0,
        "forward_ms": (t2 - t1) * 1000.0,
        "postprocess_ms": (t3 - t2) * 1000.0,
        "total_ms": (t3 - t0) * 1000.0,
    }


def benchmark_resnet(cfg: dict[str, Any], sess: ort.InferenceSession) -> list[dict[str, Any]]:
    input_shape = [int(x) for x in cfg.get("input_shape", [3, 224, 224])]
    if len(input_shape) != 3:
        raise ValueError("input_shape must be [C, H, W].")

    c, default_h, default_w = input_shape
    cases = cfg.get("cases", [{"batch_size": 1}, {"batch_size": 4}, {"batch_size": 8}])
    warmup_iters = int(cfg.get("warmup_iters", 20))
    measure_iters = int(cfg.get("measure_iters", 100))

    results: list[dict[str, Any]] = []

    for case in cases:
        batch_size = int(case.get("batch_size", 1))
        h = int(case.get("height", default_h))
        w = int(case.get("width", default_w))

        for _ in range(warmup_iters):
            _ = run_one_iter_resnet(sess, batch_size, c, h, w)

        pre, fwd, post, total = [], [], [], []
        for _ in range(measure_iters):
            m = run_one_iter_resnet(sess, batch_size, c, h, w)
            pre.append(m["preprocess_ms"])
            fwd.append(m["forward_ms"])
            post.append(m["postprocess_ms"])
            total.append(m["total_ms"])

        total_p50 = percentile(total, 50)
        total_p95 = percentile(total, 95)
        throughput = (1000.0 / total_p50) * batch_size if total_p50 > 0 else 0.0

        results.append(
            {
                "batch_size": batch_size,
                "height": h,
                "width": w,
                "preprocess_p50_ms": round(percentile(pre, 50), 4),
                "preprocess_p95_ms": round(percentile(pre, 95), 4),
                "forward_p50_ms": round(percentile(fwd, 50), 4),
                "forward_p95_ms": round(percentile(fwd, 95), 4),
                "postprocess_p50_ms": round(percentile(post, 50), 4),
                "postprocess_p95_ms": round(percentile(post, 95), 4),
                "total_p50_ms": round(total_p50, 4),
                "total_p95_ms": round(total_p95, 4),
                "throughput_samples_per_s": round(float(throughput), 4),
            }
        )

    return results


def benchmark_bert(cfg: dict[str, Any], sess: ort.InferenceSession) -> list[dict[str, Any]]:
    cases = cfg.get(
        "cases",
        [
            {"batch_size": 1, "seq_len": 32},
            {"batch_size": 1, "seq_len": 128},
            {"batch_size": 4, "seq_len": 128},
            {"batch_size": 8, "seq_len": 256},
        ],
    )
    warmup_iters = int(cfg.get("warmup_iters", 20))
    measure_iters = int(cfg.get("measure_iters", 100))
    vocab_size = int(cfg.get("vocab_size", 30522))

    results: list[dict[str, Any]] = []

    for case in cases:
        batch_size = int(case.get("batch_size", 1))
        seq_len = int(case.get("seq_len", 128))

        for _ in range(warmup_iters):
            _ = run_one_iter_bert(sess, batch_size, seq_len, vocab_size)

        pre, fwd, post, total = [], [], [], []
        for _ in range(measure_iters):
            m = run_one_iter_bert(sess, batch_size, seq_len, vocab_size)
            pre.append(m["preprocess_ms"])
            fwd.append(m["forward_ms"])
            post.append(m["postprocess_ms"])
            total.append(m["total_ms"])

        total_p50 = percentile(total, 50)
        total_p95 = percentile(total, 95)
        throughput = (1000.0 / total_p50) * batch_size if total_p50 > 0 else 0.0

        results.append(
            {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "preprocess_p50_ms": round(percentile(pre, 50), 4),
                "preprocess_p95_ms": round(percentile(pre, 95), 4),
                "forward_p50_ms": round(percentile(fwd, 50), 4),
                "forward_p95_ms": round(percentile(fwd, 95), 4),
                "postprocess_p50_ms": round(percentile(post, 50), 4),
                "postprocess_p95_ms": round(percentile(post, 95), 4),
                "total_p50_ms": round(total_p50, 4),
                "total_p95_ms": round(total_p95, 4),
                "throughput_samples_per_s": round(float(throughput), 4),
            }
        )

    return results


def render_summary(model_family: str, run_id: str, cases: list[dict[str, Any]]) -> str:
    lines = [
        "# Week02 ONNX Runtime Baseline Summary",
        "",
        f"- run_id: `{run_id}`",
        f"- model_family: `{model_family}`",
        f"- cases: `{len(cases)}`",
        "",
    ]

    if model_family == "resnet":
        lines.extend(
            [
                "| batch | h | w | fwd_p50_ms | total_p50_ms | throughput(samples/s) |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for c in cases:
            lines.append(
                f"| {c['batch_size']} | {c['height']} | {c['width']} | {c['forward_p50_ms']} | {c['total_p50_ms']} | {c['throughput_samples_per_s']} |"
            )
    else:
        lines.extend(
            [
                "| batch | seq_len | fwd_p50_ms | total_p50_ms | throughput(samples/s) |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for c in cases:
            lines.append(
                f"| {c['batch_size']} | {c['seq_len']} | {c['forward_p50_ms']} | {c['total_p50_ms']} | {c['throughput_samples_per_s']} |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Week02 ONNX Runtime baseline")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    exp_name = str(cfg.get("experiment_name", "week02_ort_baseline"))
    model_family = str(cfg.get("model_family", "resnet")).lower()
    onnx_path = Path(str(cfg.get("onnx_model_path", "")))
    providers = [str(x) for x in cfg.get("providers", ["CPUExecutionProvider"])]
    output_root = Path(str(cfg.get("output_dir", f"experiments/runs/week02/ort/{exp_name}")))

    if not onnx_path.exists():
        raise FileNotFoundError(f"onnx_model_path not found: {onnx_path}")

    session_options = build_session_options(cfg)
    sess = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=providers)

    if model_family == "resnet":
        case_results = benchmark_resnet(cfg, sess)
    elif model_family == "bert":
        case_results = benchmark_bert(cfg, sess)
    else:
        raise ValueError(f"Unsupported model_family: {model_family}. Use 'resnet' or 'bert'.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    result = {
        "mode": "week02_onnx_runtime",
        "experiment_name": exp_name,
        "model_family": model_family,
        "onnx_model_path": str(onnx_path),
        "providers": providers,
        "graph_optimization_level": str(cfg.get("graph_optimization_level", "ORT_ENABLE_ALL")).upper(),
        "warmup_iters": int(cfg.get("warmup_iters", 20)),
        "measure_iters": int(cfg.get("measure_iters", 100)),
        "cases": case_results,
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    (run_dir / "summary.md").write_text(render_summary(model_family, run_id, case_results), encoding="utf-8")

    print(f"[OK] ONNX Runtime baseline finished: {run_dir}")


if __name__ == "__main__":
    main()
