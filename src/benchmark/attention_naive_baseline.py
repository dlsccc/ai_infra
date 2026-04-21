#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml


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


def parse_dtype(name: str) -> torch.dtype:
    key = name.strip().lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def build_qkv(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch_size, num_heads, seq_len, head_dim)
    q = torch.randn(*shape, device=device, dtype=dtype)
    k = torch.randn(*shape, device=device, dtype=dtype)
    v = torch.randn(*shape, device=device, dtype=dtype)
    return q, k, v


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if is_causal:
        seq_len = q.size(-2)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)


def run_impl(impl: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    if impl == "naive":
        return naive_attention(q, k, v, is_causal=is_causal)
    if impl == "sdpa":
        return sdpa_attention(q, k, v, is_causal=is_causal)
    raise ValueError(f"Unsupported implementation: {impl}")


def run_one_iter(
    impl: str,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    is_causal: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        e3 = torch.cuda.Event(enable_timing=True)

        e0.record()
        q, k, v = build_qkv(batch_size, num_heads, seq_len, head_dim, device, dtype)
        e1.record()

        with torch.inference_mode():
            out = run_impl(impl, q, k, v, is_causal=is_causal)
        e2.record()

        _ = torch.mean(out, dim=(-1, -2, -3))
        e3.record()
        e3.synchronize()

        peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        return {
            "preprocess_ms": float(e0.elapsed_time(e1)),
            "forward_ms": float(e1.elapsed_time(e2)),
            "postprocess_ms": float(e2.elapsed_time(e3)),
            "total_ms": float(e0.elapsed_time(e3)),
            "peak_memory_mb": peak_mem_mb,
        }

    t0 = time.perf_counter()
    q, k, v = build_qkv(batch_size, num_heads, seq_len, head_dim, device, dtype)
    t1 = time.perf_counter()

    with torch.inference_mode():
        out = run_impl(impl, q, k, v, is_causal=is_causal)
    t2 = time.perf_counter()

    _ = torch.mean(out, dim=(-1, -2, -3))
    t3 = time.perf_counter()

    return {
        "preprocess_ms": (t1 - t0) * 1000.0,
        "forward_ms": (t2 - t1) * 1000.0,
        "postprocess_ms": (t3 - t2) * 1000.0,
        "total_ms": (t3 - t0) * 1000.0,
        "peak_memory_mb": 0.0,
    }


def measure_correctness(
    impl: str,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    is_causal: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float] | None:
    if impl == "naive":
        return None

    q, k, v = build_qkv(batch_size, num_heads, seq_len, head_dim, device, dtype)
    with torch.inference_mode():
        ref = naive_attention(q, k, v, is_causal=is_causal)
        out = run_impl(impl, q, k, v, is_causal=is_causal)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    err = (out - ref).abs()
    return {
        "max_abs_err": float(err.max().item()),
        "mean_abs_err": float(err.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Week03 naive attention baseline")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    exp_name = str(cfg.get("experiment_name", "week03_attention_baseline"))
    device_name = str(cfg.get("device", "cpu")).lower()
    dtype_name = str(cfg.get("dtype", "float32"))
    batch_size = int(cfg.get("batch_size", 1))
    num_heads = int(cfg.get("num_heads", 8))
    head_dim = int(cfg.get("head_dim", 64))
    seq_lens = [int(x) for x in cfg.get("seq_lens", [64, 128, 256, 512])]
    implementations = [str(x).lower() for x in cfg.get("implementations", ["naive", "sdpa"])]
    is_causal = bool(cfg.get("is_causal", False))
    warmup_iters = int(cfg.get("warmup_iters", 10))
    measure_iters = int(cfg.get("measure_iters", 50))
    seed = int(cfg.get("seed", 42))
    check_correctness = bool(cfg.get("check_correctness", True))
    output_root = Path(str(cfg.get("output_dir", f"experiments/runs/week03/{exp_name}")))

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config asks for cuda but torch.cuda.is_available() is False.")

    device = torch.device(device_name)
    dtype = parse_dtype(dtype_name)

    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        raise ValueError("CPU baseline currently expects float32 for numerical stability.")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize(device)

    case_results: list[dict[str, Any]] = []

    for impl in implementations:
        for seq_len in seq_lens:
            for _ in range(warmup_iters):
                _ = run_one_iter(
                    impl=impl,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    seq_len=seq_len,
                    head_dim=head_dim,
                    is_causal=is_causal,
                    device=device,
                    dtype=dtype,
                )

            pre, fwd, post, total, mem = [], [], [], [], []
            for _ in range(measure_iters):
                m = run_one_iter(
                    impl=impl,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    seq_len=seq_len,
                    head_dim=head_dim,
                    is_causal=is_causal,
                    device=device,
                    dtype=dtype,
                )
                pre.append(m["preprocess_ms"])
                fwd.append(m["forward_ms"])
                post.append(m["postprocess_ms"])
                total.append(m["total_ms"])
                mem.append(m["peak_memory_mb"])

            total_p50 = percentile(total, 50)
            throughput_tokens_per_s = (batch_size * seq_len * 1000.0 / total_p50) if total_p50 > 0 else 0.0

            record: dict[str, Any] = {
                "implementation": impl,
                "batch_size": batch_size,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "seq_len": seq_len,
                "preprocess_p50_ms": round(percentile(pre, 50), 4),
                "preprocess_p95_ms": round(percentile(pre, 95), 4),
                "forward_p50_ms": round(percentile(fwd, 50), 4),
                "forward_p95_ms": round(percentile(fwd, 95), 4),
                "postprocess_p50_ms": round(percentile(post, 50), 4),
                "postprocess_p95_ms": round(percentile(post, 95), 4),
                "total_p50_ms": round(total_p50, 4),
                "total_p95_ms": round(percentile(total, 95), 4),
                "peak_memory_p50_mb": round(percentile(mem, 50), 4),
                "peak_memory_p95_mb": round(percentile(mem, 95), 4),
                "throughput_tokens_per_s": round(float(throughput_tokens_per_s), 4),
            }

            if check_correctness:
                err = measure_correctness(
                    impl=impl,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    seq_len=seq_len,
                    head_dim=head_dim,
                    is_causal=is_causal,
                    device=device,
                    dtype=dtype,
                )
                if err is not None:
                    record["max_abs_err_vs_naive"] = round(err["max_abs_err"], 6)
                    record["mean_abs_err_vs_naive"] = round(err["mean_abs_err"], 6)

            case_results.append(record)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "mode": "week03_attention_baseline",
        "experiment_name": exp_name,
        "device": device_name,
        "dtype": dtype_name,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "seq_lens": seq_lens,
        "implementations": implementations,
        "is_causal": is_causal,
        "warmup_iters": warmup_iters,
        "measure_iters": measure_iters,
        "check_correctness": check_correctness,
        "cases": case_results,
    }

    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    lines = [
        "# Week03 Attention Baseline Summary",
        "",
        f"- run_id: `{run_id}`",
        f"- device: `{device_name}`",
        f"- dtype: `{dtype_name}`",
        f"- batch_size: `{batch_size}`",
        f"- heads: `{num_heads}`",
        f"- head_dim: `{head_dim}`",
        f"- cases: `{len(case_results)}`",
        "",
        "| impl | seq_len | fwd_p50_ms | total_p50_ms | peak_mem_p50_mb | throughput_tokens/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for c in case_results:
        lines.append(
            f"| {c['implementation']} | {c['seq_len']} | {c['forward_p50_ms']} | {c['total_p50_ms']} | {c['peak_memory_p50_mb']} | {c['throughput_tokens_per_s']} |"
        )

    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] Week03 attention baseline finished: {run_dir}")


if __name__ == "__main__":
    main()
