#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.models import resnet50


def load_config(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError('Config must be a YAML mapping.')
    return cfg


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def build_model(model_name: str, device: torch.device) -> torch.nn.Module:
    if model_name != 'resnet50':
        raise ValueError(f'Unsupported model: {model_name}. Week1 baseline currently supports resnet50 only.')
    model = resnet50(weights=None)
    model.eval()
    model.to(device)
    return model


def run_one_iter(model: torch.nn.Module, device: torch.device, batch_size: int, input_shape: tuple[int, int, int]) -> dict:
    c, h, w = input_shape

    # CUDA path: use CUDA events for stage timing.
    # This avoids putting multiple explicit synchronize() calls in the hot path.
    if device.type == 'cuda':
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        e3 = torch.cuda.Event(enable_timing=True)

        e0.record()
        x = torch.randn(batch_size, c, h, w, dtype=torch.float32, device=device)
        e1.record()

        with torch.inference_mode():
            y = model(x)
        e2.record()

        _ = torch.argmax(y, dim=1)
        e3.record()

        e3.synchronize()

        preprocess_ms = float(e0.elapsed_time(e1))
        forward_ms = float(e1.elapsed_time(e2))
        postprocess_ms = float(e2.elapsed_time(e3))
        total_ms = float(e0.elapsed_time(e3))
    else:
        t0 = time.perf_counter()
        x = torch.randn(batch_size, c, h, w, dtype=torch.float32, device=device)
        t1 = time.perf_counter()

        with torch.inference_mode():
            y = model(x)
        t2 = time.perf_counter()

        _ = torch.argmax(y, dim=1)
        t3 = time.perf_counter()

        preprocess_ms = (t1 - t0) * 1000.0
        forward_ms = (t2 - t1) * 1000.0
        postprocess_ms = (t3 - t2) * 1000.0
        total_ms = (t3 - t0) * 1000.0

    return {
        'preprocess_ms': preprocess_ms,
        'forward_ms': forward_ms,
        'postprocess_ms': postprocess_ms,
        'total_ms': total_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Week1 real PyTorch baseline')
    parser.add_argument('--config', required=True, type=Path, help='Path to YAML config')
    args = parser.parse_args()

    cfg = load_config(args.config)

    exp_name = str(cfg.get('experiment_name', 'week01_baseline'))
    model_name = str(cfg.get('model', 'resnet50'))
    device_name = str(cfg.get('device', 'cpu')).lower()
    batch_size = int(cfg.get('batch_size', 1))
    warmup_iters = int(cfg.get('warmup_iters', 20))
    measure_iters = int(cfg.get('measure_iters', 100))
    num_threads = int(cfg.get('num_threads', 0))
    output_root = Path(str(cfg.get('output_dir', f'experiments/runs/week01/{exp_name}')))
    input_shape = tuple(cfg.get('input_shape', [3, 224, 224]))

    if len(input_shape) != 3:
        raise ValueError('input_shape must be [C, H, W].')

    if num_threads > 0 and device_name == 'cpu':
        torch.set_num_threads(num_threads)

    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Config asks for cuda but torch.cuda.is_available() is False.')

    device = torch.device(device_name)
    model = build_model(model_name, device)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    for _ in range(warmup_iters):
        _ = run_one_iter(model, device, batch_size, input_shape)

    totals: list[float] = []
    pre: list[float] = []
    fwd: list[float] = []
    post: list[float] = []

    for _ in range(measure_iters):
        metrics = run_one_iter(model, device, batch_size, input_shape)
        pre.append(metrics['preprocess_ms'])
        fwd.append(metrics['forward_ms'])
        post.append(metrics['postprocess_ms'])
        totals.append(metrics['total_ms'])

    p50 = percentile(totals, 50)
    p95 = percentile(totals, 95)
    throughput = (1000.0 / p50) * batch_size if p50 > 0 else 0.0

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result = {
        'mode': 'real_pytorch',
        'timing_backend': 'cuda_event' if device.type == 'cuda' else 'perf_counter',
        'experiment_name': exp_name,
        'model': model_name,
        'device': device_name,
        'batch_size': batch_size,
        'warmup_iters': warmup_iters,
        'measure_iters': measure_iters,
        'num_threads': num_threads,
        'input_shape': list(input_shape),
        'p50_latency_ms': round(p50, 4),
        'p95_latency_ms': round(p95, 4),
        'throughput_samples_per_s': round(float(throughput), 4),
        'phase_p50_ms': {
            'preprocess': round(percentile(pre, 50), 4),
            'forward': round(percentile(fwd, 50), 4),
            'postprocess': round(percentile(post, 50), 4),
        },
        'phase_p95_ms': {
            'preprocess': round(percentile(pre, 95), 4),
            'forward': round(percentile(fwd, 95), 4),
            'postprocess': round(percentile(post, 95), 4),
        },
    }

    with (run_dir / 'config_snapshot.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    with (run_dir / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    summary = (
        '# Week1 PyTorch Baseline Summary\n\n'
        f"- run_id: `{run_id}`\n"
        f"- mode: `{result['mode']}`\n"
        f"- timing_backend: `{result['timing_backend']}`\n"
        f"- device: `{device_name}`\n"
        f"- model: `{model_name}`\n"
        f"- batch_size: `{batch_size}`\n"
        f"- p50_latency_ms: `{result['p50_latency_ms']}`\n"
        f"- p95_latency_ms: `{result['p95_latency_ms']}`\n"
        f"- throughput_samples_per_s: `{result['throughput_samples_per_s']}`\n"
    )
    (run_dir / 'summary.md').write_text(summary, encoding='utf-8')

    print(f'[OK] Week1 baseline finished: {run_dir}')


if __name__ == '__main__':
    main()
