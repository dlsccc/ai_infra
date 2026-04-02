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
from transformers import AutoModel, AutoTokenizer


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


def build_text_batch(batch_size: int, approx_words: int) -> list[str]:
    # Use repeated simple words to build deterministic synthetic text.
    text = ' '.join(['hello'] * max(approx_words, 8))
    return [text] * batch_size


def run_one_iter(
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    approx_words: int,
) -> dict:
    texts = build_text_batch(batch_size, approx_words)

    # CPU path: pure wall-time timing.
    if device.type != 'cuda':
        t0 = time.perf_counter()
        encoded = tokenizer(
            texts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=seq_len,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        t1 = time.perf_counter()

        with torch.inference_mode():
            outputs = model(**encoded)
        t2 = time.perf_counter()

        cls_repr = outputs.last_hidden_state[:, 0, :]
        _ = torch.mean(cls_repr, dim=1)
        t3 = time.perf_counter()

        return {
            'tokenization_ms': (t1 - t0) * 1000.0,
            'forward_ms': (t2 - t1) * 1000.0,
            'postprocess_ms': (t3 - t2) * 1000.0,
            'total_ms': (t3 - t0) * 1000.0,
        }

    # CUDA path (mixed timing):
    # - tokenization is CPU-bound, timed by wall-time (perf_counter)
    # - forward/postprocess are GPU-bound, timed by cuda.Event
    t0 = time.perf_counter()
    encoded = tokenizer(
        texts,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=seq_len,
    )
    encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    e1 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)
    e3 = torch.cuda.Event(enable_timing=True)

    e1.record()
    with torch.inference_mode():
        outputs = model(**encoded)
    e2.record()

    cls_repr = outputs.last_hidden_state[:, 0, :]
    _ = torch.mean(cls_repr, dim=1)
    e3.record()

    e3.synchronize()
    t3 = time.perf_counter()

    forward_ms = float(e1.elapsed_time(e2))
    postprocess_ms = float(e2.elapsed_time(e3))

    return {
        'tokenization_ms': (t1 - t0) * 1000.0,
        'forward_ms': forward_ms,
        'postprocess_ms': postprocess_ms,
        'total_ms': (t3 - t0) * 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Week1 BERT tokenization baseline')
    parser.add_argument('--config', required=True, type=Path, help='Path to YAML config')
    args = parser.parse_args()

    cfg = load_config(args.config)

    exp_name = str(cfg.get('experiment_name', 'week01_bert_tokenization'))
    model_name = str(cfg.get('model_name', 'bert-base-uncased'))
    device_name = str(cfg.get('device', 'cpu')).lower()
    batch_sizes = [int(x) for x in cfg.get('batch_sizes', [1, 4, 8])]
    seq_lens = [int(x) for x in cfg.get('seq_lens', [32, 128, 256])]
    warmup_iters = int(cfg.get('warmup_iters', 20))
    measure_iters = int(cfg.get('measure_iters', 100))
    approx_words = int(cfg.get('approx_words', 256))
    output_root = Path(str(cfg.get('output_dir', f'experiments/runs/week01/{exp_name}')))

    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Config asks for cuda but torch.cuda.is_available() is False.')

    device = torch.device(device_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    case_results: list[dict] = []

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            for _ in range(warmup_iters):
                _ = run_one_iter(tokenizer, model, device, batch_size, seq_len, approx_words)

            tok_times: list[float] = []
            fwd_times: list[float] = []
            post_times: list[float] = []
            total_times: list[float] = []

            for _ in range(measure_iters):
                m = run_one_iter(tokenizer, model, device, batch_size, seq_len, approx_words)
                tok_times.append(m['tokenization_ms'])
                fwd_times.append(m['forward_ms'])
                post_times.append(m['postprocess_ms'])
                total_times.append(m['total_ms'])

            tok_p50 = percentile(tok_times, 50)
            fwd_p50 = percentile(fwd_times, 50)
            post_p50 = percentile(post_times, 50)
            total_p50 = percentile(total_times, 50)

            tok_p95 = percentile(tok_times, 95)
            fwd_p95 = percentile(fwd_times, 95)
            post_p95 = percentile(post_times, 95)
            total_p95 = percentile(total_times, 95)

            tok_ratio = (tok_p50 / total_p50) if total_p50 > 0 else 0.0
            throughput = (1000.0 / total_p50) * batch_size if total_p50 > 0 else 0.0

            case_results.append(
                {
                    'device': device_name,
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'tokenization_p50_ms': round(tok_p50, 4),
                    'tokenization_p95_ms': round(tok_p95, 4),
                    'forward_p50_ms': round(fwd_p50, 4),
                    'forward_p95_ms': round(fwd_p95, 4),
                    'postprocess_p50_ms': round(post_p50, 4),
                    'postprocess_p95_ms': round(post_p95, 4),
                    'total_p50_ms': round(total_p50, 4),
                    'total_p95_ms': round(total_p95, 4),
                    'tokenization_ratio_p50': round(tok_ratio, 4),
                    'throughput_samples_per_s': round(float(throughput), 4),
                }
            )

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result = {
        'mode': 'week01_bert_tokenization',
        'timing_backend': 'mixed_cuda_event' if device.type == 'cuda' else 'perf_counter',
        'experiment_name': exp_name,
        'model_name': model_name,
        'device': device_name,
        'batch_sizes': batch_sizes,
        'seq_lens': seq_lens,
        'warmup_iters': warmup_iters,
        'measure_iters': measure_iters,
        'approx_words': approx_words,
        'cases': case_results,
    }

    with (run_dir / 'config_snapshot.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    with (run_dir / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    lines = [
        '# Week1 BERT Tokenization Baseline Summary',
        '',
        f'- run_id: `{run_id}`',
        f'- timing_backend: `{result["timing_backend"]}`',
        f'- model: `{model_name}`',
        f'- device: `{device_name}`',
        f'- cases: `{len(case_results)}`',
        '',
        '| batch | seq_len | tok_p50_ms | fwd_p50_ms | total_p50_ms | tok_ratio | throughput(samples/s) |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for c in case_results:
        lines.append(
            f"| {c['batch_size']} | {c['seq_len']} | {c['tokenization_p50_ms']} | {c['forward_p50_ms']} | {c['total_p50_ms']} | {c['tokenization_ratio_p50']} | {c['throughput_samples_per_s']} |"
        )

    (run_dir / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(f'[OK] BERT tokenization baseline finished: {run_dir}')


if __name__ == '__main__':
    main()
