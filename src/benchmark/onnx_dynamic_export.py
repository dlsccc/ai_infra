#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torchvision.models import resnet50
from transformers import AutoModel, AutoTokenizer


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping.")
    return cfg


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def build_resnet50() -> torch.nn.Module:
    model = resnet50(weights=None)
    model.eval()
    return model


class BertExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, use_token_type_ids: bool) -> None:
        super().__init__()
        self.model = model
        self.use_token_type_ids = use_token_type_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_token_type_ids:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return outputs.last_hidden_state


def export_resnet(cfg: dict[str, Any]) -> dict[str, Any]:
    model = build_resnet50()

    model_name = str(cfg.get("model_name", "resnet50"))
    if model_name != "resnet50":
        raise ValueError(f"Unsupported model_name: {model_name}.")

    opset = int(cfg.get("opset_version", 17))
    dummy_shape = [int(x) for x in cfg.get("dummy_input_shape", [1, 3, 224, 224])]
    if len(dummy_shape) != 4:
        raise ValueError("dummy_input_shape must be [N, C, H, W].")

    dynamic_batch = bool(cfg.get("dynamic_batch", True))
    dynamic_hw = bool(cfg.get("dynamic_hw", False))

    onnx_path = Path(str(cfg.get("onnx_output_path", "artifacts/onnx/week02/resnet50_dynamic.onnx")))
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*dummy_shape, dtype=torch.float32)
    dynamic_axes: dict[str, dict[int, str]] = {}
    if dynamic_batch:
        dynamic_axes.setdefault("input", {})[0] = "batch"
        dynamic_axes.setdefault("logits", {})[0] = "batch"
    if dynamic_hw:
        dynamic_axes.setdefault("input", {})[2] = "height"
        dynamic_axes.setdefault("input", {})[3] = "width"

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes if dynamic_axes else None,
        )

    return {
        "mode": "week02_export",
        "model_family": "resnet",
        "model_name": model_name,
        "onnx_output_path": str(onnx_path),
        "opset_version": opset,
        "dummy_input_shape": dummy_shape,
        "dynamic_batch": dynamic_batch,
        "dynamic_hw": dynamic_hw,
        "dynamic_axes": dynamic_axes,
    }


def export_bert(cfg: dict[str, Any]) -> dict[str, Any]:
    model_name = str(cfg.get("model_name", "bert-base-uncased"))
    opset = int(cfg.get("opset_version", 17))
    dummy_batch = int(cfg.get("dummy_batch_size", 1))
    dummy_seq_len = int(cfg.get("dummy_seq_len", 128))
    dynamic_batch = bool(cfg.get("dynamic_batch", True))
    dynamic_seq = bool(cfg.get("dynamic_seq", True))
    onnx_path = Path(str(cfg.get("onnx_output_path", "artifacts/onnx/week02/bert_base_dynamic.onnx")))
    hf_cache_dir = Path(str(cfg.get("hf_cache_dir", ".cache/huggingface")))

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(hf_cache_dir))
    model = AutoModel.from_pretrained(model_name, cache_dir=str(hf_cache_dir))
    model.eval()

    use_token_type_ids = bool(cfg.get("use_token_type_ids", True))
    if use_token_type_ids and getattr(model.config, "type_vocab_size", 0) <= 1:
        use_token_type_ids = False

    vocab_size = int(tokenizer.vocab_size)
    input_ids = torch.randint(0, max(vocab_size, 1000), (dummy_batch, dummy_seq_len), dtype=torch.long)
    attention_mask = torch.ones((dummy_batch, dummy_seq_len), dtype=torch.long)
    token_type_ids = torch.zeros((dummy_batch, dummy_seq_len), dtype=torch.long)

    wrapper = BertExportWrapper(model, use_token_type_ids=use_token_type_ids)
    input_names = ["input_ids", "attention_mask"]
    model_inputs: tuple[torch.Tensor, ...] = (input_ids, attention_mask)

    if use_token_type_ids:
        input_names.append("token_type_ids")
        model_inputs = (input_ids, attention_mask, token_type_ids)

    dynamic_axes: dict[str, dict[int, str]] = {}
    for name in input_names:
        if dynamic_batch:
            dynamic_axes.setdefault(name, {})[0] = "batch"
        if dynamic_seq:
            dynamic_axes.setdefault(name, {})[1] = "seq_len"
    if dynamic_batch:
        dynamic_axes.setdefault("last_hidden_state", {})[0] = "batch"
    if dynamic_seq:
        dynamic_axes.setdefault("last_hidden_state", {})[1] = "seq_len"

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            model_inputs,
            str(onnx_path),
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic_axes if dynamic_axes else None,
        )

    return {
        "mode": "week02_export",
        "model_family": "bert",
        "model_name": model_name,
        "onnx_output_path": str(onnx_path),
        "hf_cache_dir": str(hf_cache_dir),
        "opset_version": opset,
        "dummy_batch_size": dummy_batch,
        "dummy_seq_len": dummy_seq_len,
        "use_token_type_ids": use_token_type_ids,
        "dynamic_batch": dynamic_batch,
        "dynamic_seq": dynamic_seq,
        "dynamic_axes": dynamic_axes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Week02 dynamic ONNX export")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_family = str(cfg.get("model_family", "resnet")).lower()

    if model_family == "resnet":
        result = export_resnet(cfg)
    elif model_family == "bert":
        result = export_bert(cfg)
    else:
        raise ValueError(f"Unsupported model_family: {model_family}. Use 'resnet' or 'bert'.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(str(cfg.get("output_dir", f"experiments/runs/week02/export/{model_family}")))
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dump_yaml(run_dir / "config_snapshot.yaml", cfg)
    with (run_dir / "export_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    summary_lines = [
        "# Week02 ONNX Export Summary",
        "",
        f"- run_id: `{run_id}`",
        f"- model_family: `{result['model_family']}`",
        f"- model_name: `{result['model_name']}`",
        f"- opset_version: `{result['opset_version']}`",
        f"- onnx_output_path: `{result['onnx_output_path']}`",
        "",
        "## Dynamic Axes",
        "```json",
        json.dumps(result.get("dynamic_axes", {}), indent=2, ensure_ascii=False),
        "```",
    ]
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[OK] ONNX export finished: {run_dir}")
    print(f"[OK] ONNX model saved: {result['onnx_output_path']}")


if __name__ == "__main__":
    main()
