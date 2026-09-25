"""Pinned public checkpoints for six approximate A1 quality signals.

Five checkpoints correspond to the released model families. An exact public
WanJuan fluency checkpoint was not located; CoLA is an explicitly weaker proxy
whose A1 calibration must pass before use. No model is downloaded on import.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


MODEL_REGISTRY = {
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu-classifier", "284663cbb2dabf9bda30d8f8cc49601251ee1631", 1, "public_family"),
    "modernbert_cleanliness": ("opendatalab/meta-rater-cleanliness-rating", "4403a9535d47cbc7cc99de26b25099335fe2d9b6", 6, "same_checkpoint_family"),
    "modernbert_readability": ("opendatalab/meta-rater-readability-rating", "5bfbee1110869ddcbf23447354a7311374784952", 6, "same_checkpoint_family"),
    "modernbert_reasoning": ("opendatalab/meta-rater-reasoning-rating", "0072a9a83971eb4af6d689dfc64f8f203c45b398", 6, "same_checkpoint_family"),
    "modernbert_professionalism": ("opendatalab/meta-rater-professionalism-rating", "fc91d4be35fc91de3c65654bb59655ec533a1f61", 6, "same_checkpoint_family"),
    "fluency_en": ("textattack/roberta-base-CoLA", "3ccf3a400f2fa75ff257eac171047603ffbe84f1", 2, "grammar_acceptability_proxy_not_wanjuan"),
}


def infer_logits(texts: list[str], field: str, cache_dir: Path, *, batch_size: int = 8,
                 max_length: int = 512, device: str | None = None) -> np.ndarray:
    """Infer one field; cache is keyed by the exact text sequence and settings."""
    import hashlib

    if field not in MODEL_REGISTRY:
        raise ValueError(f"Unknown public quality field: {field}")
    model_id, revision, width, _ = MODEL_REGISTRY[field]
    digest = hashlib.sha256()
    digest.update(json.dumps([field, revision, max_length, texts], ensure_ascii=False).encode("utf-8"))
    key = digest.hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{field}_{key[:16]}.npy"
    partial_file = cache_dir / f"{field}_{key[:16]}.partial.npy"
    if cache_file.exists():
        values = np.load(cache_file, allow_pickle=False)
        if values.shape != (len(texts), width):
            raise ValueError(f"Cached {field} logits have wrong shape")
        return values
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Public model inference needs Code's optional `quality-models` dependencies: torch and transformers") from exc
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Try the pinned local snapshot first. Transformers otherwise probes the Hub
    # for a safetensors file even when a usable PyTorch checkpoint is cached.
    # The CoLA proxy is distributed here as pytorch_model.bin, not safetensors.
    model_options = {"use_safetensors": False} if field == "fluency_en" else {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                                   local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision, local_files_only=True, **model_options)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision, **model_options)
    model = model.to(device).eval()
    if model.config.num_labels != width:
        raise ValueError(f"{field}: expected {width} logits, got {model.config.num_labels}")
    chunks = []
    done = 0
    if partial_file.exists():
        try:
            previous = np.load(partial_file, allow_pickle=False)
            if previous.ndim == 2 and previous.shape[1] == width and previous.shape[0] <= len(texts):
                chunks.append(previous)
                done = len(previous)
        except (OSError, ValueError):
            pass
    with torch.inference_mode():
        for start in range(done, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                              max_length=max_length, return_tensors="pt")
            logits = model(**{key: val.to(device) for key, val in batch.items()}).logits
            chunks.append(logits.float().cpu().numpy().reshape(-1, width))
            if (start // batch_size) % 20 == 0 or start + batch_size >= len(texts):
                np.save(partial_file, np.concatenate(chunks, axis=0))
                print(f"{field}: {min(start + batch_size, len(texts))}/{len(texts)} texts", flush=True)
    values = np.concatenate(chunks, axis=0) if chunks else np.empty((0, width))
    if values.shape != (len(texts), width) or not np.isfinite(values).all():
        raise ValueError(f"{field}: nonfinite or incomplete model output")
    np.save(cache_file, values)
    partial_file.unlink(missing_ok=True)
    return values
