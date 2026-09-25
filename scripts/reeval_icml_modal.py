"""Re-evaluate the ICML_2026 Modal checkpoints and dump per-example test arrays.

The Modal runs (volume `icml-2026-credal-results`) trained with Credal_Sets commit
532fd05 on branch ICML_2026, plus MAQA support files copied from Variational_CBM.
Run this from a worktree of that commit so model code and data processing match:

    python scripts/reeval_icml_modal.py --code-dir <worktree> \
        --ckpt-root checkpoints_from_modal/icml_2026 --out outputs/icml_2026_reeval

For each run it writes <out>/<run_id>/test_arrays.npz and reproduced_metrics.json.
Metric definitions follow HybridCredalCBMTrainer.evaluate: EU and AU are averaged
over concepts per example, error is 1[pred != label], correlations are Spearman.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

RUNS = {
    "cebab_3class_seed123_100ep": ("cebab3", "distilbert", 123),
    "cebab_3class_seed2024_100ep": ("cebab3", "distilbert", 2024),
    "cebab_3class_roberta_base_seed123_100ep": ("cebab3", "roberta-base", 123),
    "cebab_3class_roberta_base_seed2024_100ep": ("cebab3", "roberta-base", 2024),
    "hatexplain_seed123_100ep": ("hatexplain", "distilbert", 123),
    "hatexplain_seed2024_100ep": ("hatexplain", "distilbert", 2024),
    "goemotions_seed123_100ep": ("goemotions", "distilbert", 123),
    "goemotions_seed2024_100ep": ("goemotions", "distilbert", 2024),
    "maqa_seed123_100ep": ("maqa", "distilbert", 123),
    "maqa_seed2024_100ep": ("maqa", "distilbert", 2024),
    "cebab_3class_microsoft_deberta_v3_base_seed123_100ep": ("cebab3", "microsoft/deberta-v3-base", 123),
    "cebab_3class_microsoft_deberta_v3_base_seed2024_100ep": ("cebab3", "microsoft/deberta-v3-base", 2024),
    "cebab_3class_answerdotai_ModernBERT_base_seed123_100ep": ("cebab3", "answerdotai/ModernBERT-base", 123),
    "cebab_3class_answerdotai_ModernBERT_base_seed2024_100ep": ("cebab3", "answerdotai/ModernBERT-base", 2024),
}

# Retraining campaign (fixed CEBaB labels, seed 42 added, ablations, AmbigQA*).
for _seed in (42, 123, 2024):
    for _tag in ("100ep_fixed", "100ep_fixed_noale", "100ep_fixed_decorr5"):
        RUNS[f"cebab_3class_distilbert_seed{_seed}_{_tag}"] = ("cebab3", "distilbert", _seed)
    RUNS[f"cebab_3class_roberta_base_seed{_seed}_100ep_fixed"] = ("cebab3", "roberta-base", _seed)
    RUNS[f"ambigqa_distilbert_seed{_seed}_100ep"] = ("ambigqa", "distilbert", _seed)
RUNS["hatexplain_distilbert_seed42_100ep"] = ("hatexplain", "distilbert", 42)
RUNS["goemotions_distilbert_seed42_100ep"] = ("goemotions", "distilbert", 42)
RUNS["maqa_distilbert_seed42_100ep"] = ("maqa", "distilbert", 42)

ENTROPY_KEYS = ("annotator_entropy", "_annotator_entropy", "_rating_entropy")


def seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cebab3_loaders(tokenizer):
    """Mirror modal_icml_2026_multiseed.train_one for cebab_three_class=True."""
    from datasets import load_dataset

    import load_cebab_direct as cebab

    cebab = importlib.reload(cebab)

    def load_cebab_paper(split_ids_path=None, include_edits=False, seed=42):
        del split_ids_path, include_edits, seed
        ds = load_dataset("CEBaB/CEBaB")
        return {
            "train": list(ds["train_inclusive"]),
            "validation": list(ds["validation"]),
            "test": list(ds["test"]),
        }

    original_process = cebab.process_cebab_raw

    def process_cebab_ternary(raw_data):
        raw_data = [
            item for item in raw_data
            if str(item.get("review_majority", "")).strip() in {"1", "2", "3", "4", "5"}
        ]
        processed = original_process(raw_data)
        for item in processed:
            rating = item["label"] + 1
            item["label"] = 0 if rating <= 2 else (1 if rating == 3 else 2)
        return processed

    cebab.load_cebab = load_cebab_paper
    cebab.process_cebab_raw = process_cebab_ternary
    _, _, test_loader, _, _ = cebab.get_cebab_dataloaders(
        tokenizer=tokenizer, batch_size=8, max_length=256, num_workers=0, include_edits=False
    )
    return test_loader


def cbm_eval(run_id, kind, encoder, seed, ckpt_path, device):
    import torch
    from transformers import AutoTokenizer

    import main_train_hybrid_multi_dataset as experiment
    from VCBM import HybridCredalCBM, HybridCredalConfig

    encoder_name = experiment.expand_encoder_name(encoder)
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    dataset = "cebab" if kind == "cebab3" else kind
    cfg = experiment.DATASET_CONFIGS[dataset]

    if kind == "cebab3":
        test_loader = cebab3_loaders(tokenizer)
        num_classes = 3
    elif cfg.get("use_multi_loader"):
        ds_config = experiment.DatasetConfig(
            max_length=cfg["loader_kwargs"]["max_length"],
            batch_size=cfg["loader_kwargs"]["batch_size"],
            tokenizer_name=encoder_name,
        )
        _, _, test_loader, _, _ = experiment.load_dataset_splits(dataset_name=dataset, config=ds_config)
        num_classes = cfg["num_classes"]
    else:
        _, _, test_loader, _, _ = cfg["data_loader"](tokenizer=tokenizer, **cfg["loader_kwargs"])
        num_classes = cfg["num_classes"]

    model_config = HybridCredalConfig(
        encoder_name=encoder_name,
        freeze_encoder=True,
        num_concepts=cfg["num_concepts"],
        concept_names=cfg["concept_names"],
        num_classes=num_classes,
        prior_sigma=cfg["prior_sigma"],
        error_scale=cfg["error_scale"],
    )
    # transformers>=5 may load encoder weights in half precision; training ran in fp32.
    model = HybridCredalCBM(model_config).float().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    seed_all(seed)
    out = {k: [] for k in ("y_true", "y_pred", "probs", "eu", "au", "concept_probs", "concept_labels", "H")}
    with torch.inference_mode():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            out["y_true"].append(batch["labels"].numpy())
            out["y_pred"].append(outputs["predictions"].cpu().numpy())
            out["probs"].append(outputs["probs"].cpu().numpy())
            out["eu"].append(outputs["epistemic"].cpu().numpy())
            out["au"].append(outputs["aleatoric"].cpu().numpy())
            out["concept_probs"].append(outputs["concept_probs"].cpu().numpy())
            if "concept_labels" in batch:
                out["concept_labels"].append(batch["concept_labels"].numpy())
            for key in ENTROPY_KEYS:
                if key in batch:
                    out["H"].append(np.asarray(batch[key], dtype=np.float32))
                    break

    n_mc = int(os.environ.get("REEVAL_MC", "0"))
    if n_mc:
        # MC dropout baseline: frozen encoder stays deterministic; dropout in the
        # trainable heads is sampled n_mc times.
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout) and not name.startswith("encoder"):
                module.train()
        torch.manual_seed(seed)
        passes = []
        with torch.inference_mode():
            for _ in range(n_mc):
                probs = [model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))["probs"].cpu().numpy()
                         for b in test_loader]
                passes.append(np.concatenate(probs))
        model.eval()
        mc = np.stack(passes)                       # [T, N, J]
        mean = mc.mean(0)
        ent = lambda p: -(p * np.log(p + 1e-12)).sum(-1)
        out["mc_probs"] = [mean]
        out["mc_entropy"] = [ent(mean)]
        out["mc_mutual_info"] = [ent(mean) - ent(mc).mean(0)]
    arrays = {k: np.concatenate(v) for k, v in out.items() if v}
    meta = {
        "run_id": run_id,
        "kind": kind,
        "encoder": encoder_name,
        "seed": seed,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_rule": "best validation task accuracy (HybridCredalCBMTrainer.fit)",
        "lambda_d": 0.0,
        "H_source": next((k for k in ENTROPY_KEYS if k in batch), None),
    }
    return arrays, meta


def maqa_eval(run_id, seed, ckpt_path, device, subset=""):
    import torch
    from torch.utils.data import DataLoader

    import main_train_hybrid_multi_dataset as experiment
    from load_maqa_real import load_combined_maqa_ambigqa
    from maqa_credal_model import MAQADataset, maqa_collate_fn
    from v7b_complete_integration import CredalMAQA_V7b

    encoder_name = "distilbert-base-uncased"
    encoder, tokenizer = experiment.load_encoder_with_quantization(encoder_name, "none", device)
    raw = load_combined_maqa_ambigqa()
    if subset:
        # Same split as MAQA*, restricted to one source (as in training).
        raw = {k: [x for x in v if x.get("dataset") == subset] for k, v in raw.items()}
    data = {}
    for split in ("train", "validation", "test"):
        processed = []
        for item in raw[split]:
            p = np.array(item.get("probabilities", []), dtype=np.float32)
            if p.size == 0 or p.sum() <= 0:
                continue
            p = p / p.sum()
            entropy = -np.sum(p * np.log(p + 1e-10))
            processed.append({
                "text": item["question"],
                "p_star": p.tolist(),
                "entropy": float(entropy),
                "ambiguity_level": 0 if entropy < 0.1 else 2,
                "num_answers": len(p),
                "answers": item.get("answers", []) or [0] * len(p),
                "dominant_answer_idx": int(np.argmax(p)),
            })
        data[split] = processed
    max_answers = max(d["num_answers"] for split in data.values() for d in split)

    model = CredalMAQA_V7b(
        encoder=encoder,
        hidden_size=768,
        num_answers=max_answers,
        projection_dim=experiment.DATASET_CONFIGS["maqa"]["projection_dim"],
        dropout=0.1,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_loader = DataLoader(
        MAQADataset(data["test"], tokenizer, max_length=128, use_paired=False),
        batch_size=16, shuffle=False, collate_fn=maqa_collate_fn,
    )
    seed_all(seed)
    out = {k: [] for k in ("eu", "au_with_H_input", "au_no_H_input", "H", "y_pred", "y_true", "maxprob")}
    items = iter(data["test"])
    with torch.inference_mode():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            entropy = batch["entropy"].to(device)
            leaky = model(ids, mask, entropy=entropy)
            clean = model(ids, mask, entropy=None)
            out["eu"].append(clean.sigma_epi.cpu().numpy())
            out["au_with_H_input"].append(leaky.sigma_ale.cpu().numpy())
            out["au_no_H_input"].append(clean.sigma_ale.cpu().numpy())
            out["H"].append(entropy.cpu().numpy())
            mu = clean.mu.cpu().numpy()
            for row in mu:
                item = next(items)
                logits = row[: item["num_answers"]]
                out["y_pred"].append(int(np.argmax(logits)))
                out["y_true"].append(item["dominant_answer_idx"])
                # Softmax over the valid answers, as in the v7b answer loss.
                z = np.exp(logits - logits.max())
                out["maxprob"].append(float(z.max() / z.sum()))

    arrays = {k: np.asarray(np.concatenate(v) if isinstance(v[0], np.ndarray) else v) for k, v in out.items()}
    meta = {
        "run_id": run_id,
        "kind": "maqa",
        "qa_subset": subset or None,
        "encoder": encoder_name,
        "seed": seed,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_rule": "lowest validation loss (train_maqa_model)",
        "loss_version": "v7b",
        "lambda_decorr": 5.0,  # CONFIG_V7B in maqa_credal_loss_v7b_fixed.py at 532fd05
        "note": "sigma_ale head takes ground-truth entropy as input; au_no_H_input zeroes it",
        "accuracy_definition": "argmax of answer logits over valid answers == majority answer",
    }
    return arrays, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", default="")
    args = parser.parse_args()

    ckpt_root = Path(args.ckpt_root).resolve()
    out_root = Path(args.out).resolve()
    os.chdir(args.code_dir)
    sys.path.insert(0, str(Path(args.code_dir).resolve()))

    import torch

    device = os.environ.get("REEVAL_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )  # REEVAL_DEVICE=cpu works around an MPS matmul assertion with DeBERTa-v3
    selected = args.runs.split(",") if args.runs else list(RUNS)
    for run_id in selected:
        kind, encoder, seed = RUNS[run_id]
        ckpt_path = ckpt_root / run_id / "best_model.pt"
        if not ckpt_path.exists():
            print(f"skip {run_id}: no checkpoint")
            continue
        print(f"== {run_id} on {device}", flush=True)
        if kind in ("maqa", "ambigqa"):
            arrays, meta = maqa_eval(run_id, seed, ckpt_path, device, subset="ambigqa" if kind == "ambigqa" else "")
        else:
            arrays, meta = cbm_eval(run_id, kind, encoder, seed, ckpt_path, device)
        run_out = out_root / run_id
        run_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(run_out / "test_arrays.npz", **arrays)
        meta["n_test"] = int(len(arrays["y_true"]))
        meta["accuracy"] = float((arrays["y_pred"] == arrays["y_true"]).mean())
        (run_out / "run_manifest.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
