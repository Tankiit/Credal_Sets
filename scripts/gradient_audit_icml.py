"""Loss-by-parameter-block gradient audit for an ICML_2026 CBM checkpoint.

For each loss term returned by HybridCredalCBM._compute_losses, differentiate
with respect to every parameter block and record whether the graph has no
path (None), a numerically zero gradient, or a nonzero gradient norm.

    python scripts/gradient_audit_icml.py --code-dir <worktree 532fd05> \
        --ckpt checkpoints_from_modal/icml_2026/cebab_3class_seed123_100ep/best_model.pt \
        --out outputs/icml_2026_reeval/gradient_audit.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

BLOCKS = OrderedDict([
    ("W_mu", "projection.W_concept"),
    ("W_epi", "projection.W_epi"),
    ("W_ale", "projection.W_ale"),
    ("mean head", "credal_head.mu_net"),
    ("epi head", "credal_head.log_sigma_net"),
    ("ale head", "aleatoric_head"),
    ("task head", "task_classifier"),
])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batches", type=int, default=20)
    args = parser.parse_args()
    ckpt_path, out_path = Path(args.ckpt).resolve(), Path(args.out).resolve()
    os.chdir(args.code_dir)
    sys.path.insert(0, str(Path(args.code_dir).resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    from transformers import AutoTokenizer

    from reeval_icml_modal import cebab3_loaders, seed_all
    from VCBM import HybridCredalCBM, HybridCredalConfig

    seed_all(123)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    loader = cebab3_loaders(tokenizer)
    config = HybridCredalConfig(
        encoder_name="distilbert-base-uncased", freeze_encoder=True, num_concepts=4,
        concept_names=["food", "service", "ambiance", "noise"], num_classes=3,
        prior_sigma=0.5, error_scale=2.0,
    )
    model = HybridCredalCBM(config)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"])
    model.train()

    blocks = {name: [p for n, p in model.named_parameters() if n.startswith(prefix)] for name, prefix in BLOCKS.items()}
    encoder_trainable = sum(p.requires_grad for n, p in model.named_parameters() if n.startswith("encoder."))
    weights = {
        "task_loss": 1.0, "concept_bce": config.concept_weight,
        "error_supervision": config.error_supervision_weight, "credal_kl": config.kl_weight,
        "aleatoric_loss": config.aleatoric_weight, "aleatoric_unknown": 0.1 * config.aleatoric_weight,
        "orth_penalty": config.orth_weight,
    }
    sums: dict = {}
    for i, batch in enumerate(loader):
        if i == args.batches:
            break
        result = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"],
            concept_labels=batch["concept_labels"], annotator_entropy=batch["annotator_entropy"],
        )
        losses = {k: v for k, v in result.items() if k in weights or k == "loss"}
        if not losses and "losses" in result:
            losses = {k: v for k, v in result["losses"].items() if k in weights or k == "loss"}
        for lname, lval in losses.items():
            if not torch.is_tensor(lval) or not lval.requires_grad:
                continue
            for bname, params in blocks.items():
                grads = torch.autograd.grad(lval, params, retain_graph=True, allow_unused=True)
                present = [g for g in grads if g is not None]
                entry = sums.setdefault(lname, {}).setdefault(bname, {"path": False, "sq": 0.0})
                if present:
                    entry["path"] = True
                    entry["sq"] += float(sum((g ** 2).sum() for g in present))
    n = min(args.batches, len(loader))
    report = {
        "checkpoint": str(ckpt_path), "batches": n, "encoder_trainable_tensors": int(encoder_trainable),
        "loss_weights": weights,
        "grad_norm_rms_over_batches": {
            l: {b: (None if not e["path"] else (e["sq"] / n) ** 0.5) for b, e in d.items()} for l, d in sums.items()
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"{'loss':20s}" + "".join(f"{b:>11s}" for b in BLOCKS))
    for l, d in report["grad_norm_rms_over_batches"].items():
        print(f"{l:20s}" + "".join(f"{'none' if v is None else f'{v:.2e}':>11s}" for v in d.values()))
    print("encoder trainable tensors:", encoder_trainable)


if __name__ == "__main__":
    main()
