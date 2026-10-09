"""Show, for a few examples, which concepts a trained model detected and how they led to its answer.

    python explain.py runs/cebab-cem-s0
    python explain.py runs/goemotions-cbm-s0 --n 5 --top 6
    python explain.py runs/cebab-cbm-r16-s0 --split test --ids 0 1 2

For each example: the text, the true and predicted answers, every concept's true label and
predicted probability, and how much each concept pushed the model towards its answer.

The head is linear, so the score of each answer is exactly a sum of one term per concept
(+ one for the residual, if any, + a constant). A concept's "push" is its term for the
predicted answer minus the average of its terms over all answers: > 0 means "this concept
pushed towards the predicted answer", < 0 "away from it". The pushes add up to the
predicted answer's lead over the average score.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from concept_datasets import load_meta, load_split
from concept_models.features import get_features
from concept_models.models import load_model


def contributions(model, x: torch.Tensor) -> tuple[torch.Tensor, list[str] | None, torch.Tensor, dict]:
    """Per-concept terms of the task logits: (terms (n, k [+1], classes), extra names, bias, out)."""
    kind = model.config["kind"]
    if kind == "pyc-hyper":  # logits = W(x) p + b
        out = model(x)
        terms = out["concept_probs"].unsqueeze(-1) * out["concept_weights"].transpose(1, 2)
        return terms, None, model.task_bias.detach(), out
    if kind in ("pyc-cbm", "pyc-cem"):
        lin = model.predictor.predictor
        W, bias = lin.weight.detach(), lin.bias.detach()
        out = model(x)
        if kind == "pyc-cbm":
            return out["concept_probs"].unsqueeze(-1) * W.T.unsqueeze(0), None, bias, out
        z = out["z"].view(len(x), model.k, model.m)
        return torch.einsum("bkm,ckm->bkc", z, W.view(-1, model.k, model.m)), None, bias, out
    if len(model.head) != 1:
        raise ValueError("only a linear head can be split into per-concept terms")
    W, bias = model.head[0].weight.detach(), model.head[0].bias.detach()  # (classes, in)
    out = model(x)
    k = model.k
    if model.config["kind"] == "cem":
        m = model.m
        z = out["z"].view(len(x), k, m)
        terms = torch.einsum("bkm,ckm->bkc", z, W.view(-1, k, m))
        return terms, None, bias, out
    c = out["concept_logits"]
    head_in = torch.cat([torch.sigmoid(c) if model.head_input == "probs" else c, out["z"][:, k:]], 1)
    per_dim = head_in.unsqueeze(-1) * W.T.unsqueeze(0)  # (n, in, classes)
    terms = per_dim[:, :k]
    if model.r:
        terms = torch.cat([terms, per_dim[:, k:].sum(1, keepdim=True)], 1)
        return terms, [f"(residual: {model.r} unlabelled numbers)"], bias, out
    return terms, None, bias, out


def bar(p: float, width: int = 20) -> str:
    filled = round(p * width)
    return "█" * filled + "·" * (width - filled)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("run", help="run directory, e.g. runs/cebab-cem-s0")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=10, help="number of random examples")
    p.add_argument("--ids", type=int, nargs="*", help="explicit example indices (overrides --n)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top", type=int, default=0, help="only show the N concepts with the largest push (0 = all)")
    p.add_argument("--max_chars", type=int, default=400, help="truncate long texts")
    args = p.parse_args()

    run = Path(args.run)
    model = load_model(run / "model.pt")
    dataset = json.loads((run / "metrics.json").read_text())["config"]["dataset"]
    meta = load_meta(dataset)
    classes, concepts = meta["class_names"], meta["concept_names"]
    examples = load_split(dataset, args.split)
    feats = get_features(dataset, args.split, model.config["encoder"])
    assert len(examples) == len(feats["X"]), "cached features do not match the data file"

    ids = args.ids if args.ids else random.Random(args.seed).sample(range(len(examples)), args.n)
    with torch.no_grad():
        terms, extra_names, bias, out = contributions(model, feats["X"][ids])
    probs_c = torch.sigmoid(out["concept_logits"])
    probs_y = torch.softmax(out["logits"], 1)
    names = concepts + (extra_names or [])

    print(f"\nModel: {run.name}   dataset: {dataset}   split: {args.split}")
    n_ok = 0
    for row, i in enumerate(ids):
        ex = examples[i]
        pred = int(probs_y[row].argmax())
        n_ok += pred == ex["label"]
        push = terms[row, :, pred] - terms[row].mean(-1)  # (k [+1],)
        const = float(bias[pred] - bias.mean())
        text = ex["text"].replace("\n", " ")
        if len(text) > args.max_chars:
            text = text[: args.max_chars] + " …"

        print("\n" + "=" * 100)
        print(f"Example #{i}")
        print(f"  Text: {text}")
        mark = "✓ correct" if pred == ex["label"] else "✗ wrong"
        print(f"  True answer: {classes[ex['label']]}   Model's answer: {classes[pred]} "
              f"({probs_y[row, pred]:.0%} sure)   {mark}")
        print("  All answers: " + "  ".join(f"{c} {probs_y[row, j]:.0%}" for j, c in enumerate(classes)))

        order = list(range(len(concepts)))
        if args.top:
            order = sorted(order, key=lambda j: -abs(float(push[j])))[: args.top]
        print(f"\n  {'concept':<22}{'human label':<13}{'model thinks':<34}push towards '{classes[pred]}'")
        for j in order:
            truth = "yes" if ex["concepts"][j] else "no"
            pc = float(probs_c[row, j])
            agree = "" if (pc > 0.5) == bool(ex["concepts"][j]) else "  ← disagrees with human"
            print(f"  {concepts[j]:<22}{truth:<13}{bar(pc)} {pc:>5.0%}       {float(push[j]):+6.2f}{agree}")
        if extra_names:
            print(f"  {extra_names[0]:<69}{float(push[-1]):+6.2f}")
        if args.top and len(order) < len(concepts):
            rest = float(push[: len(concepts)].sum() - push[order].sum())
            print(f"  {'(other concepts)':<69}{rest:+6.2f}")
        print(f"  {'(constant, same for every text)':<69}{const:+6.2f}")
        print(f"  {'TOTAL lead of ' + repr(classes[pred]) + ' over the average answer':<69}"
              f"{float(push.sum()) + const:+6.2f}")
    print("\n" + "=" * 100)
    print(f"{n_ok}/{len(ids)} of these examples answered correctly.\n")


if __name__ == "__main__":
    main()
